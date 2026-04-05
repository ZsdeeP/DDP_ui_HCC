import sys

sys.path.append("../")
sys.path.append("../../")


import re
import os
import glob
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import argparse
from pathlib import Path

import pydicom
import monai
from monai.data.dataset import Dataset
from monai.data.dataloader import DataLoader
from monai.metrics.meandice import DiceMetric
from monai.inferers.utils import sliding_window_inference

from typing import List, Union, Tuple
#from UNET_plus_plus.archs import NestedUNet, UNet

from monai.transforms.transform import Transform
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (
    CropForegroundd,
    SpatialPadd,
)
from monai.transforms.utility.dictionary import EnsureTyped
from monai.transforms.post.dictionary import Invertd

class_tissue_dict = {
    1: "liver",
    2: "tumour",
    3: "bloodvessels",
    4: "abdominalaorta"
}

from monai.transforms.io.array import LoadImage

import sys

sys.path.append("../")
from argparse import Namespace
from torch.utils.data._utils.collate import default_collate

import pydicom
from pydicom_seg import reader #also create nifti loader

def affine2d(ds):
    F11, F21, F31 = ds.ImageOrientationPatient[3:]
    F12, F22, F32 = ds.ImageOrientationPatient[:3]

    dr, dc = ds.PixelSpacing
    Sx, Sy, Sz = ds.ImagePositionPatient

    return np.array(
        [
            [F11 * dr, F12 * dc, 0, Sx],
            [F21 * dr, F22 * dc, 0, Sy],
            [F31 * dr, F32 * dc, 0, Sz],
            [0, 0, 0, 1]
        ]
    )

def apply_window(image, window_center=30, window_width=150):
    """
    Apply windowing to a CT image to enhance specific tissue visibility

    Args:
        image: Input CT image in Hounsfield Units
        window_center: Center of the window in HU
        window_width: Width of the window in HU

    Returns:
        Windowed image normalized to [0, 1]
    """
    # Calculate window boundaries
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2

    # Apply windowing
    windowed_image = np.clip(image, window_min, window_max)

    # Normalize to [0, 1]
    windowed_image = (windowed_image - window_min) / (window_max - window_min)

    return windowed_image

def read_dicom_series(directory):
    """
    Read a series of DICOM files and stack them into a 3D volume
    Sorts slices based on ImagePositionPatient z-coordinate for correct ordering
    """
    # Find all DICOM files in the directory
    # First try to find DICOM files starting with "1-"
    dicom_files = glob.glob(os.path.join(directory, "1-*.dcm"))

    # If no files with "1-" prefix are found, look for files starting with "2-"
    if not dicom_files:
        dicom_files = glob.glob(os.path.join(directory, "2-*.dcm"))

    # If still no files found, get all DICOM files
    if not dicom_files:
        dicom_files = glob.glob(os.path.join(directory, "*.dcm"))

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {directory}")

    # Read all files first to get positions for sorting
    slices = []
    for file_path in dicom_files:
        dicom = pydicom.dcmread(file_path)
        # Ensure the file has ImagePositionPatient tag
        if hasattr(dicom, 'ImagePositionPatient'):
            slices.append((file_path, dicom))
        else:
            # If some files don't have the position, fall back to regular sorting
            slices = [(file_path, pydicom.dcmread(file_path)) for file_path in dicom_files]
            print("Warning: Some files missing ImagePositionPatient. Using filename sorting.")
            break

    # Sort by ImagePositionPatient's z-coordinate (third value)
    if all(hasattr(s[1], 'ImagePositionPatient') for s in slices):
        slices.sort(key=lambda s: float(s[1].ImagePositionPatient[2]))
    else:
        # Fall back to filename sorting if ImagePositionPatient is not consistently available
        slices.sort(key=lambda s: s[0])

    # Read the first file to get metadata
    ref_dicom = slices[0][1]

    # Read Affine transformation matrix
    affine_matrix = affine2d(ref_dicom)

    # Extract patient information
    patient_id = ref_dicom.PatientID if hasattr(ref_dicom, 'PatientID') else "Unknown"

    #Extract image position and image orientation

    image_position = np.array(ref_dicom.ImagePositionPatient)
    image_orientation = np.array(ref_dicom.ImageOrientationPatient)


    # Get slice thickness and pixel spacing
    slice_thickness = float(ref_dicom.SliceThickness) if hasattr(ref_dicom, 'SliceThickness') else 1.0
    pixel_spacing = ref_dicom.PixelSpacing if hasattr(ref_dicom, 'PixelSpacing') else [1.0, 1.0]

    # Initialize 3D volume
    num_slices = len(slices)
    rows, cols = ref_dicom.pixel_array.shape
    volume = np.zeros((num_slices, rows, cols), dtype=np.float32)

    # Read all slices in the sorted order
    for i, (_, dicom) in enumerate(slices):
        pixel_array = dicom.pixel_array

        # Apply rescale slope and intercept if available
        if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
            pixel_array = pixel_array * float(dicom.RescaleSlope) + float(dicom.RescaleIntercept)

        volume[i, :, :] = pixel_array

    # Return volume and metadata
    metadata = {
        'patient_id': patient_id,
        'slice_thickness': slice_thickness,
        'pixel_spacing': pixel_spacing,
        'shape': volume.shape,
        'image_position': image_position,
        'image_orientation': image_orientation,

    }

    return volume, metadata

def read_segmentation_dicom(file_path, class_number=None):
    """
    Read segmentation DICOM file and extract mask
    """
    dcm_seg = pydicom.dcmread(file_path)
    seg_reader = reader.SegmentReader()
    # Read the segmentation data
    result = seg_reader.read(dcm_seg)
    segments_info = result.segment_infos
    if class_number is not None:
        seg_array = result.segment_data(class_number)
    else:
        segments = []
        for segment_number, _ in segments_info.items():
            segments.append(result.segment_data(segment_number))
        seg_array = np.stack(segments, axis=0)
    return seg_array.astype(np.float32)



def collate_with_metadata(batch):
    #sample could be a list of dicts or just a dict

    if isinstance(batch[0], dict):
        flat_batch = [batch]
    elif isinstance(batch[0], list) and isinstance(batch[0][0], dict):
        flat_batch = batch
    else:
        raise ValueError("Each sample must either be a dict or a list of dicts")
    images = default_collate([item[0]['image'] for item in flat_batch])
    labels = default_collate([item[0]['label'] for item in flat_batch])
    metadata = [item[0]['metadata'] for item in flat_batch] # Keep metadata as a list
    return {'image': images, 'label': labels, 'metadata': metadata}

#mask other classes

class DummyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class HCCDataset(Dataset):
    def __init__(
        self, metadata_path, root_dir, transform=None, apply_liver_window=False, class_number=1,
    ):
        """
        Custom dataset for HCC segmentation

        Args:
            metadata_path: Path to metadata.csv file
            root_dir: Root directory containing the data
            transform: MONAI transforms to apply
            apply_liver_window: Whether to apply liver window to CT images
        """
        self.metadata_df = pd.read_csv(metadata_path)
        self.root_dir = root_dir
        self.transform = transform
        self.apply_liver_window = apply_liver_window
        self.class_number = class_number
        self.img_loader = LoadImage(image_only=True, ensure_channel_first=True)

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.metadata_df.iloc[idx]
        patient_id = row["PATIENT_ID"]

        scan_folder = self.root_dir / row["SCAN_FOLDER"]  # os.path.join(self.root_dir, row["SCAN_FOLDER"])
        seg_file = self.root_dir / row["SEGMENTATION_FILE"]  # os.path.join(self.root_dir, row["SEGMENTATION_FILE"])

        # Load the CT volumeTrue
        volume, metadata = read_dicom_series(scan_folder)

        # Apply liver window if requested
        if self.apply_liver_window:
            volume = apply_window(volume, window_center=150, window_width=250) #worked well for blood vessels; originally was 40, 140

        # Load the segmentation mask
        segmentation = read_segmentation_dicom(seg_file, class_number=self.class_number)

        other_classes = [i for i in range(1, 5) if i != self.class_number]
        segmentation = np.transpose(segmentation, (1, 2, 0))
        not_seg = np.zeros_like(segmentation)

        for i in other_classes:
            segi_path = Path(f"intermediate_masks/{class_tissue_dict[i]}/{patient_id}.nii.gz")
            if segi_path.is_file():
                segi = np.transpose(self.img_loader(segi_path).squeeze(0), (1, 2, 0))
                not_seg = not_seg + segi

        #not_seg = seg1 + seg2 + seg4

        # Convert volume from (D,H,W) to (H,W,D)
        volume = np.transpose(volume, (1, 2, 0))

        #if self.class_number == 3:
        volume = volume * (not_seg == 0) #mask out other organs (best case result for blood vessels)


        # Convert to tensors
        sample = {
            "image": torch.from_numpy(volume).unsqueeze(0),  # Add channel dimension
            "label": torch.from_numpy(segmentation).unsqueeze(0),  # Add channel dimension
            "metadata": metadata,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample



work_dir = Path(__file__).resolve().parent
model_dir = work_dir / "models" / "Segmentation"



def get_dice_score(prev_masks, gt3D):

        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)

            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = (prev_masks > 0.5)
        true_masks = (gt3D > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list) / len(dice_list))


def save_val_results(result, class_, metadata: dict|None):
    # Ideally the output is a dicom so the information about the location of each segmentation is preserved
    # without this, the nifti file cannot be used to rule out regions containing blood vessels directly.
    
    if isinstance(result, torch.Tensor):
        result = result.cpu().permute(0, 1, 4, 2, 3).numpy()[0][0]
    elif isinstance(result, np.ndarray) and result.ndim == 5:
        result = result[0, 0]

    if metadata:
        sear = re.search(r"\w+", str(metadata["patient_id"]))
        patient_id = sear.group() if sear else "e20" #dummy value
        image_position = metadata["image_position"]
        image_orientation = metadata["image_orientation"].squeeze()
        slice_thickness = list(metadata["slice_thickness"])
        pixel_spacing = list(metadata["pixel_spacing"])
        shape = np.array(metadata["shape"])

        #save result to the specified directory
        #file_dir = Path(f"{output_dir}/{class_tissue_dict[class_]}")
        #file_dir.mkdir(parents=True, exist_ok=True)

        row_cosines = np.array(image_orientation[0:3])
        col_cosines = np.array(image_orientation[3:6])
        slice_cosines = np.cross(row_cosines, col_cosines)

        if not (row_cosines.shape == col_cosines.shape == slice_cosines.shape and row_cosines.ndim == 1):
            raise ValueError("row_cosines, col_cosines, and slice_cosines must be 1D arrays of the same length.")

        direction = np.stack([row_cosines, col_cosines, slice_cosines], axis=1)
        spacing = np.array([pixel_spacing[0], pixel_spacing[1], slice_thickness[0]])# (3, 1)

        affine = np.eye(4)
        affine[:3, :3] = direction * spacing
        affine[:3, 3] = image_position

    #file_path = file_dir / f"{patient_id}.nii.gz"
    
    else:
        affine = np.eye(4)

    return result.astype(np.float32), affine
    #nifti_img = nib.Nifti1Image(result.astype(np.float32), affine=affine)
    #nib.save(nifti_img, file_path)
    #print(f"Saved NIfTI to {file_path}")


# function to load a list of indices, each corresponding to a scan in the metadata.csv --> output volumes and labels
# function to load a trained model from a checkpoint file
# run inference on the loaded scans
# save the output volumes as NIfTI files

def load_metadata_indices(indices: List[int], master_transforms: Compose, root_dir: Path, class_number: int) -> DataLoader:
    metadata_file = "data/val.csv"
    metadata_df = pd.read_csv(metadata_file)
    selected_rows = metadata_df.iloc[indices]

    os.makedirs(f"temp_data", exist_ok=True)
    write_path = f"temp_data/temp_metadata.csv"

    selected_rows.to_csv(write_path, index=False)


    sub_dataset = HCCDataset(write_path,
                            root_dir,
                            transform=master_transforms,
                            apply_liver_window=True,
                            class_number=class_number)
    dataloader = DataLoader(sub_dataset, batch_size=1, shuffle=False)
    return dataloader

def load_trained_model(model_name: str, device: torch.device, class_num: int) -> torch.nn.Module| None:
    
    if model_name not in ["NestedUNet", "UNet"]:
        model_name = "model1" #dummy value for testing
        return None
    elif model_name == "NestedUNet":
        from UNET_plus_plus.archs import NestedUNet
        model = NestedUNet(input_channels=1, num_classes=1).to(device)
    else:
        from UNET_plus_plus.archs import UNet
        model = UNet(input_channels=1, num_classes=1).to(device)
    
    checkpoint_path = model_dir / model_name / str(class_num)
    model_dict = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(model_dict['model'], strict=True)
    return model

def run_inference(dataloader: DataLoader, model: torch.nn.Module, device: torch.device, patch_size: List[int], post_transforms: Transform) -> tuple[list[np.ndarray], list[dict], float]:
    all_outputs = []
    all_metadata = []
    #dice_metric = DiceMetric(reduction="mean", get_not_nans=False)
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            metadata = batch_data["metadata"]
            outputs = sliding_window_inference(inputs=inputs, roi_size=patch_size, sw_batch_size=1, predictor=model, overlap=0.75)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            elif isinstance(outputs, dict): 
                outputs = outputs["pred"]  # Adjust the key if necessary    
            outputs = torch.sigmoid(outputs)
            dice_score += get_dice_score(outputs, labels)
            outputs = outputs > 0.5
            #convert outputs to dict for post transforms
            outputs_dict = {"image": inputs, "label": outputs}
            outputs = post_transforms(outputs_dict)
            #dice_metric(y_pred=outputs, y=labels)
            all_outputs.append(outputs["label"].cpu().permute(0, 1, 4, 2, 3).numpy())
            all_metadata.append(metadata)
        dice_score /= len(dataloader)
        #dice_score = dice_metric.aggregate().item()
        #dice_metric.reset()
    return all_outputs, all_metadata, dice_score

def save_outputs_as_nifti(outputs: List[np.ndarray], all_metadata: List[dict], output_dir: Path, class_number:int):
    os.makedirs(output_dir, exist_ok=True)
    for i, output in enumerate(outputs):
        volume = output[0, 0]  # Assuming batch size 1 and single channel
        metadata = all_metadata[i]
        output_path = os.path.join(output_dir, f"output_{i}.nii.gz")
        save_val_results(volume, class_number, metadata)


