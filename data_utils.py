import os
import glob
import numpy as np
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
