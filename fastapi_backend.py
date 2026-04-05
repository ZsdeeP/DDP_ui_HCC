import nibabel as nib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from pathlib import Path

import torch
from infer_model import DummyDataset

import pydicom as dicom
import monai
from typing import List, Union, Tuple
from monai.data.dataloader import DataLoader
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (
    CropForegroundd,
    SpatialPadd,
)
from monai.transforms.utility.dictionary import EnsureTyped
from monai.transforms.post.dictionary import Invertd

from infer_model import load_metadata_indices, load_trained_model, run_inference, save_val_results, collate_with_metadata

class ModelSelect(BaseModel):
    models: str

ROOT_DIR = Path(__file__).resolve().parent #PLACEHOLDER
PATCH_SIZE = [64, 64, 64] #FROZEN

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

static_path = current_dir / "static"
templates_path = static_path / "templates"
data_path = current_dir / "data"

app = FastAPI()
app.mount("/static", StaticFiles(directory=static_path), name="static")
app.mount("/data", StaticFiles(directory=data_path), name="data")

#templates = Jinja2Templates(directory=templates_path)

def get_image_path(case_id: str, suffix: str = "", category: int = 0, model_name: str = "model1") -> str:
    cat_dict = {0: "base_images", 1: "ground_truths", 2: "segmentations"}

    if category == 2:
        pred_path = data_path / "segmentations" / model_name / f"{case_id}_pred.nii.gz"
        if pred_path.exists():
            return f"/data/segmentations/{model_name}/{case_id}_pred.nii.gz"
        else:
            raise FileNotFoundError(f"No segmentation file found for {case_id} with model {model_name}")

    nii_path = data_path / cat_dict[category] / f"{case_id}{suffix}.nii.gz"
    dcm_path = data_path / cat_dict[category] / f"{case_id}{suffix}.dcm"
    if nii_path.exists():
        return f"/data/{cat_dict[category]}/{case_id}{suffix}.nii.gz"
    elif dcm_path.exists():
        return f"/data/{cat_dict[category]}/{case_id}{suffix}.dcm"
    else:
        raise FileNotFoundError(f"No image file found for {case_id}{suffix}")

def segment_images(case_id: str, model_name: str, class_number: int, patch_size: List[int]):
    master_transforms = Compose([EnsureTyped(keys=["image", "label"]),
                                CropForegroundd(keys=["image", "label"], source_key='label', margin=5, allow_smaller=True),
                                SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
                             ])
    post_transforms = Invertd(keys=["image", "label"],
                            transform=master_transforms,
                            orig_keys=["image", "label"],
                        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #check if case id can be converted to int, 
    if case_id.isdigit():
        dataloader = load_metadata_indices([int(case_id)], master_transforms, ROOT_DIR, class_number)
    else:
        #load nifti image directly and create dataloader with dummy label and metadata
        image = nib.load(get_image_path(case_id)).get_fdata() # type: ignore
        label = nib.load(get_image_path(case_id, "_gt")).get_fdata() if (data_path / f"{case_id}_gt.nii.gz").exists() else np.zeros_like(image) # type: ignore

        data = [{"image": image, "label": label, "metadata": {"affine": np.eye(4)}}]
        dataset = DummyDataset(data, master_transforms)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_with_metadata)


    model = load_trained_model(model_name, device, class_number)
    if model:
        outputs, all_metadata, dice = run_inference(dataloader, model, device, patch_size, post_transforms)
    else:
        labels = None
        all_metadata = None
        for batch in next(iter(dataloader)):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            all_metadata = batch["metadata"]
        outputs, dice = labels, 1.0  #dummy outputs for testing
    return outputs, all_metadata, dice
    #print(f"DICE score on the class {class_number} is {dice}")
    #save_outputs_as_nifti(outputs, all_metadata, data_path, class_number)

@app.get("/segment/{case_id}")
def segment_case(case_id: str, model_name: str):
    if data_path / "segmentations" / model_name / f"{case_id}_pred.nii.gz" in data_path.glob(f"segmentations/{model_name}/{case_id}_pred.nii.gz"):
        return {"output_path": f"/data/segmentations/{model_name}/{case_id}_pred.nii.gz"}
    
    Path.mkdir(data_path / "segmentations" / model_name, exist_ok=True)
    result_list = []
    affine0 = np.eye(4)
    for class_nums in range(1, 5):
        outputs, all_metadata, dice = segment_images(case_id, model_name, class_nums, PATCH_SIZE)
        result, affine0 = save_val_results(outputs, class_nums, all_metadata)
        result_list.append(result*class_nums)
    final_result = np.stack(result_list, axis=0).max(axis=0)
    final_path = data_path / f"segmentations/{model_name}/{case_id}_pred.nii.gz"
    nib.save(nib.Nifti1Image(final_result.astype(np.uint8), affine0), final_path)
    return {"output_path": f"/data/segmentations/{model_name}/{case_id}_pred.nii.gz"}


@app.get("/load/{case_id}")
def load_gts(case_id: str, model_name: str):
    image_path = get_image_path(case_id, category=0)
    gt_path = get_image_path(case_id, "_gt", category=1)
    pred_path = get_image_path(case_id, "_pred", category=2, model_name=model_name)

    return {
        "image" : image_path,
        "gt" : gt_path,
        "pred": pred_path
    }


@app.get("/models", response_model=List[str])
def get_available_models():
    """Return list of available segmentation models."""
    # Get list of model directories from segmentations folder
    segmentations_path = data_path / "segmentations"
    
    if segmentations_path.exists():
        models = [d.name for d in segmentations_path.iterdir() if d.is_dir()]
        return models

    return ['model1']




