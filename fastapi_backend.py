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
import pandas as pd
import json
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
from radiomics_extractor import RadiomicsExtractor
from tace_planner import TACEPlanner
from data_utils import read_dicom_series, read_segmentation_dicom

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

    nii_path = data_path / cat_dict[category] / f"{case_id}{suffix}.nii"
    nii_gz_path = data_path / cat_dict[category] / f"{case_id}{suffix}.nii.gz"

    dcm_path = data_path / cat_dict[category] / f"{case_id}{suffix}.dcm"
    if nii_path.exists():
        return f"/data/{cat_dict[category]}/{case_id}{suffix}.nii"
    elif nii_gz_path.exists():
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
        #account for dicom images and segmentations as well
        image_path = get_image_path(case_id, category=0)
        label_path = get_image_path(case_id, "_gt", category=1)
        if image_path.endswith(".dcm"):
            image, metadata = read_dicom_series(str(image_path))
            label = read_segmentation_dicom(str(label_path)) 
            data = [{"image": image, 
                     "label": label, 
                     "metadata": {**metadata}
                    }]
    
        elif image_path.endswith(".nii.gz") or image_path.endswith(".nii"):
            image = nib.load(get_image_path(case_id)).get_fdata() # type: ignore
            label = nib.load(get_image_path(case_id, "_gt", category=1)).get_fdata() if (data_path / f"{case_id}_gt.nii.gz").exists() else np.zeros_like(image) # type: ignore
            data = [{"image": image, "label": label, "metadata": {"affine": np.eye(4)}}]
        else:
            raise ValueError(f"Unsupported file format for case {case_id}")
        
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


@app.post("/extract-radiomics/{case_id}")
def extract_radiomics(case_id: str, tissue_class: int = 2):
    """Extract radiomics features from a segmented case"""
    try:
        # Find scan folder and segmentation file
        scan_folder = data_path / "base_images" / case_id
        seg_file = data_path / "segmentations" / "model1" / f"{case_id}_pred.nii.gz"

        if not scan_folder.exists():
            return JSONResponse(status_code=404, content={"error": f"Scan folder not found: {scan_folder}"})

        if not seg_file.exists():
            return JSONResponse(status_code=404, content={"error": f"Segmentation file not found: {seg_file}"})

        # Extract features
        extractor = RadiomicsExtractor(class_number=tissue_class)
        features = extractor.extract_features_from_patient(str(scan_folder), str(seg_file), case_id)

        if features is None:
            return JSONResponse(status_code=400, content={"error": "Failed to extract features"})

        # Save features to CSV
        features_df = pd.DataFrame([features])
        output_dir = data_path / "radiomics"
        output_dir.mkdir(exist_ok=True)
        csv_path = output_dir / f"{case_id}_radiomics_class_{tissue_class}.csv"
        features_df.to_csv(csv_path, index=False)

        return {
            "case_id": case_id,
            "tissue_class": tissue_class,
            "features_count": len(features) - 2,  # Exclude patient_id and class_number
            "csv_path": f"/data/radiomics/{case_id}_radiomics_class_{tissue_class}.csv",
            "features": features
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/tace-plan/{case_id}")
def plan_tace_procedure(case_id: str):
    """Run TACE planning analysis on a case"""
    try:
        # Find scan folder and segmentation file
        scan_folder = data_path / "base_images" / case_id
        seg_file = data_path / "segmentations" / "model1" / f"{case_id}_pred.nii.gz"

        if not scan_folder.exists():
            return JSONResponse(status_code=404, content={"error": f"Scan folder not found: {scan_folder}"})

        if not seg_file.exists():
            return JSONResponse(status_code=404, content={"error": f"Segmentation file not found: {seg_file}"})

        # Run TACE planning
        planner = TACEPlanner()
        report = planner.analyze_patient(str(scan_folder), str(seg_file), case_id)

        # Save report
        output_dir = data_path / "tace_reports" / case_id
        output_dir.mkdir(parents=True, exist_ok=True)

        report_file = output_dir / f"{case_id}_tace_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate visualizations
        viz_dir = output_dir / "visualizations"
        planner.visualize_patient(report, str(viz_dir))

        # Find visualization files
        visualizations = {}
        if viz_dir.exists():
            for file_path in viz_dir.glob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(data_path)
                    visualizations[file_path.stem] = f"/data/{rel_path}"

        return {
            "case_id": case_id,
            "report_path": f"/data/tace_reports/{case_id}/{case_id}_tace_report.json",
            "visualizations": visualizations,
            "summary": {
                "tumor_volume_ml": report.get('tumor_analysis', {}).get('tumor_volume_ml', 0),
                "feeding_vessels": report.get('vessel_analysis', {}).get('num_feeding_candidates', 0),
                "tace_feasibility": report.get('tumor_analysis', {}).get('tumor_location', 'unknown')
            }
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})




