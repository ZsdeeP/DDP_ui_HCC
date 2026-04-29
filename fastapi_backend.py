from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import nibabel as nib
import numpy as np
import subprocess
import glob
import os
import shutil 
import tempfile

from pathlib import Path
from urllib.parse import unquote
#segment all four classes and save results in same file with different class numbers, then extract the class of interest in the frontend for visualization and radiomics extraction
#next step: select a custom window around the tumor for radiomics extraction instead of the whole image, to better capture peritumoral features and reduce noise from irrelevant areas. This will require modifying the radiomics extractor to take in a bounding box or mask for feature calculation.
#the user should be able to define the window limits
 
#important: modify the import statements to reflect the actual structure of the project and where the modules are located. The current imports assume all modules are in the same directory, which may not be the case. Adjust the import paths as needed based on your project organization. 
import torch
#import subprocess
from infer_model import DummyDataset
import pandas as pd
import json
import pydicom as dicom
import monai
from typing import List, Union
from monai.data.dataloader import DataLoader
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import (
    CropForegroundd,
    SpatialPadd,
)
from monai.transforms.utility.dictionary import EnsureTyped
from monai.transforms.post.dictionary import Invertd

from infer_model import load_metadata_indices, load_trained_model, run_inference, save_val_results, collate_with_metadata, compute_dice, compute_iou, compute_hausdorff, compute_boundary_confusion, per_slice_dice, per_slice_iou
from radiomics_extractor import RadiomicsExtractor
from tace_planner import TACEPlanner
from survival_analysis import run_survival_analysis
from outcome_predictor import run_outcome_prediction
#replace these data_utils placeholders with the actual functions from the repo
from data_utils import read_dicom_series, read_segmentation_dicom, seg_dicom_to_nifti_aligned, calculate_transform_matrix, generate_coordinate_matrix 

class ModelSelect(BaseModel):
    models: str

class AnalysisRequest(BaseModel):
    clinical_csv: str = "clinical.csv"
    radiomics_dir: str = "radiomics"
    output_dir: str = "survival"

ROOT_DIR = Path(__file__).resolve().parent #PLACEHOLDER
PATCH_SIZE = [64, 64, 64] #FROZEN

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
#data_dir = parent_dir / "nifti_data" / target_dir

#images_dir = data_dir / CLASS_NAME / "ct_HCC" / "imagesTr"
#labels_dir = data_dir / CLASS_NAME / "ct_HCC" / "labelsTr"

#DICOM - ONLY LOAD FILES STARTING WITH 1

static_path = current_dir / "static"
templates_path = static_path / "templates"
data_path = current_dir / "data"

app = FastAPI()
app.mount("/static", StaticFiles(directory=static_path), name="static")
app.mount("/data", StaticFiles(directory=data_path), name="data")

#templates = Jinja2Templates(directory=templates_path)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)},
    )

@app.get("/dicom_files")
def list_dicom_files(url: str):
    if not url.startswith("/data/"):
        return JSONResponse(status_code=400, content={"error": "Invalid URL for DICOM listing"})

    rel_url = unquote(url[len("/data/"):].lstrip("/"))
    fs_path = (data_path / rel_url).resolve()
    base_path = data_path.resolve()

    if not fs_path.is_relative_to(base_path):
        return JSONResponse(status_code=400, content={"error": "Invalid DICOM directory path"})

    if not fs_path.exists() or not fs_path.is_dir():
        return JSONResponse(status_code=404, content={"error": f"Directory not found: {url}"})

    files = []
    for file_path in sorted(fs_path.rglob("*")):
        if file_path.is_file():
            rel_path = file_path.relative_to(data_path).as_posix()
            files.append(f"/data/{rel_path}")

    return {"files": files}

#currently, the dicom is converted to nifti and then the segmentation is shown
#instead, convert the nifti segmentation's affine to match the original dicom orientation and spacing, then save the converted segmentation as nifti for visualization and radiomics extraction. This way we can ensure the segmentations are properly aligned with the original images without needing to modify the frontend visualization code.
#The conversion function should check if the volume has already been converted and use its affine for consistency when converting the segmentation. If the segmentation is in DICOM format, it should be read and converted using the same affine as the volume to ensure proper alignment.
#If segmentation is dicom, do nothing and return the path to the dicom folder, and modify the frontend to read the dicom segmentation and apply the same affine transformation as the nifti volume for visualization. This way we can avoid unnecessary conversions and ensure proper alignment between the images and segmentations regardless of their original format. The key is to maintain consistency in the affine transformations applied to both the images and segmentations, whether they are originally in DICOM or NIfTI format.
# Also, prioritize showing the segmentation in the original DICOM format if it exists, and only convert to NIfTI if necessary for visualization or radiomics extraction. This way we can preserve the original data format and ensure the most accurate representation of the segmentations while still providing flexibility for analysis and visualization.


def reorient_segmentation(case_id: str, model_name : str, cat:int):
    cat_dict = {0: "base_images", 1: "ground_truths", 2: f"segmentations/{model_name}"} 
    suffix_dict = {0: "", 1: "_gt", 2: "_pred"}
    # Load the volume to get its affine matrix
    '''
    nifti_affine = nib.load(data_path / f"{cat_dict[0]}/{case_id}.nii.gz").affine # type: ignore
    m_lps2ras = calculate_transform_matrix("RAS", "LPS")
    m_nifti2dicom = np.linalg.inv(m_lps2ras @ dicom_affine) @ nifti_affine
    ornt = nib.orientations.io_orientation(m_nifti2dicom)
    reoriented_nifti = nib.orientations.apply_orientation(volume, ornt)

    converted_file = nib.Nifti1Image(reoriented_nifti.astype(np.float32), None)
    nib.save(img=converted_file, filename=data_path /f"{cat_dict[cat]}/{case_id}{suffix_dict[cat]}.nii.gz")
    '''
    nifti_path = data_path / f"{cat_dict[cat]}/{case_id}{suffix_dict[cat]}.nii.gz"
    source_dicom_path = data_path / f"{cat_dict[0]}/{case_id}"
    seg_dcm_path = data_path / f"{cat_dict[cat]}/{case_id}{suffix_dict[cat]}"
    
    return Path(seg_dicom_to_nifti_aligned(seg_dcm_path, source_dicom_path, nifti_path))



def get_image_path(case_id: str, suffix: str = "", category: int = 0, model_name: str = "model1") -> str:
    cat_dict = {0: "base_images", 1: "ground_truths", 2: "segmentations"}
    base_dcm_path = data_path / "base_images" / f"{case_id}"

    if category == 2:
        pred_path = data_path / "segmentations" / model_name / f"{case_id}_pred.nii.gz"
        if pred_path.exists():
            return f"/data/segmentations/{model_name}/{case_id}_pred.nii.gz"
        elif (data_path / "segmentations" / model_name / f"{case_id}_pred.nii").exists():
            return f"/data/segmentations/{model_name}/{case_id}_pred.nii"
        else:
            pred_dcm_path = data_path / "segmentations" / model_name / f"{case_id}_pred"
            #look for dcm file
            pred_dcm_file = [file_path for file_path in pred_dcm_path.glob("*") if file_path.is_file()][0]
            if pred_dcm_file.exists():
                pred_path = reorient_segmentation(case_id, model_name, cat=2)
                return f"/data/segmentations/{model_name}/{case_id}_pred.nii.gz"
            else:
                raise FileNotFoundError(f"No segmentation file found for {case_id} with model {model_name}")

    else:
        # Prioritize DICOM directories for images and ground truths
        dcm_path = data_path / cat_dict[category] / f"{case_id}{suffix}"
        if dcm_path.exists() and dcm_path.is_dir():
            if category == 1:
                gt_path = reorient_segmentation(case_id, model_name, cat=1)
                return f"/data/ground_truths/{case_id}_gt.nii.gz"
            else:
                dicom_files = glob.glob(os.path.join(str(dcm_path), "1-*.dcm"))
                if not dicom_files:
                    dicom_files = glob.glob(os.path.join(str(dcm_path), "2-*.dcm"))
                if not dicom_files:
                    dicom_files = glob.glob(os.path.join(str(dcm_path), "*.dcm"))
                if not dicom_files:
                    raise ValueError(f"No DICOM files found in {str(dcm_path)}")
                output_dir = data_path / "segmentations" / model_name
                with tempfile.TemporaryDirectory() as tmpdir:
                    for f in dicom_files:
                        shutil.copy(f, tmpdir)
                    result = subprocess.run(
                        ["dcm2niix", "-z", "y", "-f", "%i_%p", "-o", str(output_dir), str(tmpdir)],
                        capture_output=True,
                        text=True
                    )

                if result.returncode != 0:
                    raise RuntimeError(f"dcm2niix failed: {result.stderr}")

                # Find the output nii.gz
                nifti_files = list(output_dir.glob("*.nii.gz"))
                if not nifti_files:
                    raise ValueError("dcm2niix produced no NIfTI output")

                return f"/data/{cat_dict[category]}/{case_id}{suffix}/"
        
        nii_gz_path = data_path / cat_dict[category] / f"{case_id}{suffix}.nii.gz"
        if nii_gz_path.exists():
            return f"/data/{cat_dict[category]}/{case_id}{suffix}.nii.gz"
        
        nii_path = data_path / cat_dict[category] / f"{case_id}{suffix}.nii"
        if nii_path.exists():
            return f"/data/{cat_dict[category]}/{case_id}{suffix}.nii"
        
        raise FileNotFoundError(f"No image file found for {case_id}{suffix}")


def build_analysis_file_list(output_dir: Path):
    if not output_dir.exists():
        return []
    files = []
    for file_path in sorted(output_dir.rglob("*")):
        if file_path.is_file():
            rel_path = file_path.relative_to(data_path)
            files.append(f"/data/{rel_path}")
    return files


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
        image_path = Path(get_image_path(case_id, category=0))
        label_path = Path(get_image_path(case_id, "_gt", category=1))

        if image_path.is_dir() and label_path.is_dir():
            image, metadata = read_dicom_series(str(image_path))
            image = np.transpose(image, (1, 2, 0))
            label = read_segmentation_dicom(str(label_path)) 
            label = np.transpose(label, (1, 2, 0))

            data = [{"image": image, 
                     "label": label, 
                     "metadata": {**metadata}
                    }]
    
        elif (image_path.suffix == ".nii.gz" or image_path.suffix == ".nii") and \
             (label_path.suffix == ".nii.gz" or label_path.suffix == ".nii"):
            image = nib.load(get_image_path(case_id)).get_fdata() # type: ignore
            label = nib.load(get_image_path(case_id, "_gt", category=1)).get_fdata() if (data_path / f"{case_id}_gt.nii.gz").exists() else np.zeros_like(image) # type: ignore
            data = [{"image": image, "label": label, "metadata": {"affine": np.eye(4)}}]
        else:
            raise ValueError(f"Unsupported file format for case {case_id}")
        
        dataset = DummyDataset(data, master_transforms) #replace with actual HCCDataset
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_with_metadata)

    model = load_trained_model(model_name, device, class_number)
    if model:
        outputs, all_metadata, dice = run_inference(dataloader, model, device, patch_size, post_transforms)
    else:
        labels = None
        all_metadata = None
        for i, batch in enumerate(dataloader):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            all_metadata = batch["metadata"]
        outputs, dice = labels, 1.0  #dummy outputs for testing
    return outputs, all_metadata, dice

@app.get("/segment/{case_id}")
def segment_case(case_id: str, model_name: str):
    print(1)
    seg_dir = data_path / "segmentations" / model_name
    seg_nifti_gz = seg_dir / f"{case_id}_pred.nii.gz"
    seg_nifti = seg_dir / f"{case_id}_pred.nii"
    seg_dcm_dir = seg_dir / f"{case_id}_pred"
    print(seg_dcm_dir)

    if seg_nifti_gz.exists():
        return {"output_path": f"/data/segmentations/{model_name}/{case_id}_pred.nii.gz"}
    elif seg_nifti.exists():
        return {"output_path": f"/data/segmentations/{model_name}/{case_id}_pred.nii"}
    elif seg_dcm_dir.exists():
        return {"output_path": get_image_path(case_id, "_pred", category=2, model_name=model_name)}
    
    else:
        Path.mkdir(seg_dir, parents=True, exist_ok=True)
        result_list = []
        affine0 = np.eye(4)
        for class_nums in range(1, 5): #segment each class separately and save results in same file with different class numbers,
            # then extract the class of interest. Only meant for the HCC-TACE dataset
            outputs, all_metadata, dice = segment_images(case_id, model_name, class_nums, PATCH_SIZE)
            result, affine0 = save_val_results(outputs, all_metadata)
            result_list.append(result*class_nums)
        final_result = np.stack(result_list, axis=0).max(axis=0)
        final_filename = reorient_segmentation(case_id, model_name, 2)
        return {"output_path": f"/data/segmentations/{model_name}/{case_id}_pred.nii.gz"}




@app.get("/load/{case_id}")
def load_gts(case_id: str, model_name: str):
    image_path = get_image_path(case_id, category=0)
    gt_path = get_image_path(case_id, "_gt", category=1)

    return {
        "image" : image_path,
        "gt" : gt_path,
    }

from fastapi import Response
import io




@app.get("/evaluate_case/{case_id}/{structure}")
async def evaluate_case(case_id: str, model_name: str, structure: str):
    tissue_class_dict = {"liver" : 1, "tumor" : 2, "bloodvessels" : 3, "abdominalaorta" : 4}
    tissue_class = tissue_class_dict[structure]
    pred_mask = get_image_path(case_id, "_pred", 2, model_name=model_name)
    em_mask_dir = data_path / "error_masks" / f"{model_name}" / f"{structure}"
    em_mask_dir.mkdir(parents=True, exist_ok=True)
    em_mask = em_mask_dir / f"{case_id}_em.nii.gz"
    gt_mask   = get_image_path(case_id, "_gt", 1, model_name=model_name)


    pred_mask = (current_dir / pred_mask[1:]).resolve()
    gt_mask = (current_dir / gt_mask[1:]).resolve()
    #em_mask = (current_dir / em_mask).resolve()

    return {
        "dice":   compute_dice(tissue_class, pred_mask, gt_mask),
        "iou":    compute_iou(tissue_class, pred_mask, gt_mask),
        "hd":     compute_hausdorff(tissue_class, pred_mask, gt_mask, percentile=100),
        #"hd95":   compute_hausdorff(tissue_class, pred_mask, gt_mask, percentile=95),
        "per_slice_iou" : per_slice_iou(tissue_class, pred_mask, gt_mask), #this is a list idk how js handles it
        "per_slice_dice" : per_slice_dice(tissue_class, pred_mask, gt_mask), # same here
        "boundary_error_map": compute_boundary_confusion(tissue_class, pred_mask, gt_mask, em_mask),
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


@app.post("/extract_radiomics/{case_id}")
def extract_radiomics(case_id: str):
    """Extract radiomics features from a segmented case for all tissue classes"""
    try:
        # Find scan folder and segmentation file
        scan_folder = data_path / "base_images" / case_id
        seg_file = data_path / "segmentations" / "model1" / f"{case_id}_pred.nii.gz"

        if not scan_folder.exists():
            return JSONResponse(status_code=404, content={"error": f"Scan folder not found: {scan_folder}"})

        if not seg_file.exists():
            return JSONResponse(status_code=404, content={"error": f"Segmentation file not found: {seg_file}"})

        # Extract features for all tissue classes
        output_dir = data_path / "radiomics"
        output_dir.mkdir(exist_ok=True)
        
        extractor = RadiomicsExtractor()
        features_dict = extractor.extract_features_from_patient(str(scan_folder), str(seg_file), case_id, output_dir=str(output_dir))
        print("features extracted:", features_dict)
        if features_dict is None:
            return JSONResponse(status_code=400, content={"error": "Failed to extract features"})

        # Prepare response with paths to saved CSV files
        csv_paths = {}
        feature_counts = {}
        for class_name in extractor.CLASS_NAMES.values():
            if features_dict.get(class_name):
                csv_paths[class_name] = f"/data/radiomics/{case_id}_radiomics_{class_name}.csv"
                feature_counts[class_name] = len(features_dict[class_name]) - 1  # Exclude patient_id

        return {
            "case_id": case_id,
            "status": "success",
            "csv_paths": csv_paths,
            "feature_counts": feature_counts
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/run_survival_analysis")
def run_survival_analysis_endpoint(request: AnalysisRequest):
    clinical_path = data_path / request.clinical_csv
    radiomics_path = data_path / request.radiomics_dir
    output_path = data_path / request.output_dir

    if not clinical_path.exists():
        return JSONResponse(status_code=404, content={"error": f"Clinical data not found: {clinical_path}"})
    if not radiomics_path.exists():
        return JSONResponse(status_code=404, content={"error": f"Radiomics directory not found: {radiomics_path}"})

    output_path.mkdir(parents=True, exist_ok=True)
    try:
        run_survival_analysis(str(clinical_path), radiomics_dir=str(radiomics_path), output_dir=str(output_path))
        return {
            "status": "complete",
            "output_dir": f"/data/{request.output_dir}",
            "files": build_analysis_file_list(output_path)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/run_outcome_prediction")
def run_outcome_prediction_endpoint(request: AnalysisRequest):
    clinical_path = data_path / request.clinical_csv
    radiomics_path = data_path / request.radiomics_dir
    output_path = data_path / request.output_dir

    if not clinical_path.exists():
        return JSONResponse(status_code=404, content={"error": f"Clinical data not found: {clinical_path}"})
    if not radiomics_path.exists():
        return JSONResponse(status_code=404, content={"error": f"Radiomics directory not found: {radiomics_path}"})

    output_path.mkdir(parents=True, exist_ok=True)
    try:
        run_outcome_prediction(str(clinical_path), radiomics_dir=str(radiomics_path), output_dir=str(output_path))
        return {
            "status": "complete",
            "output_dir": f"/data/{request.output_dir}",
            "files": build_analysis_file_list(output_path)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/tace_plan/{case_id}")
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




