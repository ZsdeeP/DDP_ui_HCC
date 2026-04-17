import nibabel as nib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import dicom2nifti
import subprocess


from pathlib import Path
#segment all four classes and save results in same file with different class numbers, then extract the class of interest in the frontend for visualization and radiomics extraction
#next step: select a custom window around the tumor for radiomics extraction instead of the whole image, to better capture peritumoral features and reduce noise from irrelevant areas. This will require modifying the radiomics extractor to take in a bounding box or mask for feature calculation.
#the user should be able to define the window limits
 
#important: modify the import statements to reflect the actual structure of the project and where the modules are located. The current imports assume all modules are in the same directory, which may not be the case. Adjust the import paths as needed based on your project organization. 
import torch
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

from infer_model import load_metadata_indices, load_trained_model, run_inference, save_val_results, collate_with_metadata
from radiomics_extractor import RadiomicsExtractor
from tace_planner import TACEPlanner
from survival_analysis import run_survival_analysis
from outcome_predictor import run_outcome_prediction
#replace these data_utils placeholders with the actual functions from the repo
from data_utils import read_dicom_series, read_segmentation_dicom, calculate_transform_matrix, generate_coordinate_matrix 

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


static_path = current_dir / "static"
templates_path = static_path / "templates"
data_path = current_dir / "data"

app = FastAPI()
app.mount("/static", StaticFiles(directory=static_path), name="static")
app.mount("/data", StaticFiles(directory=data_path), name="data")

#templates = Jinja2Templates(directory=templates_path)

def handle_dcm_paths(case_id: str, dcm_path: Path, nii_gz_path: Path, suffix: str, cat:str, base_cat="base_images"):
    seg_file_name = [file_path for file_path in dcm_path.glob("*") if file_path.is_file()]
    
    if len(seg_file_name) == 1:
        # Handle segmentation - use the converted volume's affine for consistency
        # First, ensure the volume has been converted
        volume_path = data_path / base_cat / f"{case_id}.nii.gz"
        if not volume_path.exists():
            volume_path = data_path / base_cat / f"{case_id}.nii"
        
        if not volume_path.exists():
            raise FileNotFoundError(f"Volume file not found for {case_id} in {base_cat}. Please convert the volume first.")
        
        # Load the volume to get its affine matrix
        volume_nib = nib.load(volume_path)
        #print(volume_nib.get_fdata().shape)
        affine = volume_nib.affine
        
        # Read segmentation
        seg_volume1 = read_segmentation_dicom(str(dcm_path / seg_file_name[0]), class_number=1)
        seg_volume2 = read_segmentation_dicom(str(dcm_path / seg_file_name[0]), class_number=2)
        seg_volume3 = read_segmentation_dicom(str(dcm_path / seg_file_name[0]), class_number=3)
        seg_volume4 = read_segmentation_dicom(str(dcm_path / seg_file_name[0]), class_number=4)
        seg_volume = np.zeros_like(seg_volume1, dtype=np.uint8)
        seg_volume[seg_volume1 > 0] = 1
        seg_volume[seg_volume2 > 0] = 2
        seg_volume[seg_volume3 > 0] = 3 
        seg_volume[seg_volume4 > 0] = 4


        #seg_volume = np.transpose(seg_volume, (1, 2, 0))  # Reorient to match volume orientation
        #print(seg_volume.shape)
        
        # Apply the volume's affine to the segmentation
        converted_file = nib.Nifti1Image(seg_volume.astype(np.float32), affine=affine)
        nib.save(img=converted_file, filename=nii_gz_path)
    else:
        volume, metadata = read_dicom_series(str(dcm_path))
        m_lps2ras = calculate_transform_matrix("LPS", "RAS")
        affine = metadata.get("affine", np.eye(4))
        nifti_affine = np.eye(4)
        m_dicom2nifti = np.linalg.inv(m_lps2ras @ nifti_affine) @ affine
        #print(volume.shape)
        converted_file = nib.Nifti1Image(volume.astype(np.float32), affine=m_dicom2nifti)
        nib.save(img=converted_file, filename=nii_gz_path)
        '''# Use dcm2niix for regular DICOM series
        output_dir = str(nii_gz_path.parent)
        filename = nii_gz_path.stem  # filename without .nii.gz
        if filename.endswith(".nii"):
            filename = filename[:-4]  # remove .nii if present
        cmd = f"dcm2niix -z y -f {filename} -o {output_dir} {str(dcm_path)}"
        subprocess.run(cmd, shell=True, check=True)'''
    
    print(f" NiFti File saved to {nii_gz_path.name}")
    return f"/data/{cat}/{case_id}{suffix}.nii.gz"



def get_image_path(case_id: str, suffix: str = "", category: int = 0, model_name: str = "model1") -> str:
    cat_dict = {0: "base_images", 1: "ground_truths", 2: "segmentations"}

    if category == 2:
        pred_path = data_path / "segmentations" / model_name / f"{case_id}_pred.nii.gz"
        if pred_path.exists():
            return f"/data/segmentations/{model_name}/{case_id}_pred.nii.gz"
        elif (data_path / "segmentations" / model_name / f"{case_id}_pred.nii").exists():
            return f"/data/segmentations/{model_name}/{case_id}_pred.nii"
        else:
            pred_dcm_path = data_path / "segmentations" / model_name / f"{case_id}_pred"
            if pred_dcm_path.exists():
                dcm2nii_path = handle_dcm_paths(case_id, pred_dcm_path, pred_path, "_pred", cat_dict[category], "base_images")
                return dcm2nii_path
            else:
                raise FileNotFoundError(f"No segmentation file found for {case_id} with model {model_name}")

    nii_path = data_path / cat_dict[category] / f"{case_id}{suffix}.nii"
    nii_gz_path = data_path / cat_dict[category] / f"{case_id}{suffix}.nii.gz"

    dcm_path = data_path / cat_dict[category] / f"{case_id}{suffix}"
    if nii_path.exists():
        return f"/data/{cat_dict[category]}/{case_id}{suffix}.nii"
    elif nii_gz_path.exists():
        return f"/data/{cat_dict[category]}/{case_id}{suffix}.nii.gz"
    elif dcm_path.exists():
        return handle_dcm_paths(case_id, dcm_path, nii_gz_path, suffix, cat_dict[category], "base_images")
        
    else:
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
    seg_dir = data_path / "segmentations" / model_name
    seg_nifti_gz = seg_dir / f"{case_id}_pred.nii.gz"
    seg_nifti = seg_dir / f"{case_id}_pred.nii"
    seg_dcm_dir = seg_dir / f"{case_id}_pred"

    if seg_nifti_gz.exists():
        return {"output_path": f"/data/segmentations/{model_name}/{case_id}_pred.nii.gz"}
    elif seg_nifti.exists():
        return {"output_path": f"/data/segmentations/{model_name}/{case_id}_pred.nii"}
    elif seg_dcm_dir.exists():
        try:
            converted_file = handle_dcm_paths(case_id, seg_dcm_dir, seg_nifti_gz, "_pred", "segmentations", "base_images")
            return {"output_path": f"/data/segmentations/{model_name}/{Path(converted_file).name}"}
        except subprocess.CalledProcessError as e:
            raise FileNotFoundError(f"Failed to convert segmentation DICOM to NIfTI: {e}")

    Path.mkdir(seg_dir, parents=True, exist_ok=True)
    result_list = []
    affine0 = np.eye(4)
    for class_nums in range(1, 5): #segment each class separately and save results in same file with different class numbers,
        # then extract the class of interest. Only meant for the HCC-TACE dataset
        outputs, all_metadata, dice = segment_images(case_id, model_name, class_nums, PATCH_SIZE)
        result, affine0 = save_val_results(outputs, class_nums, all_metadata)
        result_list.append(result*class_nums)
    final_result = np.stack(result_list, axis=0).max(axis=0)
    final_path = data_path / f"segmentations/{model_name}/{case_id}_pred.nii.gz"
    nib.save(nib.Nifti1Image(final_result.astype(np.uint8), affine0), final_path)
    return {"output_path": f"/data/segmentations/{model_name}/{case_id}_pred.nii.gz"}

'''def copy_to_data(case_id:str):
    image_path = images_dir / f"{case_id}.nii.gz"
    label_path = labels_dir / f"{case_id}.nii.gz"
    shutil.copy2(image_path, data_path)
    shutil.copy2(label_path, data_path)'''




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




