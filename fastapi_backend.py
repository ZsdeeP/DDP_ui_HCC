import nibabel as nib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil

CLASS_NAME = "abdominalaorta"
IS_VAL = True

target_dir = "train"
if IS_VAL:
    target_dir = "val"

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


def compute_error_maps(case_id: str):
    """
    Computes error maps (false positives and false negatives)
    
    :param case_id: the id number (e.g. 2)
    :type case_id: str
    """
    gt = nib.load(f"{data_path}/label{case_id}.nii.gz").get_fdata()
    pred = nib.load(f"{data_path}/label{case_id}.nii.gz").get_fdata()

    gt = gt > 0
    pred = pred > 0

    fp = np.logical_and(pred, np.logical_not(gt))
    fn = np.logical_and(gt, np.logical_not(pred))

    fp_path = f"data/{case_id}_fp.nii.gz"
    fn_path = f"data/{case_id}_fn.nii.gz"

    nib.save(nib.Nifti1Image(fp.astype(np.uint8), np.eye(4)), fp_path)
    nib.save(nib.Nifti1Image(fn.astype(np.uint8), np.eye(4)), fn_path)

    return fp_path, fn_path


'''def copy_to_data(case_id:str):
    image_path = images_dir / f"{case_id}.nii.gz"
    label_path = labels_dir / f"{case_id}.nii.gz"
    shutil.copy2(image_path, data_path)
    shutil.copy2(label_path, data_path)'''




@app.get("/load/{case_id}")
def load_gts(case_id: str): #nifti_data/val/abdominalaorta/ct_HCC/imagesTr/29.nii.gz
    #copy_to_data(case_id)
    image_path = f"data/image{case_id}.nii.gz"
    gt_path = f"data/label{case_id}.nii.gz"
    pred_path = f"data/label{case_id}.nii.gz" #change this

    return {
        "image" : image_path,
        "gt" : gt_path,
        "pred": pred_path
    }

@app.get("/error/{case_id}")
def get_error_maps(case_id: str):

    fp_path, fn_path = compute_error_maps(case_id)

    return {
        "fp": fp_path,
        "fn": fn_path
    }

@app.get("/largest_error_slice/{case_id}")
def largest_error_slice(case_id: str):

    gt = nib.load(f"{data_path}/label{case_id}.nii.gz").get_fdata() #change
    pred = nib.load(f"{data_path}/label{case_id}.nii.gz").get_fdata() #change

    error = np.logical_xor(gt > 0, pred > 0)

    try:
        slice_error = error.sum(axis=(0,1)).reshape(-1, 1).squeeze()
    except Exception as e:
        print(f"An error occured {e}. Falling back to slice error = 0")
        slice_error = np.zeros(64)
    

    max_slice = int(np.argmax(slice_error))

    return {
        "slice": max_slice,
        "error_voxels": int(slice_error[max_slice]),
        "slice_error": slice_error.tolist()
    }


'''@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    # This will render the "index.html" file from the "templates" directory
    return templates.TemplateResponse("index.html", {"request": request})'''


