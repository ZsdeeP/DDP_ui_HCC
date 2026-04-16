import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import logging
from tqdm import tqdm
import nibabel as nib
from pathlib import Path
from data_utils import read_dicom_series, read_segmentation_dicom, apply_window

class RadiomicsExtractor:
    """Extract radiomics features"""

    def __init__(self, class_number=2, apply_liver_window=True, wc=150, ww=250):
        """
        Initialize radiomics extractor

        Args:
            class_number: Which tissue class to extract features from
                         1=liver, 2=tumour, 3=bloodvessels, 4=abdominalaorta
            apply_liver_window: Whether to apply windowing to CT images
            wc: Window center for CT windowing
            ww: Window width for CT windowing
        """
        self.class_number = class_number
        self.apply_liver_window = apply_liver_window
        self.wc = wc
        self.ww = ww

        # Configure radiomics extractor
        self.extractor = featureextractor.RadiomicsFeatureExtractor()

        # Enable ALL feature classes explicitly
        self.extractor.enableAllFeatures()

        # Enable ALL image types (filters) explicitly
        self.extractor.enableAllImageTypes()

        # Additional: Explicitly enable each image type to ensure all are active
        # Original image
        self.extractor.enableImageTypeByName('Original')
        self.extractor.enableImageTypeByName('Wavelet')
        self.extractor.enableImageTypeByName('LoG')
        self.extractor.enableImageTypeByName('Square')
        self.extractor.enableImageTypeByName('SquareRoot')
        self.extractor.enableImageTypeByName('Logarithm')
        self.extractor.enableImageTypeByName('Exponential')
        self.extractor.enableImageTypeByName('Gradient')

        # Set parameters
        params = {
            'binWidth': 25,
            'resampledPixelSpacing': [1, 1, 1],
            'interpolator': sitk.sitkBSpline,
            'normalize': True,
            'verbose': False,
            'force2D': False,  # Extract 3D features
            'force2Ddimension': 0  # If force2D is True, which dimension to use
        }

        for key, value in params.items():
            self.extractor.settings[key] = value

        # Setup logging
        logging.getLogger('radiomics').setLevel(logging.ERROR)

    def _load_volume(self, volume_path):
        """
        Load volume from either DICOM series or NIfTI file
        
        Args:
            volume_path: Path to DICOM folder or NIfTI file
            
        Returns:
            tuple: (volume array, metadata dict)
        """
        volume_path = Path(volume_path)
        
        # Check if it's a NIfTI file
        if volume_path.is_file() and (volume_path.suffix in ['.nii', '.gz']):
            # Load NIfTI file
            nib_img = nib.load(volume_path)
            volume = nib_img.get_fdata() #type: ignore
            metadata = {
                'affine': nib_img.affine,
                'spacing': nib_img.header.get_zooms()[:3]  # Get voxel spacing
            }
            return volume, metadata
        # Check if it's a DICOM directory
        elif volume_path.is_dir():
            # Load DICOM series
            volume, metadata = read_dicom_series(str(volume_path))
            return volume, metadata
        else:
            raise ValueError(f"Invalid volume path: {volume_path}. Must be a DICOM directory or NIfTI file.")

    def _load_segmentation(self, seg_path, class_number=None):
        """
        Load segmentation from either DICOM or NIfTI file
        
        Args:
            seg_path: Path to DICOM SEG file or NIfTI file
            class_number: Class number to extract (for DICOM SEG)
            
        Returns:
            np.ndarray: Segmentation mask
        """
        seg_path = Path(seg_path)
        
        # Check if it's a NIfTI file
        if seg_path.is_file() and (seg_path.suffix in ['.nii', '.gz']):
            # Load NIfTI file
            nib_img = nib.load(seg_path)
            segmentation = nib_img.get_fdata() #type: ignore
            
            # If requesting a specific class, extract it
            if class_number is not None:
                segmentation = (segmentation == class_number).astype(np.float32)
            
            return segmentation
        # Check if it's a DICOM SEG file
        elif seg_path.is_file() and seg_path.suffix.lower() == '.dcm':
            # Load DICOM segmentation
            segmentation = read_segmentation_dicom(str(seg_path), class_number=class_number)
            return segmentation
        else:
            raise ValueError(f"Invalid segmentation path: {seg_path}. Must be a DICOM SEG file or NIfTI file.")

    def extract_features_from_patient(self, scan_folder, seg_file, patient_id):
        """
        Extract radiomics features for a single patient

        Args:
            scan_folder: Path to DICOM series folder or NIfTI volume file
            seg_file: Path to segmentation DICOM file or NIfTI file
            patient_id: Patient identifier

        Returns:
            dict: Dictionary of features with patient_id
        """
        try:
            # Load CT volume
            volume, metadata = self._load_volume(scan_folder)

            # Apply windowing if requested
            if self.apply_liver_window:
                volume = apply_window(volume, window_center=self.wc, window_width=self.ww)

            # Load segmentation mask
            segmentation = self._load_segmentation(seg_file, class_number=self.class_number)

            if np.sum(segmentation) == 0:
                logging.warning(f"Empty mask for {patient_id}, class {self.class_number}")
                return None

            # Convert to SimpleITK images
            volume_sitk = sitk.GetImageFromArray(volume)
            mask_sitk = sitk.GetImageFromArray(segmentation)

            # Set spacing from metadata
            if 'spacing' in metadata:
                spacing = metadata['spacing']
                volume_sitk.SetSpacing(spacing)
                mask_sitk.SetSpacing(spacing)

            # Extract features
            features = self.extractor.execute(volume_sitk, mask_sitk)

            # Convert to regular dict and filter out diagnostic info
            feature_dict = {
                'patient_id': patient_id,
                'class_number': self.class_number
            }

            for key, value in features.items():
                # Skip diagnostic information
                if not key.startswith('diagnostics_'):
                    # Convert numpy types to Python types
                    if isinstance(value, (np.integer, np.floating)):
                        value = float(value)
                    feature_dict[key] = value

            return feature_dict

        except Exception as e:
            logging.error(f"Error extracting features for {patient_id}: {str(e)}")
            return None

    def extract_features_from_metadata(self, metadata_csv, root_dir, output_csv=None):
        """
        Extract radiomics features for all patients in metadata CSV

        Args:
            metadata_csv: Path to metadata CSV file
            root_dir: Root directory containing data
            output_csv: Path to save features CSV (optional)

        Returns:
            pd.DataFrame: DataFrame of extracted features
        """
        metadata_df = pd.read_csv(metadata_csv)
        feature_list = []

        print(f"Extracting radiomics features for class {self.class_number}...")
        print(f"Processing {len(metadata_df)} patients...")

        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            patient_id = row['PATIENT_ID']
            scan_folder = os.path.join(root_dir, row['SCAN_FOLDER'])
            seg_file = os.path.join(root_dir, row['SEGMENTATION_FILE'])

            features = self.extract_features_from_patient(scan_folder, seg_file, patient_id)

            if features is not None:
                feature_list.append(features)

        # Create DataFrame
        features_df = pd.DataFrame(feature_list)

        print(f"\nExtracted {len(features_df)} feature sets")
        print(f"Number of features per patient: {len(features_df.columns) - 2}")  # Exclude patient_id and class_number

        # Save to CSV if requested
        if output_csv:
            features_df.to_csv(output_csv, index=False)
            print(f"Features saved to {output_csv}")

        return features_df
