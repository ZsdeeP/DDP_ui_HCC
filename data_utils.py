import os
import glob
from pathlib import Path
import numpy as np
import pydicom
from pydicom_seg import reader #also create nifti loader

def read_dicom_series_uniform(dicom_dir):
    """
    Robust DICOM series reader that handles non-uniform spacing warnings
    by explicitly sorting on ImagePositionPatient and filtering outliers.
    
    import os

    # Step 1: Read all files and extract position + metadata
    files = [os.path.join(dicom_dir, f) 
             for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    
    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f, stop_before_pixels=True)
            # Skip non-image SOPs (reports, segmentations, etc.)
            if not hasattr(dcm, 'ImagePositionPatient'):
                continue
            if not hasattr(dcm, 'PixelData') and not hasattr(dcm, 'Rows'):
                continue
            # Skip seg DICOMs that may be in same folder
            if hasattr(dcm, 'SOPClassUID'):
                if '1.2.840.10008.5.1.4.1.1.66' in str(dcm.SOPClassUID):  # seg UID
                    continue
            slices.append((f, float(dcm.ImagePositionPatient[2]), dcm))
        except Exception:
            continue

    # Step 2: Sort by Z position
    slices.sort(key=lambda x: x[1])
    
    # Step 3: Detect and filter the dominant series by SeriesInstanceUID
    from collections import Counter
    series_counts = Counter(s[2].SeriesInstanceUID for s in slices)
    dominant_series = series_counts.most_common(1)[0][0]
    slices = [s for s in slices if s[2].SeriesInstanceUID == dominant_series]
    
    print(f"  Kept {len(slices)} slices from series {dominant_series}")"""

    """
    Read a series of DICOM files and stack them into a 3D volume
    Sorts slices based on ImagePositionPatient z-coordinate for correct ordering
    """
    # Find all DICOM files in the directory
    # First try to find DICOM files starting with "1-"
    dicom_files = glob.glob(os.path.join(dicom_dir, "1-*.dcm"))

    # If no files with "1-" prefix are found, look for files starting with "2-"
    if not dicom_files:
        dicom_files = glob.glob(os.path.join(dicom_dir, "2-*.dcm"))

    # If still no files found, get all DICOM files
    if not dicom_files:
        dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

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

    # Step 4: Check spacing uniformity after filtering
    z_positions = [s[1].ImagePositionPatient[2] for s in slices]
    spacings = np.diff(z_positions)
    median_spacing = np.median(spacings)
    outliers = np.where(np.abs(spacings - median_spacing) > 0.1 * median_spacing)[0]
    
    if len(outliers) > 0:
        print(f"  WARNING: {len(outliers)} non-uniform gaps remain at indices: {outliers}")
        print(f"  Spacings at those gaps: {spacings[outliers]}")
        # Remove slices causing gaps (keep the cleaner run)
        # Drop the slice after each gap
        drop_indices = set(outliers + 1)
        slices = [s for i, s in enumerate(slices) if i not in drop_indices]
        print(f"  After gap removal: {len(slices)} slices")

    # Step 5: Load with SimpleITK using explicit sorted file list
    sorted_files = [s[0] for s in slices]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    
    print(f"  Final volume size: {image.GetSize()}")
    print(f"  Spacing: {image.GetSpacing()}")
    
    return image, slices  # return slices too so you can map Z later


def generate_coordinate_matrix(coord_system):
    """
    Generate a transformation matrix from the given coordinate system
    to the world coordinate system.
    :param coord_system: String representing the coordinate system (e.g., 'LSP', 'RAS', etc.)
    :return: 4x4 transformation matrix
    """
    if len(coord_system) != 3:
        raise ValueError("Coordinate system must have exactly 3 characters (e.g., 'RAS', 'LSP').")
    
    # Mapping from coordinate labels to axis directions
    axis_mapping = {
        'R': [1, 0, 0],   # Right (positive X)
        'L': [-1, 0, 0],  # Left (negative X)
        'A': [0, 1, 0],   # Anterior (positive Y)
        'P': [0, -1, 0],  # Posterior (negative Y)
        'S': [0, 0, 1],   # Superior (positive Z)
        'I': [0, 0, -1]   # Inferior (negative Z)
    }

    # Extract the directions for X, Y, Z
    x_axis = axis_mapping[coord_system[0]]
    y_axis = axis_mapping[coord_system[1]]
    z_axis = axis_mapping[coord_system[2]]

    # Ensure the coordinate system is valid 
    if not np.isclose(np.dot(x_axis, y_axis), 0) or not np.isclose(np.dot(x_axis, z_axis), 0) or not np.isclose(np.dot(y_axis, z_axis), 0):
        raise ValueError(f"Invalid coordinate system: {coord_system}. Axes must be orthogonal.")
    
    # Create the transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, 0] = x_axis  # X axis
    transform_matrix[:3, 1] = y_axis  # Y axis
    transform_matrix[:3, 2] = z_axis  # Z axis

    return transform_matrix

def calculate_transform_matrix(source_coord, target_coord):
    """
    Calculate the transformation matrix to convert coordinates
    from source_coord to target_coord.
    :param source_coord: Source coordinate system (e.g., 'LSP')
    :param target_coord: Target coordinate system (e.g., 'RAS')
    :return: 4x4 transformation matrix
    """
    # Generate matrices for both systems
    source_to_world = generate_coordinate_matrix(source_coord)
    target_to_world = generate_coordinate_matrix(target_coord)

    # Calculate the transform from source to target
    transform_matrix = np.linalg.inv(target_to_world) @ source_to_world
    return transform_matrix


def compute_affine(slices):
    dicom_files = [s[0] for s in slices] 
    ds = pydicom.dcmread(dicom_files[0])
    
    image_orientation = np.array(ds.ImageOrientationPatient) 
    #print(f'image_orientation: {image_orientation}')
    #print(f'image_position: {ds.ImagePositionPatient}')
    row_cosine = image_orientation[3:]
    col_cosine = image_orientation[:3]
    #col_cosine = image_orientation[:3]
    #row_cosine = image_orientation[3:]
    pixel_spacing = np.array(ds.PixelSpacing) 
    
    first_position = np.array(ds.ImagePositionPatient)  
    if len(dicom_files) > 1:
        ds_next = pydicom.dcmread(dicom_files[1]) 
        second_position = np.array(ds_next.ImagePositionPatient)
        #print(f'first_instance_number: {ds.InstanceNumber}')
        #print(f'second_instance_number: {ds_next.InstanceNumber}')
        #print(f'first_position: {first_position}')
        #print(f'second_position: {second_position}')
        
        slice_direction = second_position - first_position
        #print(f'slice_direction before: {slice_direction}')
        slice_spacing = np.linalg.norm(slice_direction)
        slice_cosine = slice_direction / slice_spacing

    else:
        slice_cosine = np.cross(col_cosine,row_cosine)
        slice_spacing = 1.0
        ds_next = pydicom.dcmread(dicom_files[0]) 

    
    affine = np.eye(4)
    affine[:3, 0] = row_cosine * pixel_spacing[0]
    affine[:3, 1] = col_cosine * pixel_spacing[1]
    affine[:3, 2] = slice_cosine * slice_spacing* np.sign(ds_next.InstanceNumber-ds.InstanceNumber)
    affine[:3, 3] = first_position
    return affine


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
    
    affine_matrix = compute_affine(slices)
    
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
        'affine':affine_matrix,
        'patient_id': patient_id,
        'slice_thickness': slice_thickness,
        'pixel_spacing': pixel_spacing,
        'shape': volume.shape,
        'image_position': image_position,
        'image_orientation': image_orientation,

    }

    return volume, metadata

import pydicom
import highdicom as hd
import SimpleITK as sitk
import numpy as np
import nibabel as nib

def read_source_dicom_geometry(dicom_dir):
    """Get geometry from the source DICOM series (the one dcm2niix processed)"""
    import os
    files = sorted([
        os.path.join(dicom_dir, f) 
        for f in os.listdir(dicom_dir) 
        if f.endswith('.dcm')
    ])
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    return image  # has correct origin, spacing, direction

def seg_dicom_to_nifti_aligned(seg_dcm_path: Path, source_dicom_dir: Path, output_nifti_path: Path):
    """
    Convert segmentation DICOM to NIfTI aligned with dcm2niix output.
    
    Args:
        seg_dcm_path: path to segmentation .dcm file
        source_dicom_dir: directory of original DICOM series
        output_nifti_path: where to save the aligned segmentation NIfTI
    """
    # Step 1: Read source geometry (what dcm2niix used as reference)
    
    source_image, source_slices = read_dicom_series_uniform(source_dicom_dir)
    
    # Build a Z-position → slice index lookup from actual positions
    # This is more reliable than TransformPhysicalPointToIndex when spacing varies
    z_positions = np.array([s[1].ImagePositionPatient[2] for s in source_slices])
    #print(f"Source Z positions: {z_positions}")
    
    def physical_z_to_index(z):
        """Find nearest slice index for a given Z position"""
        idx = np.argmin(np.abs(z_positions - z))
        dist = abs(z_positions[idx] - z)
        if dist > 5.0:  # more than 5mm away — likely a real miss
            return None, dist
        return idx, dist

    source_size = source_image.GetSize()       # (W, H, D) in SimpleITK order
    source_origin = source_image.GetOrigin()
    source_spacing = source_image.GetSpacing()
    source_direction = source_image.GetDirection()

    # Step 2: Read segmentation with highdicom (preserves per-frame geometry)
    seg_reader = reader.SegmentReader()
    seg_dcm_file = [file_path for file_path in seg_dcm_path.glob("*") if file_path.is_file()][0]
    #m = read_segmentation_dicom(str(seg_dcm_file))  # just to check if it's readable
    #print(m.shape)
    seg_dcm = pydicom.dcmread(seg_dcm_file)
    seg_dcm = seg_reader.read(seg_dcm)
    result = hd.seg.segread(seg_dcm_file)
    
    # Step 3: Build empty volume matching source geometry
    seg_volume = np.zeros(
        (source_size[2], source_size[1], source_size[0]),  # D, H, W
        dtype=np.uint8
    )

    # Step 4: For each segment, map voxels into source image space
    for segment_number, segment_info in seg_dcm.segment_infos.items():
        #print(segment_number)
        seg_array = seg_dcm.segment_data(segment_number)  # raw mask in seg's frame
        
        # Get the per-frame positions from the seg DICOM
        frame_positions = _get_frame_positions(result)
        #print(frame_positions)
        for frame_idx, position in enumerate(frame_positions):
            if frame_idx >= seg_array.shape[0]:
                break
                
            # Convert LPS position to voxel index in source image
            physical_point = (float(position[0]), float(position[1]), float(position[2]))
            
            try:
                z, dist = physical_z_to_index(position[2])
                if z is None: 
                    continue
                
                if 0 <= z < source_size[2]:
                    frame_mask = seg_array[frame_idx]
                    # Assign segment label (handles overlapping segments)
                    seg_volume[z][frame_mask > 0] = segment_number
            except Exception:
                continue

    # Step 5: Convert SimpleITK geometry to nibabel affine
    seg_sitk = sitk.GetImageFromArray(seg_volume)
    seg_sitk.SetOrigin(source_origin)
    seg_sitk.SetSpacing(source_spacing)
    seg_sitk.SetDirection(source_direction)
    
    affine = _sitk_to_nibabel_affine(seg_sitk)
    
    #print(seg_volume.sum())
    nifti_img = nib.Nifti1Image(
        np.transpose(seg_volume, (2, 1, 0)),  # SimpleITK WHD -> nibabel XYZ
        affine
    )
    nib.save(nifti_img, output_nifti_path)
    print(f"Saved aligned segmentation to {output_nifti_path}")
    return output_nifti_path


def _get_frame_positions(seg_dcm):
    """Extract ImagePositionPatient for each frame in seg DICOM"""
    positions = []
    for frame_item in seg_dcm.PerFrameFunctionalGroupsSequence:
        pos = frame_item.PlanePositionSequence[0].ImagePositionPatient
        positions.append([float(x) for x in pos])
    return positions


def _sitk_to_nibabel_affine(sitk_image):
    """Convert SimpleITK image metadata to a nibabel-compatible affine matrix"""
    origin = np.array(sitk_image.GetOrigin())
    spacing = np.array(sitk_image.GetSpacing())
    direction = np.array(sitk_image.GetDirection()).reshape(3, 3)
    
    affine = np.eye(4)
    affine[:3, :3] = direction * spacing  # scale columns by spacing
    affine[:3, 3] = origin
    
    # DICOM is LPS, NIfTI is RAS — flip X and Y
    lps_to_ras = calculate_transform_matrix('LPS', 'RAS')
    affine = lps_to_ras @ affine
    
    return affine



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
        # Create multi-label volume where each voxel contains the class number
        first_segment = result.segment_data(1)
        seg_array = np.zeros_like(first_segment, dtype=np.uint8)
        for segment_number, _ in segments_info.items():
            mask = result.segment_data(segment_number)
            seg_array[mask > 0] = segment_number
    return seg_array.astype(np.float32)
