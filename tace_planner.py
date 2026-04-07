"""
TACE Planning Module with Comprehensive Visualizations
Provides evidence-based visualizations to support all clinical decisions
"""
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import json
from tqdm import tqdm
import warnings
import plotly.graph_objects as go

from data_utils import read_dicom_series, read_segmentation_dicom, apply_window


class VesselAnalyzer:
    """Analyze blood vessel structure and topology"""

    def __init__(self):
        self.vessel_skeleton = None
        self.vessel_branches = None
        self.branch_points = None
        self.endpoint_points = None

    def extract_skeleton(self, vessel_mask):
        """Extract vessel centerline skeleton using morphological skeletonization"""
        vessel_binary = (vessel_mask > 0).astype(np.uint8)

        if np.sum(vessel_binary) == 0:
            warnings.warn("Empty vessel mask provided")
            return np.zeros_like(vessel_binary)

        skeleton = skeletonize(vessel_binary)
        self.vessel_skeleton = skeleton
        return skeleton

    def find_branch_points(self, skeleton):
        """Identify branching points in vessel skeleton"""
        if np.sum(skeleton) == 0:
            return np.array([]), np.array([])

        kernel = ndimage.generate_binary_structure(3, 3)
        neighbor_count = ndimage.convolve(skeleton.astype(int), kernel.astype(int), mode='constant')

        branch_points = np.argwhere((neighbor_count > 4) & skeleton)
        endpoint_points = np.argwhere((neighbor_count == 2) & skeleton)

        self.branch_points = branch_points
        self.endpoint_points = endpoint_points

        return branch_points, endpoint_points

    def segment_vessels(self, vessel_mask):
        """Segment individual vessel branches"""
        if np.sum(vessel_mask) == 0:
            return np.zeros_like(vessel_mask), 0

        labeled_vessels, num_vessels = ndimage.label(vessel_mask > 0)
        self.vessel_branches = labeled_vessels

        return labeled_vessels, num_vessels

    def compute_vessel_properties(self, vessel_mask, skeleton, spacing=(1.0, 1.0, 1.0)):
        """Compute properties of vessels"""
        mean_spacing = np.mean(spacing)

        properties = {
            'total_vessel_volume_ml': float(np.sum(vessel_mask) * np.prod(spacing) / 1000),
            'total_skeleton_length_mm': float(np.sum(skeleton) * mean_spacing),
            'num_branch_points': int(len(self.branch_points)) if self.branch_points is not None else 0,
            'num_endpoints': int(len(self.endpoint_points)) if self.endpoint_points is not None else 0,
            'vessel_density': float(np.sum(vessel_mask) / vessel_mask.size),
        }

        return properties


class TumorVesselAnalyzer:
    """Analyze spatial relationships between tumor and vessels for TACE planning"""

    def __init__(self, critical_distance_mm=5.0, proximity_distance_mm=10.0):
        self.critical_distance_mm = critical_distance_mm
        self.proximity_distance_mm = proximity_distance_mm
        self.vessel_analyzer = VesselAnalyzer()

    def compute_distance_map(self, tumor_mask, spacing=(1.0, 1.0, 1.0)):
        """Compute distance from each point to tumor surface"""
        if np.sum(tumor_mask) == 0:
            warnings.warn("Empty tumor mask provided")
            return np.full_like(tumor_mask, np.inf, dtype=float)

        distance_mm = ndimage.distance_transform_edt(1 - tumor_mask, sampling=spacing)
        return distance_mm

    def identify_feeding_vessels(self, vessel_mask, tumor_mask, liver_mask=None,
                                 spacing=(1.0, 1.0, 1.0), vessel_type='arterial'):
        """Identify vessels that likely feed the tumor"""
        if np.sum(vessel_mask) == 0:
            warnings.warn("Empty vessel mask - cannot identify feeding vessels")
            return self._empty_feeding_analysis()

        if np.sum(tumor_mask) == 0:
            warnings.warn("Empty tumor mask - cannot analyze tumor-vessel relationship")
            return self._empty_feeding_analysis()

        # Compute distance from tumor
        distance_map = self.compute_distance_map(tumor_mask, spacing)

        # Extract vessel skeleton and topology
        skeleton = self.vessel_analyzer.extract_skeleton(vessel_mask)
        branch_points, endpoints = self.vessel_analyzer.find_branch_points(skeleton)

        # Classify vessels by proximity to tumor
        vessel_bool = vessel_mask.astype(bool)

        critical_vessels = vessel_bool & (distance_map <= self.critical_distance_mm)
        nearby_vessels = vessel_bool & (distance_map > self.critical_distance_mm) & \
                        (distance_map <= self.proximity_distance_mm)
        distant_vessels = vessel_bool & (distance_map > self.proximity_distance_mm)

        # Analyze vessel segments
        labeled_vessels, num_vessels = self.vessel_analyzer.segment_vessels(vessel_mask)

        critical_segments = []
        feeding_candidates = []

        for vessel_id in range(1, num_vessels + 1):
            vessel_segment = (labeled_vessels == vessel_id)
            segment_distances = distance_map[vessel_segment]

            if len(segment_distances) == 0:
                continue

            min_dist = float(segment_distances.min())
            mean_dist = float(segment_distances.mean())
            max_dist = float(segment_distances.max())
            segment_volume = float(np.sum(vessel_segment) * np.prod(spacing) / 1000)

            seg_coords = np.argwhere(vessel_segment)
            segment_skeleton = vessel_segment & skeleton
            skeleton_length = float(np.sum(segment_skeleton) * np.mean(spacing))

            segment_info = {
                'vessel_id': int(vessel_id),
                'min_distance_mm': min_dist,
                'mean_distance_mm': mean_dist,
                'max_distance_mm': max_dist,
                'volume_ml': segment_volume,
                'skeleton_length_mm': skeleton_length,
                'num_voxels': int(len(seg_coords)),
                'vessel_type': vessel_type,
                'tace_priority': self._assess_tace_priority(min_dist, mean_dist, vessel_type),
                'centroid_mm': (seg_coords.mean(axis=0) * spacing).tolist(),
            }

            if min_dist <= self.critical_distance_mm:
                critical_segments.append(segment_info)
                if mean_dist <= self.proximity_distance_mm:
                    feeding_candidates.append(segment_info)

        critical_segments = sorted(critical_segments,
                                  key=lambda x: (x['tace_priority'], -x['min_distance_mm']))
        feeding_candidates = sorted(feeding_candidates,
                                   key=lambda x: (x['tace_priority'], -x['min_distance_mm']))

        vessel_coords = np.argwhere(vessel_mask > 0)
        if len(vessel_coords) > 0:
            vessel_distances = distance_map[vessel_mask > 0]
            min_vessel_distance = float(vessel_distances.min())
            mean_vessel_distance = float(vessel_distances.mean())
        else:
            min_vessel_distance = np.inf
            mean_vessel_distance = np.inf

        results = {
            'vessel_type': vessel_type,
            'num_critical_segments': len(critical_segments),
            'num_feeding_candidates': len(feeding_candidates),
            'critical_segments': critical_segments,
            'feeding_candidates': feeding_candidates[:5],
            'critical_vessel_volume_ml': float(np.sum(critical_vessels) * np.prod(spacing) / 1000),
            'nearby_vessel_volume_ml': float(np.sum(nearby_vessels) * np.prod(spacing) / 1000),
            'distant_vessel_volume_ml': float(np.sum(distant_vessels) * np.prod(spacing) / 1000),
            'min_vessel_distance_mm': min_vessel_distance,
            'mean_vessel_distance_mm': mean_vessel_distance,
            'num_branch_points': len(branch_points),
            'num_endpoints': len(endpoints),
            'critical_distance_threshold_mm': self.critical_distance_mm,
            'proximity_distance_threshold_mm': self.proximity_distance_mm,
        }

        results['masks'] = {
            'critical': critical_vessels,
            'nearby': nearby_vessels,
            'distant': distant_vessels,
            'skeleton': skeleton,
            'branch_points': branch_points,
            'endpoints': endpoints,
            'labeled_vessels': labeled_vessels,
        }

        return results

    def _empty_feeding_analysis(self):
        """Return empty analysis structure"""
        return {
            'vessel_type': 'unknown',
            'num_critical_segments': 0,
            'num_feeding_candidates': 0,
            'critical_segments': [],
            'feeding_candidates': [],
            'critical_vessel_volume_ml': 0.0,
            'nearby_vessel_volume_ml': 0.0,
            'distant_vessel_volume_ml': 0.0,
            'min_vessel_distance_mm': np.inf,
            'mean_vessel_distance_mm': np.inf,
            'num_branch_points': 0,
            'num_endpoints': 0,
            'critical_distance_threshold_mm': self.critical_distance_mm,
            'proximity_distance_threshold_mm': self.proximity_distance_mm,
            'masks': {},
            'error': 'Empty or invalid input masks'
        }

    def _assess_tace_priority(self, min_distance, mean_distance, vessel_type):
        """Assess TACE targeting priority"""
        if vessel_type == 'arterial':
            if min_distance <= 2:
                return 1  # CRITICAL
            elif min_distance <= 5:
                return 2  # HIGH
            elif min_distance <= 10:
                return 3  # MODERATE
            else:
                return 4  # LOW
        else:
            return 5  # NOT A TARGET

    def analyze_tumor_liver_relationship(self, tumor_mask, liver_mask, spacing=(1.0, 1.0, 1.0)):
        """Analyze tumor position within liver"""
        if np.sum(tumor_mask) == 0 or np.sum(liver_mask) == 0:
            warnings.warn("Empty tumor or liver mask")
            return {
                'tumor_volume_ml': 0.0,
                'liver_volume_ml': 0.0,
                'tumor_to_liver_ratio': 0.0,
                'tumor_location': 'unknown',
                'distance_to_liver_surface_mm': np.inf,
                'error': 'Empty masks'
            }

        tumor_coords = np.argwhere(tumor_mask > 0)
        liver_coords = np.argwhere(liver_mask > 0)

        tumor_coords_mm = tumor_coords * spacing
        liver_coords_mm = liver_coords * spacing

        tumor_centroid = tumor_coords_mm.mean(axis=0)
        liver_centroid = liver_coords_mm.mean(axis=0)

        liver_distance = ndimage.distance_transform_edt(liver_mask, sampling=spacing)
        tumor_distance_to_surface = float(liver_distance[tumor_mask > 0].min())

        location = 'peripheral' if tumor_distance_to_surface < 20 else 'central'

        tumor_volume_ml = float(np.sum(tumor_mask) * np.prod(spacing) / 1000)
        liver_volume_ml = float(np.sum(liver_mask) * np.prod(spacing) / 1000)

        tumor_diameter_mm = self._estimate_tumor_diameter(tumor_mask, spacing)

        relationship = {
            'tumor_volume_ml': tumor_volume_ml,
            'liver_volume_ml': liver_volume_ml,
            'tumor_to_liver_ratio': float(tumor_volume_ml / liver_volume_ml) if liver_volume_ml > 0 else 0.0,
            'tumor_diameter_mm': tumor_diameter_mm,
            'tumor_location': location,
            'distance_to_liver_surface_mm': tumor_distance_to_surface,
            'tumor_centroid_mm': tumor_centroid.tolist(),
            'liver_centroid_mm': liver_centroid.tolist(),
            'liver_segment': self._estimate_liver_segment(tumor_centroid, liver_centroid),
        }

        return relationship

    def _estimate_tumor_diameter(self, tumor_mask, spacing):
        """Estimate tumor diameter from volume"""
        volume_mm3 = np.sum(tumor_mask) * np.prod(spacing)
        diameter = 2 * np.power(3 * volume_mm3 / (4 * np.pi), 1/3)
        return float(diameter)

    def _estimate_liver_segment(self, tumor_centroid, liver_centroid):
        """Rough estimation of liver segment"""
        rel_pos = tumor_centroid - liver_centroid

        if rel_pos[2] > 0:
            side = "Right"
        else:
            side = "Left"

        if rel_pos[1] > 0:
            position = "Anterior"
        else:
            position = "Posterior"

        return f"{side} lobe, {position} region (approximate)"


class TACEVisualizer:
    """Comprehensive visualization suite for TACE planning"""

    def __init__(self):
        self.color_scheme = {
            'tumor': '#8B0000',      # Dark red
            'liver': '#8B4513',      # Saddle brown
            'critical_vessel': '#DC143C',  # Crimson
            'nearby_vessel': '#FF8C00',    # Dark orange
            'distant_vessel': '#32CD32',   # Lime green
            'skeleton': '#00CED1',   # Dark turquoise
            'branch': '#FFD700',     # Gold
            'endpoint': '#FF69B4',   # Hot pink
        }

    def create_3d_interactive_visualization(self, volume, tumor_mask, vessel_mask, liver_mask,
                                           distance_map, vessel_analysis, spacing,
                                           save_path, patient_id):
        """
        Create interactive 3D HTML visualization showing tumor-vessel connections
        Uses actual CT intensities to construct realistic 3D structures
        """
        print(f"    Creating 3D interactive visualization...")

        # Downsample for performance (every 2nd voxel)
        downsample = 2
        volume_ds = volume[::downsample, ::downsample, ::downsample]
        tumor_ds = tumor_mask[::downsample, ::downsample, ::downsample]
        vessel_ds = vessel_mask[::downsample, ::downsample, ::downsample]
        distance_ds = distance_map[::downsample, ::downsample, ::downsample]

        spacing_ds = tuple(s * downsample for s in spacing)

        # Create figure
        fig = go.Figure()

        # Add tumor surface with CT intensity for realistic rendering
        if np.sum(tumor_ds) > 0:
            try:
                # Get tumor region from CT
                tumor_ct = volume_ds * tumor_ds
                tumor_ct[tumor_ct == 0] = volume_ds.min()

                verts, faces, normals, values = measure.marching_cubes(
                    tumor_ds, level=0.5, spacing=spacing_ds
                )

                # Get intensity values at vertices
                vert_indices = (verts / spacing_ds).astype(int)
                vert_indices = np.clip(vert_indices, 0, np.array(tumor_ds.shape) - 1)
                intensities = volume_ds[vert_indices[:, 0], vert_indices[:, 1], vert_indices[:, 2]]

                fig.add_trace(go.Mesh3d(
                    x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    intensity=intensities,
                    colorscale='Reds',
                    opacity=0.9,
                    name='Tumor',
                    hovertemplate='<b>Tumor</b><br>HU: %{intensity:.0f}<extra></extra>',
                    showlegend=True,
                    showscale=False
                ))
            except Exception as e:
                print(f"      Warning: Could not generate tumor surface: {e}")

        # Add vessel structures with CT intensity
        if np.sum(vessel_ds) > 0:
            try:
                # Get vessel region from CT
                vessel_ct = volume_ds * vessel_ds
                vessel_ct[vessel_ct == 0] = volume_ds.min()

                verts, faces, normals, values = measure.marching_cubes(
                    vessel_ds, level=0.5, spacing=spacing_ds
                )

                # Get intensity values at vertices
                vert_indices = (verts / spacing_ds).astype(int)
                vert_indices = np.clip(vert_indices, 0, np.array(vessel_ds.shape) - 1)
                intensities = volume_ds[vert_indices[:, 0], vert_indices[:, 1], vert_indices[:, 2]]

                # Color vessels by distance to tumor
                distances = distance_ds[vert_indices[:, 0], vert_indices[:, 1], vert_indices[:, 2]]

                fig.add_trace(go.Mesh3d(
                    x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    intensity=distances,
                    colorscale='RdYlGn',
                    cmin=0,
                    cmax=20,
                    opacity=0.8,
                    name='Vessels',
                    hovertemplate='<b>Vessel</b><br>Distance: %{intensity:.1f}mm<br>HU: ' +
                                 str(intensities.mean())[0:5] + '<extra></extra>',
                    showlegend=True,
                    colorbar=dict(
                        title='Distance<br>from Tumor<br>(mm)',
                        x=1.0,
                        len=0.5
                    )
                ))
            except Exception as e:
                print(f"      Warning: Could not generate vessel surface: {e}")

        # Add vessel skeleton for connectivity visualization
        masks = vessel_analysis.get('masks', {})
        skeleton = masks.get('skeleton')

        if skeleton is not None and np.sum(skeleton) > 0:
            skeleton_coords = np.argwhere(skeleton > 0)
            # Sample skeleton points (every 3rd point for performance)
            skeleton_coords = skeleton_coords[::3]
            skeleton_mm = skeleton_coords * spacing

            # Color by distance to tumor
            skeleton_distances = distance_map[skeleton_coords[:, 0],
                                             skeleton_coords[:, 1],
                                             skeleton_coords[:, 2]]

            fig.add_trace(go.Scatter3d(
                x=skeleton_mm[:, 2],
                y=skeleton_mm[:, 1],
                z=skeleton_mm[:, 0],
                mode='markers',
                marker=dict(
                    size=2,
                    color=skeleton_distances,
                    colorscale='RdYlGn',
                    cmin=0,
                    cmax=20,
                    showscale=False
                ),
                name='Vessel Centerlines',
                hovertemplate='<b>Vessel Centerline</b><br>Distance: %{marker.color:.1f}mm<extra></extra>',
                showlegend=True
            ))

        # Add branch points
        branch_points = masks.get('branch_points')
        if branch_points is not None and len(branch_points) > 0:
            bp_mm = branch_points * spacing
            bp_distances = distance_map[branch_points[:, 0], branch_points[:, 1], branch_points[:, 2]]

            fig.add_trace(go.Scatter3d(
                x=bp_mm[:, 2], y=bp_mm[:, 1], z=bp_mm[:, 0],
                mode='markers',
                marker=dict(
                    size=5,
                    color=bp_distances,
                    colorscale='RdYlGn',
                    cmin=0,
                    cmax=20,
                    symbol='diamond',
                    line=dict(color='black', width=1)
                ),
                name='Branch Points',
                hovertemplate='<b>Branch Point</b><br>Distance: %{marker.color:.1f}mm<extra></extra>',
                showlegend=True
            ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'3D Tumor-Vessel Anatomy - Patient {patient_id}<br>'
                     f'<sub>Color shows distance from tumor (mm) | Red=Close, Green=Far</sub>',
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                bgcolor='white'
            ),
            width=1200,
            height=900,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            hovermode='closest',
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        # Save
        fig.write_html(save_path)
        print(f"      Saved: {save_path}")

    def create_multiplane_slices(self, volume, tumor_mask, vessel_mask, liver_mask,
                                distance_map, vessel_analysis, spacing, save_path, patient_id):
        """
        Create multi-plane slice views showing tumor-vessel connections
        """
        print(f"    Creating multi-plane slice visualization...")

        # Find tumor center
        tumor_coords = np.argwhere(tumor_mask > 0)
        if len(tumor_coords) == 0:
            return

        tumor_center = tumor_coords.mean(axis=0).astype(int)
        z_slice, y_slice, x_slice = tumor_center

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Get vessel classifications
        masks = vessel_analysis.get('masks', {})
        critical_vessels = masks.get('critical', np.zeros_like(vessel_mask))
        nearby_vessels = masks.get('nearby', np.zeros_like(vessel_mask))

        # Row 1: Axial views
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_slice(ax1, volume[z_slice], 'gray', 'Axial - CT Image')

        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_overlay_slice(ax2, volume[z_slice], tumor_mask[z_slice],
                                 vessel_mask[z_slice], liver_mask[z_slice],
                                 'Axial - Segmentation Overlay')

        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_distance_slice(ax3, distance_map[z_slice], tumor_mask[z_slice],
                                  vessel_mask[z_slice], vessel_analysis,
                                  'Axial - Distance Map')

        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_vessel_classification_slice(ax4, critical_vessels[z_slice],
                                               nearby_vessels[z_slice],
                                               tumor_mask[z_slice],
                                               'Axial - Vessel Classification')

        # Row 2: Coronal views
        ax5 = fig.add_subplot(gs[1, 0])
        self._plot_slice(ax5, volume[:, y_slice, :], 'gray', 'Coronal - CT Image')

        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_overlay_slice(ax6, volume[:, y_slice, :], tumor_mask[:, y_slice, :],
                                 vessel_mask[:, y_slice, :], liver_mask[:, y_slice, :],
                                 'Coronal - Segmentation Overlay')

        ax7 = fig.add_subplot(gs[1, 2])
        self._plot_distance_slice(ax7, distance_map[:, y_slice, :], tumor_mask[:, y_slice, :],
                                  vessel_mask[:, y_slice, :], vessel_analysis,
                                  'Coronal - Distance Map')

        ax8 = fig.add_subplot(gs[1, 3])
        self._plot_vessel_classification_slice(ax8, critical_vessels[:, y_slice, :],
                                               nearby_vessels[:, y_slice, :],
                                               tumor_mask[:, y_slice, :],
                                               'Coronal - Vessel Classification')

        # Row 3: Sagittal views
        ax9 = fig.add_subplot(gs[2, 0])
        self._plot_slice(ax9, volume[:, :, x_slice], 'gray', 'Sagittal - CT Image')

        ax10 = fig.add_subplot(gs[2, 1])
        self._plot_overlay_slice(ax10, volume[:, :, x_slice], tumor_mask[:, :, x_slice],
                                 vessel_mask[:, :, x_slice], liver_mask[:, :, x_slice],
                                 'Sagittal - Segmentation Overlay')

        ax11 = fig.add_subplot(gs[2, 2])
        self._plot_distance_slice(ax11, distance_map[:, :, x_slice], tumor_mask[:, :, x_slice],
                                  vessel_mask[:, :, x_slice], vessel_analysis,
                                  'Sagittal - Distance Map')

        ax12 = fig.add_subplot(gs[2, 3])
        self._plot_vessel_classification_slice(ax12, critical_vessels[:, :, x_slice],
                                               nearby_vessels[:, :, x_slice],
                                               tumor_mask[:, :, x_slice],
                                               'Sagittal - Vessel Classification')

        plt.suptitle(f'Multi-Plane Analysis - Patient {patient_id}\n'
                    f'Center slice through tumor at (Z={z_slice}, Y={y_slice}, X={x_slice})',
                    fontsize=16, fontweight='bold')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      Saved: {save_path}")

    def _plot_slice(self, ax, img_slice, cmap, title):
        """Plot simple CT slice"""
        ax.imshow(img_slice, cmap=cmap, aspect='auto')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

    def _plot_overlay_slice(self, ax, img_slice, tumor_slice, vessel_slice, liver_slice, title):
        """Plot CT with segmentation overlay"""
        ax.imshow(img_slice, cmap='gray', aspect='auto')

        # Create overlay
        overlay = np.zeros((*img_slice.shape, 4))

        # Liver - brown, transparent
        overlay[liver_slice > 0] = [0.55, 0.27, 0.07, 0.2]

        # Vessels - green
        overlay[vessel_slice > 0] = [0.2, 0.8, 0.2, 0.6]

        # Tumor - red (on top)
        overlay[tumor_slice > 0] = [0.9, 0.1, 0.1, 0.7]

        ax.imshow(overlay, aspect='auto')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

        # Add legend
        tumor_patch = mpatches.Patch(color='red', label='Tumor')
        vessel_patch = mpatches.Patch(color='green', label='Vessels')
        liver_patch = mpatches.Patch(color='brown', alpha=0.3, label='Liver')
        ax.legend(handles=[tumor_patch, vessel_patch, liver_patch],
                 loc='upper right', fontsize=7)

    def _plot_distance_slice(self, ax, distance_slice, tumor_slice, vessel_slice,
                            vessel_analysis, title):
        """Plot distance map with iso-distance contours"""
        # Plot distance map
        im = ax.imshow(distance_slice, cmap='hot', aspect='auto', vmin=0, vmax=20)

        # Add contours for critical distances
        critical_dist = vessel_analysis.get('critical_distance_threshold_mm', 5)
        proximity_dist = vessel_analysis.get('proximity_distance_threshold_mm', 10)

        ax.contour(distance_slice, levels=[critical_dist], colors=['green'],
                  linewidths=2, linestyles='solid')
        ax.contour(distance_slice, levels=[proximity_dist], colors=['orange'],
                  linewidths=2, linestyles='dashed')

        # Overlay tumor and vessel boundaries
        ax.contour(tumor_slice, levels=[0.5], colors=['white'], linewidths=2)
        ax.contour(vessel_slice, levels=[0.5], colors=['cyan'], linewidths=1)

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Distance (mm)', fontsize=8)

        # Add legend
        critical_line = mpatches.Patch(color='green', label=f'Critical ({critical_dist}mm)')
        proximity_line = mpatches.Patch(color='orange', label=f'Proximity ({proximity_dist}mm)')
        ax.legend(handles=[critical_line, proximity_line], loc='upper right', fontsize=7)

    def _plot_vessel_classification_slice(self, ax, critical_slice, nearby_slice,
                                          tumor_slice, title):
        """Plot vessel classification"""
        # Create color-coded overlay
        overlay = np.zeros((*critical_slice.shape, 3))

        # Critical vessels - red
        overlay[critical_slice > 0] = [0.86, 0.08, 0.24]  # Crimson

        # Nearby vessels - orange
        overlay[nearby_slice > 0] = [1.0, 0.55, 0.0]  # Dark orange

        # Tumor outline
        tumor_bool = tumor_slice.astype(bool)
        tumor_boundary = ndimage.binary_dilation(tumor_bool) & ~tumor_bool
        overlay[tumor_boundary] = [1.0, 1.0, 1.0]  # White outline

        ax.imshow(overlay, aspect='auto')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

        # Add legend
        critical_patch = mpatches.Patch(color='crimson', label='Critical Vessels')
        nearby_patch = mpatches.Patch(color='darkorange', label='Nearby Vessels')
        tumor_patch = mpatches.Patch(color='white', label='Tumor Boundary')
        ax.legend(handles=[critical_patch, nearby_patch, tumor_patch],
                 loc='upper right', fontsize=7)

    def create_results_visualization(self, report, save_path):
        """
        Create comprehensive visualization showing analysis results for comparison
        Focus on measurements and findings, not risk assessment
        """
        print(f"    Creating results visualization...")

        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

        tumor_analysis = report.get('tumor_analysis', {})
        vessel_analysis = report.get('vessel_analysis', {})

        # 1. Tumor Measurements (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_tumor_measurements(ax1, tumor_analysis)

        # 2. Vessel Distance Distribution (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_vessel_distances(ax2, vessel_analysis)

        # 3. Tumor-Liver Relationship (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_tumor_liver_metrics(ax3, tumor_analysis)

        # 4. Feeding Vessel Details (Middle spanning 2 columns)
        ax4 = fig.add_subplot(gs[1, :2])
        self._plot_feeding_vessel_details(ax4, vessel_analysis)

        # 5. Vessel Topology Metrics (Middle Right)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_vessel_topology_metrics(ax5, vessel_analysis)

        # 6. Vessel Volume Distribution (Bottom Left)
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_vessel_volume_distribution(ax6, tumor_analysis, vessel_analysis)

        # 7. Segment Analysis (Bottom Middle)
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_segment_analysis(ax7, vessel_analysis)

        # 8. Distance Statistics (Bottom Right)
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_distance_statistics(ax8, vessel_analysis)

        # 9. Key Findings Summary (Bottom spanning)
        ax9 = fig.add_subplot(gs[3, :])
        self._plot_findings_summary(ax9, report)

        plt.suptitle(f'TACE Planning Results - Patient {report["patient_id"]}\n'
                    f'Anatomical Analysis and Measurements',
                    fontsize=18, fontweight='bold')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      Saved: {save_path}")

    def _plot_tumor_measurements(self, ax, tumor_analysis):
        """Plot tumor measurements"""
        metrics = {
            'Volume\n(mL)': tumor_analysis.get('tumor_volume_ml', 0),
            'Diameter\n(mm)': tumor_analysis.get('tumor_diameter_mm', 0),
            'Dist. to\nSurface\n(mm)': tumor_analysis.get('distance_to_liver_surface_mm', 0),
        }

        colors = ['darkred', 'red', 'coral']
        bars = ax.bar(metrics.keys(), metrics.values(), color=colors, edgecolor='black', linewidth=1.5)

        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

        ax.set_title('Tumor Measurements', fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        location = tumor_analysis.get('tumor_location', 'unknown')
        ax.text(0.5, 0.95, f'Location: {location.upper()}',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9, fontweight='bold')

    def _plot_vessel_distances(self, ax, vessel_analysis):
        """Plot vessel distance measurements"""
        min_dist = vessel_analysis.get('min_vessel_distance_mm', 0)
        mean_dist = vessel_analysis.get('mean_vessel_distance_mm', 0)
        critical_thresh = vessel_analysis.get('critical_distance_threshold_mm', 5)
        proximity_thresh = vessel_analysis.get('proximity_distance_threshold_mm', 10)

        categories = ['Min\nDistance', 'Mean\nDistance']
        values = [min_dist, mean_dist]
        colors = ['crimson', 'darkorange']

        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}mm', ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax.set_title('Vessel Distance Measurements', fontsize=11, fontweight='bold')
        ax.set_ylabel('Distance (mm)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add reference lines
        ax.axhline(y=critical_thresh, color='red', linestyle='--', alpha=0.6, linewidth=2,
                  label=f'Critical: {critical_thresh}mm')
        ax.axhline(y=proximity_thresh, color='orange', linestyle='--', alpha=0.6, linewidth=2,
                  label=f'Proximity: {proximity_thresh}mm')
        ax.legend(fontsize=8)

    def _plot_tumor_liver_metrics(self, ax, tumor_analysis):
        """Plot tumor-liver relationship metrics"""
        tumor_vol = tumor_analysis.get('tumor_volume_ml', 0)
        liver_vol = tumor_analysis.get('liver_volume_ml', 0)
        ratio = tumor_analysis.get('tumor_to_liver_ratio', 0) * 100

        # Create horizontal bar chart
        metrics = ['Tumor\nVolume', 'Liver\nVolume', 'Tumor/Liver\nRatio']
        values = [tumor_vol, liver_vol, ratio]
        colors = ['darkred', 'brown', 'orange']
        units = ['mL', 'mL', '%']

        y_pos = np.arange(len(metrics))
        bars = ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)

        # Add value labels
        for i, (bar, value, unit) in enumerate(zip(bars, values, units)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {value:.1f} {unit}', ha='left', va='center', fontweight='bold', fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics, fontweight='bold')
        ax.set_xlabel('Value', fontweight='bold')
        ax.set_title('Tumor-Liver Relationship', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    def _plot_feeding_vessel_details(self, ax, vessel_analysis):
        """Plot detailed feeding vessel information"""
        feeding_vessels = vessel_analysis.get('feeding_candidates', [])

        if len(feeding_vessels) == 0:
            ax.text(0.5, 0.5, 'No feeding vessels identified',
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_title('Feeding Vessel Analysis', fontsize=11, fontweight='bold')
            ax.axis('off')
            return

        # Prepare data
        vessel_ids = [f"V{v['vessel_id']}" for v in feeding_vessels[:5]]
        distances = [v['min_distance_mm'] for v in feeding_vessels[:5]]
        volumes = [v['volume_ml'] for v in feeding_vessels[:5]]
        lengths = [v['skeleton_length_mm'] for v in feeding_vessels[:5]]

        x = np.arange(len(vessel_ids))
        width = 0.25

        # Create grouped bars
        bars1 = ax.bar(x - width, distances, width, label='Min Distance (mm)',
                      color='crimson', alpha=0.7, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x, volumes, width, label='Volume (mL)',
                      color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
        bars3 = ax.bar(x + width, [l/10 for l in lengths], width, label='Length (cm)',
                      color='green', alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Vessel ID', fontweight='bold')
        ax.set_ylabel('Measurement', fontweight='bold')
        ax.set_title('Feeding Vessel Details', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(vessel_ids, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    def _plot_vessel_topology_metrics(self, ax, vessel_analysis):
        """Plot vessel network topology"""
        metrics = {
            'Feeding\nVessels': vessel_analysis.get('num_feeding_candidates', 0),
            'Critical\nSegments': vessel_analysis.get('num_critical_segments', 0),
            'Branch\nPoints': vessel_analysis.get('num_branch_points', 0),
            'Endpoints': vessel_analysis.get('num_endpoints', 0),
        }

        colors = ['darkgreen', 'green', 'gold', 'pink']
        bars = ax.bar(metrics.keys(), metrics.values(), color=colors,
                     edgecolor='black', linewidth=1.5, alpha=0.7)

        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax.set_title('Vessel Network Topology', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    def _plot_vessel_volume_distribution(self, ax, tumor_analysis, vessel_analysis):
        """Plot volume distribution by proximity"""
        vessel_volumes = {
            'Critical\nVessels': vessel_analysis.get('critical_vessel_volume_ml', 0),
            'Nearby\nVessels': vessel_analysis.get('nearby_vessel_volume_ml', 0),
            'Distant\nVessels': vessel_analysis.get('distant_vessel_volume_ml', 0),
        }

        colors = ['crimson', 'darkorange', 'limegreen']
        bars = ax.bar(vessel_volumes.keys(), vessel_volumes.values(),
                     color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)

        for bar, value in zip(bars, vessel_volumes.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        ax.set_title('Vessel Volume by Proximity', fontsize=11, fontweight='bold')
        ax.set_ylabel('Volume (mL)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add tumor volume reference
        tumor_vol = tumor_analysis.get('tumor_volume_ml', 0)
        ax.axhline(y=tumor_vol, color='red', linestyle='--', linewidth=2,
                  label=f'Tumor: {tumor_vol:.1f} mL', alpha=0.7)
        ax.legend(fontsize=8)

    def _plot_segment_analysis(self, ax, vessel_analysis):
        """Plot critical segment analysis"""
        critical_segments = vessel_analysis.get('critical_segments', [])

        if len(critical_segments) == 0:
            ax.text(0.5, 0.5, 'No critical segments',
                   ha='center', va='center', fontsize=12)
            ax.set_title('Critical Segment Analysis', fontsize=11, fontweight='bold')
            ax.axis('off')
            return

        # Get top 5 segments
        segments = critical_segments[:5]
        seg_ids = [f"S{s['vessel_id']}" for s in segments]
        min_dists = [s['min_distance_mm'] for s in segments]
        mean_dists = [s['mean_distance_mm'] for s in segments]

        x = np.arange(len(seg_ids))
        width = 0.35

        bars1 = ax.bar(x - width/2, min_dists, width, label='Min Distance',
                      color='red', alpha=0.7, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, mean_dists, width, label='Mean Distance',
                      color='orange', alpha=0.7, edgecolor='black', linewidth=1.5)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Segment ID', fontweight='bold')
        ax.set_ylabel('Distance (mm)', fontweight='bold')
        ax.set_title('Critical Segment Analysis', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(seg_ids, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    def _plot_distance_statistics(self, ax, vessel_analysis):
        """Plot distance statistics summary"""
        stats_text = []

        stats_text.append(f"VESSEL DISTANCE STATISTICS")
        stats_text.append(f"")
        stats_text.append(f"Minimum Distance:")
        stats_text.append(f"  {vessel_analysis.get('min_vessel_distance_mm', 0):.2f} mm")
        stats_text.append(f"")
        stats_text.append(f"Mean Distance:")
        stats_text.append(f"  {vessel_analysis.get('mean_vessel_distance_mm', 0):.2f} mm")
        stats_text.append(f"")
        stats_text.append(f"Critical Threshold:")
        stats_text.append(f"  ≤{vessel_analysis.get('critical_distance_threshold_mm', 5)} mm")
        stats_text.append(f"")
        stats_text.append(f"Proximity Threshold:")
        stats_text.append(f"  ≤{vessel_analysis.get('proximity_distance_threshold_mm', 10)} mm")
        stats_text.append(f"")
        stats_text.append(f"Vessel Counts:")
        stats_text.append(f"  Critical: {vessel_analysis.get('num_critical_segments', 0)}")
        stats_text.append(f"  Feeding: {vessel_analysis.get('num_feeding_candidates', 0)}")

        ax.text(0.1, 0.95, '\n'.join(stats_text), transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        ax.set_title('Summary Statistics', fontsize=11, fontweight='bold')
        ax.axis('off')

    def _plot_findings_summary(self, ax, report):
        """Plot key findings without risk assessment"""
        tumor_analysis = report.get('tumor_analysis', {})
        vessel_analysis = report.get('vessel_analysis', {})

        findings = []

        # Tumor findings
        findings.append(f"⚫ TUMOR: {tumor_analysis.get('tumor_volume_ml', 0):.1f} mL, "
                       f"{tumor_analysis.get('tumor_diameter_mm', 0):.1f} mm diameter, "
                       f"{tumor_analysis.get('tumor_location', 'unknown')} location")

        # Vessel findings
        num_feeding = vessel_analysis.get('num_feeding_candidates', 0)
        min_dist = vessel_analysis.get('min_vessel_distance_mm', 0)
        findings.append(f"⚫ VESSELS: {num_feeding} feeding vessel(s) identified, "
                       f"minimum distance {min_dist:.1f} mm from tumor")

        # Critical segments
        num_critical = vessel_analysis.get('num_critical_segments', 0)
        if num_critical > 0:
            findings.append(f"⚫ CRITICAL SEGMENTS: {num_critical} vessel segment(s) "
                          f"within critical distance")

        # Topology
        num_branches = vessel_analysis.get('num_branch_points', 0)
        findings.append(f"⚫ TOPOLOGY: {num_branches} branch point(s) detected in vessel network")

        # Liver segment
        liver_seg = tumor_analysis.get('liver_segment', 'unknown')
        findings.append(f"⚫ LOCATION: {liver_seg}")

        y_pos = 0.9
        ax.text(0.05, y_pos, 'KEY FINDINGS', fontsize=12, fontweight='bold',
               transform=ax.transAxes)
        y_pos -= 0.15

        for finding in findings:
            ax.text(0.05, y_pos, finding, fontsize=10,
                   transform=ax.transAxes, wrap=True)
            y_pos -= 0.12

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')


class TACEPlanner:
    """Generate comprehensive TACE planning report with visualizations"""

    def __init__(self, critical_distance_mm=5.0, proximity_distance_mm=10.0):
        self.critical_distance = critical_distance_mm
        self.proximity_distance = proximity_distance_mm
        self.tumor_vessel_analyzer = TumorVesselAnalyzer(critical_distance_mm, proximity_distance_mm)
        self.visualizer = TACEVisualizer()

    def analyze_patient(self, scan_folder, seg_file, patient_id, spacing=(1.0, 1.0, 1.0)):
        """Complete TACE planning analysis for one patient"""
        print(f"\n{'='*80}")
        print(f"Analyzing patient {patient_id} for TACE procedure...")
        print(f"{'='*80}")

        try:
            # Load data
            print("  Loading imaging data...")
            volume, metadata = read_dicom_series(scan_folder)

            if 'spacing' in metadata and metadata['spacing'] is not None:
                spacing = metadata['spacing']
                print(f"    ✓ Using spacing from metadata: {spacing}")
            else:
                print(f"    ✓ Using provided spacing: {spacing}")

            # Load segmentation masks
            print("  Loading segmentation masks...")
            tumor_mask = read_segmentation_dicom(seg_file, class_number=2)
            liver_mask = read_segmentation_dicom(seg_file, class_number=1)
            vessel_mask = read_segmentation_dicom(seg_file, class_number=3)
            aorta_mask = read_segmentation_dicom(seg_file, class_number=4)

            # Validate masks
            if np.sum(tumor_mask) == 0:
                raise ValueError("Tumor mask is empty")
            if np.sum(liver_mask) == 0:
                raise ValueError("Liver mask is empty")

            print(f"    ✓ Tumor voxels: {np.sum(tumor_mask)}")
            print(f"    ✓ Liver voxels: {np.sum(liver_mask)}")
            print(f"    ✓ Vessel voxels: {np.sum(vessel_mask)}")

            report = {
                'patient_id': patient_id,
                'spacing_mm': list(spacing),
                'volume_shape': list(volume.shape),
                'analysis_date': pd.Timestamp.now().isoformat(),
            }

            # 1. Analyze tumor-liver relationship
            print("  Analyzing tumor characteristics...")
            tumor_liver = self.tumor_vessel_analyzer.analyze_tumor_liver_relationship(
                tumor_mask, liver_mask, spacing
            )
            report['tumor_analysis'] = tumor_liver
            print(f"    ✓ Tumor volume: {tumor_liver['tumor_volume_ml']:.1f} mL")
            print(f"    ✓ Tumor location: {tumor_liver['tumor_location']}")

            # 2. Analyze arterial vessels (primary TACE targets)
            print("  Identifying feeding vessels...")
            if np.sum(vessel_mask) > 0:
                vessel_analysis = self.tumor_vessel_analyzer.identify_feeding_vessels(
                    vessel_mask, tumor_mask, liver_mask, spacing, vessel_type='arterial'
                )
                print(f"    ✓ Feeding candidates: {vessel_analysis['num_feeding_candidates']}")
                print(f"    ✓ Min vessel distance: {vessel_analysis['min_vessel_distance_mm']:.1f} mm")
            else:
                warnings.warn(f"No vessel mask found for patient {patient_id}")
                vessel_analysis = self.tumor_vessel_analyzer._empty_feeding_analysis()

            report['vessel_analysis'] = vessel_analysis

            # Compute distance map for visualizations
            distance_map = self.tumor_vessel_analyzer.compute_distance_map(tumor_mask, spacing)
            report['_distance_map'] = distance_map  # Store for visualization
            report['_volume'] = volume  # Store for visualization
            report['_masks'] = {
                'tumor': tumor_mask,
                'liver': liver_mask,
                'vessel': vessel_mask,
                'aorta': aorta_mask
            }

            # 3. Analyze aorta proximity
            print("  Analyzing aorta proximity...")
            if np.sum(aorta_mask) > 0:
                distance_to_aorta = ndimage.distance_transform_edt(1 - aorta_mask, sampling=spacing)
                tumor_aorta_distance = float(distance_to_aorta[tumor_mask > 0].min())

                aorta_coords = np.argwhere(aorta_mask > 0)
                if len(aorta_coords) > 0:
                    aorta_centroid = (aorta_coords.mean(axis=0) * spacing).tolist()
                else:
                    aorta_centroid = None
            else:
                warnings.warn(f"No aorta mask found for patient {patient_id}")
                tumor_aorta_distance = np.inf
                aorta_centroid = None

            report['aorta_analysis'] = {
                'min_distance_mm': tumor_aorta_distance,
                'aorta_centroid_mm': aorta_centroid,
            }

            # 4. TACE feasibility assessment
            print("  Assessing TACE feasibility...")
            feasibility = self._assess_tace_feasibility(report)
            report['tace_feasibility'] = feasibility
            print(f"    ✓ Feasibility score: {feasibility['feasibility_score']}/{feasibility['max_score']}")
            print(f"    ✓ Complexity: {feasibility['complexity']}")

            # 5. Generate TACE-specific recommendations
            print("  Generating TACE recommendations...")
            recommendations = self._generate_tace_recommendations(report)
            report['tace_recommendations'] = recommendations
            print(f"    ✓ Generated {len(recommendations)} recommendations")

            # 6. Procedural planning
            print("  Creating procedural plan...")
            procedure_plan = self._create_procedure_plan(report)
            report['procedure_plan'] = procedure_plan

            return report

        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
            raise

    def _assess_tace_feasibility(self, report):
        """Assess TACE procedure feasibility"""
        tumor_analysis = report['tumor_analysis']
        vessel_analysis = report['vessel_analysis']

        factors = []
        feasibility_score = 0

        # 1. Tumor size
        tumor_volume = tumor_analysis['tumor_volume_ml']
        if tumor_volume < 100:
            factors.append(f"Good tumor size ({tumor_volume:.1f} mL, <100mL)")
            feasibility_score += 3
        elif tumor_volume < 200:
            factors.append(f"Moderate tumor size ({tumor_volume:.1f} mL, 100-200mL)")
            feasibility_score += 2
        else:
            factors.append(f"Large tumor ({tumor_volume:.1f} mL, >200mL) - may need staged TACE")
            feasibility_score += 1

        # 2. Feeding vessels
        num_feeders = vessel_analysis['num_feeding_candidates']
        if num_feeders >= 2:
            factors.append(f"Multiple feeding vessels identified ({num_feeders}) - good targets")
            feasibility_score += 3
        elif num_feeders == 1:
            factors.append(f"Single feeding vessel identified - targetable")
            feasibility_score += 2
        else:
            factors.append("No clear feeding vessels identified - may need advanced imaging")
            feasibility_score += 0

        # 3. Vessel proximity
        min_distance = vessel_analysis['min_vessel_distance_mm']
        if min_distance <= 2:
            factors.append(f"Excellent vessel proximity ({min_distance:.1f}mm) - direct tumor supply")
            feasibility_score += 3
        elif min_distance <= 5:
            factors.append(f"Good vessel proximity ({min_distance:.1f}mm)")
            feasibility_score += 2
        elif min_distance <= 10:
            factors.append(f"Moderate vessel proximity ({min_distance:.1f}mm)")
            feasibility_score += 1
        else:
            factors.append(f"Distant vessels ({min_distance:.1f}mm) - may need angiography")
            feasibility_score += 0

        # 4. Tumor location
        location = tumor_analysis['tumor_location']
        if location == 'peripheral':
            factors.append("Peripheral location - easier vascular access")
            feasibility_score += 1
        else:
            factors.append("Central location - may need selective catheterization")
            feasibility_score += 0

        # 5. Liver reserve
        tumor_liver_ratio = tumor_analysis['tumor_to_liver_ratio']
        if tumor_liver_ratio < 0.1:
            factors.append(f"Good liver reserve (tumor {tumor_liver_ratio*100:.1f}% of liver)")
            feasibility_score += 2
        elif tumor_liver_ratio < 0.2:
            factors.append(f"Adequate liver reserve (tumor {tumor_liver_ratio*100:.1f}% of liver)")
            feasibility_score += 1
        else:
            factors.append(f"Limited liver reserve (tumor {tumor_liver_ratio*100:.1f}% of liver)")
            feasibility_score += 0

        max_score = 12
        feasibility_percentage = (feasibility_score / max_score) * 100

        if feasibility_score >= 10:
            feasibility = "EXCELLENT - Ideal candidate for TACE"
            complexity = "LOW"
        elif feasibility_score >= 7:
            feasibility = "GOOD - Standard TACE procedure feasible"
            complexity = "MODERATE"
        elif feasibility_score >= 4:
            feasibility = "FAIR - TACE possible with careful planning"
            complexity = "MODERATE-HIGH"
        else:
            feasibility = "CHALLENGING - Consider advanced imaging or alternative approach"
            complexity = "HIGH"

        return {
            'feasibility_score': feasibility_score,
            'max_score': max_score,
            'feasibility_percentage': feasibility_percentage,
            'complexity': complexity,
            'overall_assessment': feasibility,
            'contributing_factors': factors,
        }

    def _generate_tace_recommendations(self, report):
        """Generate TACE-specific recommendations"""
        recommendations = []

        vessel_analysis = report['vessel_analysis']
        feasibility = report['tace_feasibility']
        tumor_analysis = report['tumor_analysis']

        # Catheter approach
        if feasibility['complexity'] in ['LOW', 'MODERATE']:
            recommendations.append({
                'category': 'Catheter Approach',
                'recommendation': 'Standard femoral approach via celiac axis to hepatic artery',
                'priority': 'PRIMARY',
                'details': 'Straightforward vascular anatomy expected'
            })
        else:
            recommendations.append({
                'category': 'Catheter Approach',
                'recommendation': 'Consider angiography to map arterial anatomy before TACE',
                'priority': 'HIGH',
                'details': 'Complex vascular anatomy or variant vessels possible'
            })

        # Target vessels
        feeding_vessels = vessel_analysis.get('feeding_candidates', [])
        if len(feeding_vessels) > 0:
            for idx, vessel in enumerate(feeding_vessels[:3], 1):
                recommendations.append({
                    'category': 'Target Vessels',
                    'recommendation': f"Target vessel #{idx}: Priority {vessel['tace_priority']}, "
                                    f"{vessel['min_distance_mm']:.1f}mm from tumor center",
                    'priority': f"PRIORITY_{vessel['tace_priority']}",
                    'details': f"Volume: {vessel['volume_ml']:.2f}mL, Length: ~{vessel['skeleton_length_mm']:.1f}mm"
                })
        else:
            recommendations.append({
                'category': 'Target Vessels',
                'recommendation': 'No clear feeding vessels identified on pre-procedure CT',
                'priority': 'CRITICAL',
                'details': 'Recommend contrast-enhanced angiography during procedure'
            })

        # Chemotherapy
        tumor_size = tumor_analysis['tumor_volume_ml']
        if tumor_size < 50:
            drug_rec = "Standard drug-eluting beads (DEB-TACE) with doxorubicin"
        elif tumor_size < 150:
            drug_rec = "DEB-TACE with doxorubicin or conventional TACE with lipiodol"
        else:
            drug_rec = "Consider staged TACE or conventional TACE with lipiodol"

        recommendations.append({
            'category': 'Chemotherapy Agent',
            'recommendation': drug_rec,
            'priority': 'STANDARD',
            'details': f"Based on tumor volume: {tumor_size:.1f}mL"
        })

        # Embolization strategy
        if vessel_analysis['num_feeding_candidates'] > 2:
            recommendations.append({
                'category': 'Embolization Strategy',
                'recommendation': 'Selective embolization of multiple feeding branches',
                'priority': 'HIGH',
                'details': f"{vessel_analysis['num_feeding_candidates']} feeding vessels identified"
            })

        # Post-procedure
        recommendations.append({
            'category': 'Post-Procedure',
            'recommendation': 'Follow-up CT at 4-6 weeks to assess tumor response (mRECIST criteria)',
            'priority': 'STANDARD',
            'details': 'Assess for tumor necrosis and plan additional TACE if needed'
        })

        return recommendations

    def _create_procedure_plan(self, report):
        """Create step-by-step procedure plan"""
        vessel_analysis = report['vessel_analysis']
        tumor_analysis = report['tumor_analysis']

        procedure_steps = [
            {
                'step': 1,
                'phase': 'Vascular Access',
                'action': 'Obtain femoral artery access using Seldinger technique',
                'equipment': '5F or 6F introducer sheath',
                'contrast': 'Minimal',
            },
            {
                'step': 2,
                'phase': 'Celiac Catheterization',
                'action': 'Advance catheter to celiac axis, obtain angiogram',
                'equipment': 'Cobra or Simmons catheter',
                'contrast': 'Yes - celiac angiogram',
            },
            {
                'step': 3,
                'phase': 'Hepatic Artery Selection',
                'action': f'Selective catheterization of hepatic artery feeding tumor in {tumor_analysis.get("liver_segment", "target segment")}',
                'equipment': 'Microcatheter (2.7F or 2.8F)',
                'contrast': 'Yes - selective angiogram',
            },
            {
                'step': 4,
                'phase': 'Target Identification',
                'action': f'Identify {vessel_analysis["num_feeding_candidates"]} feeding vessel(s) using selective angiography',
                'equipment': 'Microcatheter in super-selective position',
                'contrast': 'Yes - confirm tumor blush',
            },
            {
                'step': 5,
                'phase': 'Chemoembolization',
                'action': 'Deliver drug-eluting beads or lipiodol-drug mixture to feeding vessels',
                'equipment': 'Chemotherapy agent + embolic material',
                'contrast': 'Check flow after each aliquot',
            },
            {
                'step': 6,
                'phase': 'Completion',
                'action': 'Obtain completion angiogram to confirm adequate embolization',
                'equipment': 'Diagnostic catheter',
                'contrast': 'Yes - hepatic angiogram',
            },
            {
                'step': 7,
                'phase': 'Closure',
                'action': 'Remove catheters and achieve hemostasis',
                'equipment': 'Manual compression or closure device',
                'contrast': 'None',
            },
        ]

        plan = {
            'procedure_steps': procedure_steps,
            'estimated_contrast_volume': '100-150 mL',
            'estimated_procedure_time': '60-90 minutes',
            'estimated_fluoroscopy_time': '15-30 minutes',
            'special_notes': self._generate_special_notes(report),
        }

        return plan

    def _generate_special_notes(self, report):
        """Generate special procedural notes"""
        notes = []

        tumor_analysis = report['tumor_analysis']
        vessel_analysis = report['vessel_analysis']

        if tumor_analysis['tumor_volume_ml'] > 150:
            notes.append("Large tumor - consider staged TACE (2-3 sessions)")

        if vessel_analysis['num_feeding_candidates'] == 0:
            notes.append("Feeding vessels not identified on CT - intra-procedural angiography critical")

        if vessel_analysis['num_feeding_candidates'] > 3:
            notes.append("Multiple feeding vessels - may need extended procedure time")

        if tumor_analysis['tumor_location'] == 'central':
            notes.append("Central location - careful identification of portal vein branches")

        return notes

    def save_report(self, report, output_path):
        """Save report to JSON file"""
        report_clean = report.copy()

        # Remove data not needed in JSON
        for key in ['vessel_analysis', '_distance_map', '_volume', '_masks']:
            if key in report_clean:
                if key == 'vessel_analysis' and 'masks' in report_clean[key]:
                    del report_clean[key]['masks']
                elif key.startswith('_'):
                    del report_clean[key]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report_clean, f, indent=2)

        print(f"  ✓ Report saved to {output_path}")

    def visualize_patient(self, report, output_dir):
        """Create comprehensive visualizations for TACE planning"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        patient_id = report['patient_id']

        print(f"  Creating visualizations...")

        # Extract data
        volume = report.get('_volume')
        distance_map = report.get('_distance_map')
        masks = report.get('_masks', {})
        tumor_mask = masks.get('tumor')
        vessel_mask = masks.get('vessel')
        liver_mask = masks.get('liver')
        spacing = tuple(report['spacing_mm'])
        vessel_analysis = report['vessel_analysis']

        # 1. 3D Interactive Visualization (HTML)
        if all(x is not None for x in [volume, tumor_mask, vessel_mask, distance_map]):
            self.visualizer.create_3d_interactive_visualization(
                volume, tumor_mask, vessel_mask, liver_mask, distance_map,
                vessel_analysis, spacing,
                f"{output_dir}/{patient_id}_3d_interactive.html",
                patient_id
            )

        # 2. Multi-plane Slice Views
        if all(x is not None for x in [volume, tumor_mask, vessel_mask, liver_mask, distance_map]):
            self.visualizer.create_multiplane_slices(
                volume, tumor_mask, vessel_mask, liver_mask,
                distance_map, vessel_analysis, spacing,
                f"{output_dir}/{patient_id}_multiplane.png",
                patient_id
            )

        # 3. Results Comparison Visualization
        self.visualizer.create_results_visualization(
            report,
            f"{output_dir}/{patient_id}_results.png"
        )

        print(f"  ✓ All visualizations created")


def run_tace_planning(metadata_csv, root_dir, output_dir, clinical_csv=None):
    """Run TACE planning for all patients"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_csv(metadata_csv)
    planner = TACEPlanner(critical_distance_mm=5.0, proximity_distance_mm=10.0)

    all_reports = []

    print("\n" + "="*80)
    print("TACE PRE-PROCEDURE PLANNING ANALYSIS WITH COMPREHENSIVE VISUALIZATIONS")
    print("="*80)
    print(f"\nProcessing {len(metadata_df)} patients...")
    print(f"Output directory: {output_dir}\n")

    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Analyzing patients"):
        patient_id = row['PATIENT_ID']
        scan_folder = f"{root_dir}/{row['SCAN_FOLDER']}"
        seg_file = f"{root_dir}/{row['SEGMENTATION_FILE']}"

        try:
            # Analyze patient
            report = planner.analyze_patient(scan_folder, seg_file, patient_id)

            # Save JSON report
            report_path = f"{output_dir}/{patient_id}_tace_plan.json"
            planner.save_report(report, report_path)

            # Create visualizations
            vis_dir = f"{output_dir}/visualizations"
            planner.visualize_patient(report, vis_dir)

            # Remove large data before storing
            if '_distance_map' in report:
                del report['_distance_map']
            if '_volume' in report:
                del report['_volume']
            if '_masks' in report:
                del report['_masks']

            all_reports.append(report)
            break

        except Exception as e:
            print(f"\n  ❌ Error processing {patient_id}: {str(e)}")
            import traceback
            traceback.print_exc()

            failed_report = {
                'patient_id': patient_id,
                'status': 'error',
                'error_message': str(e),
                'tace_feasibility': {
                    'overall_assessment': 'Analysis Failed',
                    'complexity': 'UNKNOWN',
                    'feasibility_score': 0,
                    'contributing_factors': ['Analysis error - manual review required']
                }
            }
            all_reports.append(failed_report)
            break
            continue

    # Create summary
    print("\n" + "="*80)
    print("Generating summary report...")
    from tace_planning import create_tace_summary
    summary = create_tace_summary(all_reports, output_dir)

    print(f"\n{'='*80}")
    print("TACE PLANNING COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Processed: {len(all_reports)} patients")
    print(f"✓ Successful: {len([r for r in all_reports if r.get('status') != 'error'])}")
    print(f"✓ Failed: {len([r for r in all_reports if r.get('status') == 'error'])}")
    print(f"✓ Reports saved to: {output_dir}")
    print(f"✓ Visualizations saved to: {output_dir}/visualizations")
    print(f"  - 3D interactive HTML: Tumor-vessel anatomy with CT intensities")
    print(f"  - Multi-plane slices: Distance maps and vessel classification")
    print(f"  - Results visualization: Comprehensive measurements and findings")
    print(f"{'='*80}\n")

    return all_reports


def create_tace_summary(reports, output_dir):
    """Create summary statistics across all patients"""
    summary_data = []

    for report in reports:
        if report.get('status') == 'error':
            summary_data.append({
                'patient_id': report['patient_id'],
                'tumor_volume_ml': 0.0,
                'tumor_diameter_mm': 0.0,
                'tumor_location': 'Unknown',
                'num_feeding_vessels': 0,
                'num_critical_segments': 0,
                'min_vessel_distance_mm': 0.0,
                'feasibility_score': 0,
                'complexity': 'UNKNOWN',
                'overall_assessment': 'Error',
                'status': 'error'
            })
        else:
            tumor = report.get('tumor_analysis', {})
            vessel = report.get('vessel_analysis', {})
            feasibility = report.get('tace_feasibility', {})

            summary_data.append({
                'patient_id': report['patient_id'],
                'tumor_volume_ml': tumor.get('tumor_volume_ml', 0.0),
                'tumor_diameter_mm': tumor.get('tumor_diameter_mm', 0.0),
                'tumor_location': tumor.get('tumor_location', 'Unknown'),
                'num_feeding_vessels': vessel.get('num_feeding_candidates', 0),
                'num_critical_segments': vessel.get('num_critical_segments', 0),
                'min_vessel_distance_mm': vessel.get('min_vessel_distance_mm', np.inf),
                'feasibility_score': feasibility.get('feasibility_score', 0),
                'complexity': feasibility.get('complexity', 'UNKNOWN'),
                'overall_assessment': feasibility.get('overall_assessment', 'Unknown'),
                'status': 'success'
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}/tace_planning_summary.csv", index=False)

    # Create summary visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    valid_df = summary_df[summary_df['status'] == 'success']

    # 1. Complexity distribution
    ax = axes[0, 0]
    if len(valid_df) > 0:
        complexity_counts = valid_df['complexity'].value_counts()
        colors = ['lightgreen' if 'LOW' in c else 'yellow' if 'MODERATE' in c else 'coral'
                 for c in complexity_counts.index]
        ax.pie(complexity_counts.values, labels=complexity_counts.index, autopct='%1.1f%%',
               colors=colors, startangle=90)
    ax.set_title('TACE Complexity Distribution', fontweight='bold', fontsize=12)

    # 2. Feasibility scores
    ax = axes[0, 1]
    if len(valid_df) > 0:
        ax.hist(valid_df['feasibility_score'], bins=12, color='steelblue',
               edgecolor='black', alpha=0.7)
        ax.axvline(x=valid_df['feasibility_score'].mean(), color='red',
                  linestyle='--', linewidth=2, label=f'Mean: {valid_df["feasibility_score"].mean():.1f}')
        ax.legend()
    ax.set_xlabel('Feasibility Score', fontweight='bold')
    ax.set_ylabel('Number of Patients')
    ax.set_title('TACE Feasibility Score Distribution', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # 3. Tumor volumes
    ax = axes[0, 2]
    if len(valid_df) > 0:
        ax.hist(valid_df['tumor_volume_ml'], bins=15, color='darkred',
               alpha=0.7, edgecolor='black')
        ax.axvline(x=100, color='orange', linestyle='--', linewidth=2,
                  label='100mL threshold')
        ax.legend()
    ax.set_xlabel('Tumor Volume (mL)', fontweight='bold')
    ax.set_ylabel('Number of Patients')
    ax.set_title('Tumor Volume Distribution', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # 4. Feeding vessels
    ax = axes[1, 0]
    if len(valid_df) > 0:
        feeding_counts = valid_df['num_feeding_vessels'].value_counts().sort_index()
        ax.bar(feeding_counts.index, feeding_counts.values, color='darkgreen',
              edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Feeding Vessels', fontweight='bold')
    ax.set_ylabel('Number of Patients')
    ax.set_title('Feeding Vessel Identification', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # 5. Distance vs feasibility
    ax = axes[1, 1]
    if len(valid_df) > 0:
        plot_df = valid_df[valid_df['min_vessel_distance_mm'] != np.inf]
        if len(plot_df) > 0:
            scatter = ax.scatter(plot_df['min_vessel_distance_mm'],
                               plot_df['feasibility_score'],
                               c=plot_df['feasibility_score'], cmap='RdYlGn',
                               s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
            plt.colorbar(scatter, ax=ax, label='Feasibility Score')
            ax.axvline(x=5, color='green', linestyle='--', alpha=0.7,
                      label='Critical (5mm)', linewidth=2)
            ax.axvline(x=10, color='orange', linestyle='--', alpha=0.7,
                      label='Proximity (10mm)', linewidth=2)
            ax.legend()
    ax.set_xlabel('Minimum Vessel Distance (mm)', fontweight='bold')
    ax.set_ylabel('Feasibility Score')
    ax.set_title('Vessel Proximity vs TACE Feasibility', fontweight='bold', fontsize=12)
    ax.grid(alpha=0.3)

    # 6. Tumor location
    ax = axes[1, 2]
    if len(valid_df) > 0:
        location_counts = valid_df['tumor_location'].value_counts()
        colors = ['coral' if loc == 'central' else 'lightgreen' for loc in location_counts.index]
        ax.bar(location_counts.index, location_counts.values, color=colors,
              edgecolor='black', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Tumor Location', fontweight='bold')
    ax.set_ylabel('Number of Patients')
    ax.set_title('Tumor Location Distribution', fontweight='bold', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('TACE Planning Summary - All Patients', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tace_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    if len(valid_df) > 0:
        print(f"Average feasibility score: {valid_df['feasibility_score'].mean():.2f} / 12")
        print(f"Average tumor volume: {valid_df['tumor_volume_ml'].mean():.2f} mL")
        print(f"Average tumor diameter: {valid_df['tumor_diameter_mm'].mean():.2f} mm")
        print(f"Average feeding vessels: {valid_df['num_feeding_vessels'].mean():.2f}")

        dist_df = valid_df[valid_df['min_vessel_distance_mm'] != np.inf]
        if len(dist_df) > 0:
            print(f"Average min vessel distance: {dist_df['min_vessel_distance_mm'].mean():.2f} mm")

        print(f"\nComplexity breakdown:")
        for complexity, count in valid_df['complexity'].value_counts().items():
            print(f"  {complexity}: {count} patients ({count/len(valid_df)*100:.1f}%)")

        print(f"\nTumor location:")
        for location, count in valid_df['tumor_location'].value_counts().items():
            print(f"  {location}: {count} patients ({count/len(valid_df)*100:.1f}%)")

    return summary_df

if __name__ == "__main__":
    # Example usage
    ROOT_DIR = "/media/mirl/DATA/Projects/HCC/data/HCC-TACE-Seg"
    METADATA_CSV = "/media/mirl/DATA/Projects/HCC/data/train.csv"
    OUTPUT_DIR = "/media/mirl/DATA/Projects/HCC/results/tace_planning"

    reports = run_tace_planning(METADATA_CSV, ROOT_DIR, OUTPUT_DIR)