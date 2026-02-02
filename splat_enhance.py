"""
Gaussian Splat Enhancement Tool

Improves Gaussian splat quality by:
1. Detecting and removing floaters/outliers
2. Identifying sparse regions (potential holes)
3. Densifying sparse areas by cloning/interpolating Gaussians

Usage:
    python splat_enhance.py input.ply output.ply [options]
"""

import argparse
import numpy as np
import time
from plyfile import PlyData, PlyElement
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter, binary_dilation


def log_step(msg, indent=2):
    """Print a log message with consistent formatting."""
    prefix = " " * indent
    print(f"{prefix}[INFO] {msg}")


def sigmoid(x):
    """Sigmoid activation function."""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def inverse_sigmoid(x):
    """Inverse sigmoid (logit) function."""
    x = np.clip(x, 1e-7, 1 - 1e-7)
    return np.log(x / (1 - x))


class GaussianSplat:
    """
    Represents a 3D Gaussian Splat with all its properties.
    Handles different PLY formats from various 3DGS implementations.
    """
    
    def __init__(self, ply_path=None):
        self.positions = None  # Nx3 float32
        self.colors_dc = None  # Nx3 (SH DC component)
        self.colors_rest = None  # Nx45 (remaining SH coefficients, optional)
        self.opacity = None  # Nx1 (logit space)
        self.scales = None  # Nx3 (log space)
        self.rotations = None  # Nx4 (quaternion)
        self.extra_fields = {}  # Any additional fields
        
        self._ply_format = None  # Track the format for re-export
        
        if ply_path:
            self.load(ply_path)
    
    def load(self, ply_path):
        """Load Gaussian splat from PLY file."""
        print(f"  Loading: {ply_path}")
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        names = vertex.data.dtype.names
        
        n = len(vertex)
        print(f"  Found {n:,} Gaussians")
        print(f"  Properties: {list(names)}")
        
        # Extract positions
        self.positions = np.column_stack([
            np.array(vertex['x']),
            np.array(vertex['y']),
            np.array(vertex['z'])
        ]).astype(np.float32)
        
        # Extract opacity (may be in logit or direct form)
        if 'opacity' in names:
            opacity_raw = np.array(vertex['opacity'])
            # Store in logit form internally
            if opacity_raw.min() < -0.5 or opacity_raw.max() > 1.5:
                self.opacity = opacity_raw.astype(np.float32)
                self._opacity_is_logit = True
            else:
                self.opacity = inverse_sigmoid(opacity_raw).astype(np.float32)
                self._opacity_is_logit = False
        
        # Extract scales (may be in log or linear form)
        if all(f'scale_{i}' in names for i in range(3)):
            scales = np.column_stack([
                np.array(vertex['scale_0']),
                np.array(vertex['scale_1']),
                np.array(vertex['scale_2'])
            ])
            # Detect if in log space
            if scales.min() < 0 or scales.max() < 1:
                self.scales = scales.astype(np.float32)
                self._scales_is_log = True
            else:
                self.scales = np.log(np.clip(scales, 1e-7, None)).astype(np.float32)
                self._scales_is_log = False
        elif all(f'scaling_{i}' in names for i in range(3)):
            scales = np.column_stack([
                np.array(vertex['scaling_0']),
                np.array(vertex['scaling_1']),
                np.array(vertex['scaling_2'])
            ])
            if scales.min() < 0 or scales.max() < 1:
                self.scales = scales.astype(np.float32)
                self._scales_is_log = True
            else:
                self.scales = np.log(np.clip(scales, 1e-7, None)).astype(np.float32)
                self._scales_is_log = False
        
        # Extract rotations (quaternion)
        if all(f'rot_{i}' in names for i in range(4)):
            self.rotations = np.column_stack([
                np.array(vertex['rot_0']),
                np.array(vertex['rot_1']),
                np.array(vertex['rot_2']),
                np.array(vertex['rot_3'])
            ]).astype(np.float32)
        
        # Extract SH DC (base color)
        if all(f'f_dc_{i}' in names for i in range(3)):
            self.colors_dc = np.column_stack([
                np.array(vertex['f_dc_0']),
                np.array(vertex['f_dc_1']),
                np.array(vertex['f_dc_2'])
            ]).astype(np.float32)
            self._ply_format = 'standard_3dgs'
        elif all(c in names for c in ['red', 'green', 'blue']):
            # Convert RGB to SH DC
            r = np.array(vertex['red'])
            g = np.array(vertex['green'])
            b = np.array(vertex['blue'])
            if r.dtype == np.uint8:
                r, g, b = r / 255.0, g / 255.0, b / 255.0
            elif r.max() > 1.0:
                r, g, b = r / 255.0, g / 255.0, b / 255.0
            # Convert to SH DC: SH_DC = (color - 0.5) / SH_C0
            SH_C0 = 0.28209479177387814
            self.colors_dc = np.column_stack([
                (r - 0.5) / SH_C0,
                (g - 0.5) / SH_C0,
                (b - 0.5) / SH_C0
            ]).astype(np.float32)
            self._ply_format = 'rgb_direct'
        
        # Extract remaining SH coefficients (for view-dependent colors)
        sh_rest_fields = [f'f_rest_{i}' for i in range(45)]
        if all(f in names for f in sh_rest_fields):
            self.colors_rest = np.column_stack([
                np.array(vertex[f]) for f in sh_rest_fields
            ]).astype(np.float32)
        
        print(f"  Loaded successfully. Format: {self._ply_format}")
        return self
    
    def save(self, ply_path):
        """Save Gaussian splat to PLY file."""
        print(f"  Saving to: {ply_path}")
        
        n = len(self.positions)
        
        # Build dtype and data based on what we have
        dtype_list = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ]
        data_dict = {
            'x': self.positions[:, 0],
            'y': self.positions[:, 1],
            'z': self.positions[:, 2],
        }
        
        # Add opacity
        if self.opacity is not None:
            dtype_list.append(('opacity', 'f4'))
            data_dict['opacity'] = self.opacity
        
        # Add scales
        if self.scales is not None:
            for i in range(3):
                dtype_list.append((f'scale_{i}', 'f4'))
                data_dict[f'scale_{i}'] = self.scales[:, i]
        
        # Add rotations
        if self.rotations is not None:
            for i in range(4):
                dtype_list.append((f'rot_{i}', 'f4'))
                data_dict[f'rot_{i}'] = self.rotations[:, i]
        
        # Add SH DC
        if self.colors_dc is not None:
            for i in range(3):
                dtype_list.append((f'f_dc_{i}', 'f4'))
                data_dict[f'f_dc_{i}'] = self.colors_dc[:, i]
        
        # Add SH rest
        if self.colors_rest is not None:
            for i in range(45):
                dtype_list.append((f'f_rest_{i}', 'f4'))
                data_dict[f'f_rest_{i}'] = self.colors_rest[:, i]
        
        # Create structured array
        vertices = np.zeros(n, dtype=dtype_list)
        for key, value in data_dict.items():
            vertices[key] = value
        
        el = PlyElement.describe(vertices, 'vertex')
        PlyData([el]).write(ply_path)
        print(f"  Saved {n:,} Gaussians")
    
    def get_opacity_linear(self):
        """Get opacity values in 0-1 range."""
        return sigmoid(self.opacity)
    
    def get_scales_linear(self):
        """Get scale values in linear space."""
        return np.exp(self.scales)
    
    def __len__(self):
        return len(self.positions)


def remove_floaters(splat, std_threshold=3.0, min_opacity=0.05, verbose=True):
    """
    Remove floater Gaussians that are outliers in position or have very low opacity.
    
    Args:
        splat: GaussianSplat object
        std_threshold: Remove points beyond this many standard deviations from centroid
        min_opacity: Remove Gaussians with opacity below this threshold
        verbose: Print progress
    
    Returns:
        Modified GaussianSplat object, number of removed Gaussians
    """
    if verbose:
        print("\n  " + "-" * 50)
        print("  FLOATER REMOVAL")
        print("  " + "-" * 50)
    
    n_original = len(splat)
    mask = np.ones(n_original, dtype=bool)
    
    # Remove by position outliers
    centroid = np.mean(splat.positions, axis=0)
    distances = np.linalg.norm(splat.positions - centroid, axis=1)
    dist_mean = np.mean(distances)
    dist_std = np.std(distances)
    
    position_outliers = distances > (dist_mean + std_threshold * dist_std)
    mask &= ~position_outliers
    
    if verbose:
        print(f"    Position outliers (>{std_threshold} std): {np.sum(position_outliers):,}")
    
    # Remove by low opacity
    if splat.opacity is not None:
        opacity_linear = splat.get_opacity_linear()
        low_opacity = opacity_linear < min_opacity
        mask &= ~low_opacity
        
        if verbose:
            print(f"    Low opacity (<{min_opacity}): {np.sum(low_opacity):,}")
    
    # Remove by extreme scale (very large Gaussians are often artifacts)
    if splat.scales is not None:
        scales_linear = splat.get_scales_linear()
        avg_scale = np.mean(scales_linear, axis=1)
        scale_99th = np.percentile(avg_scale, 99)
        large_scale = avg_scale > scale_99th * 3
        mask &= ~large_scale
        
        if verbose:
            print(f"    Extreme scale (3x 99th percentile): {np.sum(large_scale):,}")
    
    # Apply mask
    n_removed = n_original - np.sum(mask)
    
    splat.positions = splat.positions[mask]
    if splat.opacity is not None:
        splat.opacity = splat.opacity[mask]
    if splat.scales is not None:
        splat.scales = splat.scales[mask]
    if splat.rotations is not None:
        splat.rotations = splat.rotations[mask]
    if splat.colors_dc is not None:
        splat.colors_dc = splat.colors_dc[mask]
    if splat.colors_rest is not None:
        splat.colors_rest = splat.colors_rest[mask]
    
    if verbose:
        print(f"    Total removed: {n_removed:,}")
        print(f"    Remaining: {len(splat):,}")
    
    return splat, n_removed


def find_sparse_regions(splat, grid_resolution=50, density_threshold_percentile=10, verbose=True):
    """
    Find regions where Gaussian density is low (potential holes).
    
    Uses a 3D grid to compute local density and identifies cells with
    significantly lower density than average.
    
    Args:
        splat: GaussianSplat object
        grid_resolution: Number of cells per axis
        density_threshold_percentile: Percentile below which regions are "sparse"
        verbose: Print progress
    
    Returns:
        sparse_cell_centers: Nx3 array of sparse region centers
        sparse_cell_indices: Indices into the grid
    """
    if verbose:
        print("\n  " + "-" * 50)
        print("  SPARSE REGION DETECTION")
        print("  " + "-" * 50)
    
    positions = splat.positions
    
    # Compute bounding box
    bbox_min = np.min(positions, axis=0)
    bbox_max = np.max(positions, axis=0)
    bbox_size = bbox_max - bbox_min
    
    if verbose:
        print(f"    Bounding box: {bbox_size}")
        print(f"    Grid resolution: {grid_resolution}^3")
    
    # Create density grid
    cell_size = bbox_size / grid_resolution
    
    # Compute cell indices for each Gaussian
    cell_indices = ((positions - bbox_min) / cell_size).astype(int)
    cell_indices = np.clip(cell_indices, 0, grid_resolution - 1)
    
    # Count Gaussians per cell
    density_grid = np.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=np.int32)
    for idx in cell_indices:
        density_grid[idx[0], idx[1], idx[2]] += 1
    
    # Apply Gaussian smoothing to get continuous density field
    density_smoothed = gaussian_filter(density_grid.astype(float), sigma=1.5)
    
    # Find cells that are occupied but sparse
    occupied_mask = density_grid > 0
    
    # Also include neighbors of occupied cells (edge regions)
    structure = np.ones((3, 3, 3))
    near_occupied = binary_dilation(occupied_mask, structure, iterations=2)
    
    # Cells that are near occupied regions but have low density
    density_values = density_smoothed[near_occupied]
    if len(density_values) > 0:
        threshold = np.percentile(density_values[density_values > 0], density_threshold_percentile)
    else:
        threshold = 0
    
    # Sparse cells: near surface but low density
    sparse_mask = near_occupied & (density_smoothed < threshold) & (density_smoothed > 0)
    
    # Get centers of sparse cells
    sparse_indices = np.argwhere(sparse_mask)
    sparse_cell_centers = bbox_min + (sparse_indices + 0.5) * cell_size
    
    if verbose:
        print(f"    Non-empty cells: {np.sum(occupied_mask):,}")
        print(f"    Density threshold: {threshold:.2f}")
        print(f"    Sparse cells found: {len(sparse_cell_centers):,}")
    
    return sparse_cell_centers, sparse_indices


def densify_sparse_regions(splat, sparse_centers, k_neighbors=5, jitter=0.02, verbose=True):
    """
    Add new Gaussians in sparse regions by interpolating from nearby existing Gaussians.
    
    Args:
        splat: GaussianSplat object
        sparse_centers: Nx3 array of sparse region centers to fill
        k_neighbors: Number of neighbors to interpolate from
        jitter: Random offset factor (relative to local scale)
        verbose: Print progress
    
    Returns:
        Modified GaussianSplat object with additional Gaussians
    """
    if verbose:
        print("\n  " + "-" * 50)
        print("  SPARSE REGION DENSIFICATION")
        print("  " + "-" * 50)
    
    if len(sparse_centers) == 0:
        if verbose:
            print("    No sparse regions to fill")
        return splat, 0
    
    n_original = len(splat)
    
    # Build KD-tree for existing Gaussians
    tree = KDTree(splat.positions)
    
    # For each sparse center, find neighbors and interpolate
    new_positions = []
    new_opacities = []
    new_scales = []
    new_rotations = []
    new_colors_dc = []
    new_colors_rest = []
    
    for center in sparse_centers:
        # Find k nearest neighbors
        distances, indices = tree.query(center, k=k_neighbors)
        
        # Weight by inverse distance
        weights = 1.0 / (distances + 1e-6)
        weights /= np.sum(weights)
        
        # Interpolate position (with jitter)
        new_pos = np.average(splat.positions[indices], axis=0, weights=weights)
        if jitter > 0:
            local_scale = np.mean(distances)
            new_pos += np.random.randn(3) * local_scale * jitter
        new_positions.append(new_pos)
        
        # Interpolate other properties
        if splat.opacity is not None:
            new_opacities.append(np.average(splat.opacity[indices], weights=weights))
        
        if splat.scales is not None:
            new_scales.append(np.average(splat.scales[indices], axis=0, weights=weights))
        
        if splat.rotations is not None:
            # Simple weighted average (not proper quaternion interpolation, but okay for similar orientations)
            avg_rot = np.average(splat.rotations[indices], axis=0, weights=weights)
            avg_rot /= np.linalg.norm(avg_rot)  # Normalize
            new_rotations.append(avg_rot)
        
        if splat.colors_dc is not None:
            new_colors_dc.append(np.average(splat.colors_dc[indices], axis=0, weights=weights))
        
        if splat.colors_rest is not None:
            new_colors_rest.append(np.average(splat.colors_rest[indices], axis=0, weights=weights))
    
    # Append new Gaussians to splat
    n_new = len(new_positions)
    
    splat.positions = np.vstack([splat.positions, np.array(new_positions)])
    
    if splat.opacity is not None and new_opacities:
        splat.opacity = np.concatenate([splat.opacity, np.array(new_opacities)])
    
    if splat.scales is not None and new_scales:
        splat.scales = np.vstack([splat.scales, np.array(new_scales)])
    
    if splat.rotations is not None and new_rotations:
        splat.rotations = np.vstack([splat.rotations, np.array(new_rotations)])
    
    if splat.colors_dc is not None and new_colors_dc:
        splat.colors_dc = np.vstack([splat.colors_dc, np.array(new_colors_dc)])
    
    if splat.colors_rest is not None and new_colors_rest:
        splat.colors_rest = np.vstack([splat.colors_rest, np.array(new_colors_rest)])
    
    if verbose:
        print(f"    Original Gaussians: {n_original:,}")
        print(f"    New Gaussians added: {n_new:,}")
        print(f"    Total Gaussians: {len(splat):,}")
    
    return splat, n_new


def clone_and_split(splat, split_threshold_percentile=95, max_new=10000, verbose=True):
    """
    Clone and split large Gaussians into smaller ones for better coverage.
    
    This mimics the densification strategy from the original 3DGS paper,
    but applied post-hoc based on scale.
    
    Args:
        splat: GaussianSplat object
        split_threshold_percentile: Split Gaussians above this percentile in scale
        max_new: Maximum number of new Gaussians to add
        verbose: Print progress
    
    Returns:
        Modified GaussianSplat object
    """
    if verbose:
        print("\n  " + "-" * 50)
        print("  CLONE AND SPLIT LARGE GAUSSIANS")
        print("  " + "-" * 50)
    
    if splat.scales is None:
        if verbose:
            print("    No scale data - skipping")
        return splat, 0
    
    n_original = len(splat)
    
    # Find large Gaussians
    scales_linear = splat.get_scales_linear()
    max_scale = np.max(scales_linear, axis=1)
    threshold = np.percentile(max_scale, split_threshold_percentile)
    
    large_mask = max_scale > threshold
    large_indices = np.where(large_mask)[0]
    
    if verbose:
        print(f"    Scale threshold (p{split_threshold_percentile}): {threshold:.6f}")
        print(f"    Gaussians to split: {len(large_indices):,}")
    
    if len(large_indices) == 0:
        return splat, 0
    
    # Limit number of splits
    if len(large_indices) > max_new // 2:
        large_indices = np.random.choice(large_indices, max_new // 2, replace=False)
    
    # For each large Gaussian, create two children offset along the largest scale axis
    new_positions = []
    new_opacities = []
    new_scales = []
    new_rotations = []
    new_colors_dc = []
    new_colors_rest = []
    
    for idx in large_indices:
        pos = splat.positions[idx]
        scale = scales_linear[idx]
        
        # Find largest scale axis
        largest_axis = np.argmax(scale)
        offset_dir = np.zeros(3)
        offset_dir[largest_axis] = 1.0
        
        # If we have rotation, transform the offset
        if splat.rotations is not None:
            # Simplified: assume rotation is close to identity for now
            # Full implementation would apply quaternion rotation
            pass
        
        offset = offset_dir * scale[largest_axis] * 0.5
        
        # Create two children
        for sign in [-1, 1]:
            new_positions.append(pos + sign * offset)
            
            if splat.opacity is not None:
                new_opacities.append(splat.opacity[idx])
            
            if splat.scales is not None:
                # Reduce scale
                new_scale = splat.scales[idx].copy()
                new_scale[largest_axis] -= 0.5  # Halve in log space
                new_scales.append(new_scale)
            
            if splat.rotations is not None:
                new_rotations.append(splat.rotations[idx].copy())
            
            if splat.colors_dc is not None:
                new_colors_dc.append(splat.colors_dc[idx].copy())
            
            if splat.colors_rest is not None:
                new_colors_rest.append(splat.colors_rest[idx].copy())
    
    # Append new Gaussians
    n_new = len(new_positions)
    
    splat.positions = np.vstack([splat.positions, np.array(new_positions)])
    
    if splat.opacity is not None and new_opacities:
        splat.opacity = np.concatenate([splat.opacity, np.array(new_opacities)])
    
    if splat.scales is not None and new_scales:
        splat.scales = np.vstack([splat.scales, np.array(new_scales)])
    
    if splat.rotations is not None and new_rotations:
        splat.rotations = np.vstack([splat.rotations, np.array(new_rotations)])
    
    if splat.colors_dc is not None and new_colors_dc:
        splat.colors_dc = np.vstack([splat.colors_dc, np.array(new_colors_dc)])
    
    if splat.colors_rest is not None and new_colors_rest:
        splat.colors_rest = np.vstack([splat.colors_rest, np.array(new_colors_rest)])
    
    if verbose:
        print(f"    New Gaussians from splitting: {n_new:,}")
        print(f"    Total Gaussians: {len(splat):,}")
    
    return splat, n_new


def enhance_splat(input_path, output_path, 
                  remove_floaters_enabled=True,
                  densify_sparse_enabled=True,
                  split_large_enabled=True,
                  grid_resolution=50,
                  density_threshold_percentile=10,
                  verbose=True):
    """
    Run full enhancement pipeline on a Gaussian splat.
    
    Args:
        input_path: Path to input PLY file
        output_path: Path to output PLY file
        remove_floaters_enabled: Remove outlier Gaussians
        densify_sparse_enabled: Add Gaussians in sparse regions
        split_large_enabled: Split large Gaussians
        grid_resolution: Resolution for sparse detection grid
        density_threshold_percentile: What percentile is "sparse"
        verbose: Print progress
    
    Returns:
        Statistics dict
    """
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print("GAUSSIAN SPLAT ENHANCEMENT")
    print("=" * 60)
    
    # Load splat
    splat = GaussianSplat(input_path)
    n_original = len(splat)
    
    stats = {
        'original_count': n_original,
        'floaters_removed': 0,
        'sparse_added': 0,
        'split_added': 0,
    }
    
    # Step 1: Remove floaters
    if remove_floaters_enabled:
        splat, n_removed = remove_floaters(splat, verbose=verbose)
        stats['floaters_removed'] = n_removed
    
    # Step 2: Find and fill sparse regions
    if densify_sparse_enabled:
        sparse_centers, _ = find_sparse_regions(
            splat, 
            grid_resolution=grid_resolution,
            density_threshold_percentile=density_threshold_percentile,
            verbose=verbose
        )
        splat, n_added = densify_sparse_regions(splat, sparse_centers, verbose=verbose)
        stats['sparse_added'] = n_added
    
    # Step 3: Split large Gaussians
    if split_large_enabled:
        splat, n_split = clone_and_split(splat, verbose=verbose)
        stats['split_added'] = n_split
    
    # Save result
    print("\n  " + "-" * 50)
    print("  SAVING RESULT")
    print("  " + "-" * 50)
    splat.save(output_path)
    
    stats['final_count'] = len(splat)
    stats['elapsed_time'] = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("ENHANCEMENT SUMMARY")
    print("=" * 60)
    print(f"  Original Gaussians:    {stats['original_count']:,}")
    print(f"  Floaters removed:      {stats['floaters_removed']:,}")
    print(f"  Sparse region fills:   {stats['sparse_added']:,}")
    print(f"  Large Gaussian splits: {stats['split_added']:,}")
    print(f"  Final Gaussians:       {stats['final_count']:,}")
    print(f"  Elapsed time:          {stats['elapsed_time']:.2f}s")
    print("=" * 60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Enhance Gaussian Splat by removing artifacts and filling holes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic enhancement
    python splat_enhance.py input.ply output.ply
    
    # Only remove floaters
    python splat_enhance.py input.ply output.ply --no-densify --no-split
    
    # Aggressive hole filling
    python splat_enhance.py input.ply output.ply --grid 30 --density-threshold 20
    
    # Fine grid for detailed models
    python splat_enhance.py input.ply output.ply --grid 100
        """
    )
    
    parser.add_argument("input", help="Input .ply Gaussian Splat file")
    parser.add_argument("output", help="Output .ply file")
    
    parser.add_argument("--no-floaters", action="store_true",
                        help="Skip floater removal")
    parser.add_argument("--no-densify", action="store_true",
                        help="Skip sparse region densification")
    parser.add_argument("--no-split", action="store_true",
                        help="Skip large Gaussian splitting")
    
    parser.add_argument("--grid", type=int, default=50,
                        help="Grid resolution for sparse detection (default: 50)")
    parser.add_argument("--density-threshold", type=int, default=10,
                        help="Percentile for sparse threshold (default: 10)")
    
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress output")
    
    args = parser.parse_args()
    
    enhance_splat(
        args.input,
        args.output,
        remove_floaters_enabled=not args.no_floaters,
        densify_sparse_enabled=not args.no_densify,
        split_large_enabled=not args.no_split,
        grid_resolution=args.grid,
        density_threshold_percentile=args.density_threshold,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
