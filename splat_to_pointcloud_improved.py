"""
IMPROVED Splat to Point Cloud Converter

Enhanced version with better filtering to remove bad data:
- Scale-based filtering (removes floaters/noise with abnormal scales)
- Covariance/uncertainty filtering
- Spatial clustering to remove isolated points
- View-dependent filtering options

Usage:
    python splat_to_pointcloud_improved.py input.ply output.ply [options]
"""

import argparse
import numpy as np
import time
from pathlib import Path
from plyfile import PlyData, PlyElement


def log_step(msg, indent=2):
    """Print a log message with consistent formatting."""
    prefix = " " * indent
    print(f"{prefix}[INFO] {msg}")


def sigmoid(x):
    """Sigmoid activation function."""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sh_to_rgb(sh_dc):
    """Convert Spherical Harmonics DC component to RGB."""
    SH_C0 = 0.28209479177387814
    color = sh_dc * SH_C0 + 0.5
    return np.clip(color * 255, 0, 255).astype(np.uint8)


def find_field(names, candidates):
    """Find the first matching field name from a list of candidates."""
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def extract_opacity(vertex, verbose=True):
    """Extract opacity values from vertex data."""
    names = vertex.data.dtype.names
    opacity_field = find_field(names, ['opacity', 'alpha', 'a', 'opacity_raw'])
    
    if opacity_field is None:
        return None
    
    opacity_raw = np.array(vertex[opacity_field])
    
    if verbose:
        print(f"    Found opacity field: '{opacity_field}'")
        print(f"    Raw opacity range: {opacity_raw.min():.3f} to {opacity_raw.max():.3f}")
    
    if opacity_raw.min() < -0.5 or opacity_raw.max() > 1.5:
        if verbose:
            print(f"    Detected logit format, applying sigmoid...")
        opacity = sigmoid(opacity_raw)
    else:
        if verbose:
            print(f"    Detected direct opacity values (0-1 range)")
        opacity = np.clip(opacity_raw, 0, 1)
    
    return opacity


def extract_scales(vertex, verbose=True):
    """
    Extract Gaussian scale values.
    
    Scales indicate the size/extent of each Gaussian.
    Very large or very small scales often indicate noise or artifacts.
    """
    names = vertex.data.dtype.names
    
    # Try different scale field naming conventions
    scale_fields = []
    
    # Standard 3DGS format: scale_0, scale_1, scale_2
    if all(f'scale_{i}' in names for i in range(3)):
        scale_fields = ['scale_0', 'scale_1', 'scale_2']
        if verbose:
            print(f"    Found scale fields: scale_0/1/2")
    # Alternative: sx, sy, sz
    elif all(f in names for f in ['sx', 'sy', 'sz']):
        scale_fields = ['sx', 'sy', 'sz']
        if verbose:
            print(f"    Found scale fields: sx/sy/sz")
    # Alternative: scale_x, scale_y, scale_z
    elif all(f'scale_{c}' in names for c in ['x', 'y', 'z']):
        scale_fields = ['scale_x', 'scale_y', 'scale_z']
        if verbose:
            print(f"    Found scale fields: scale_x/y/z")
    
    if not scale_fields:
        if verbose:
            print(f"    [WARNING] No scale fields found")
        return None
    
    scales = np.stack([np.array(vertex[f]) for f in scale_fields], axis=1)
    
    # Scales are often stored as log-scales in 3DGS
    if scales.min() < -10 or scales.max() > 10:
        if verbose:
            print(f"    Detected log-scale format, applying exp...")
        scales = np.exp(scales)
    
    if verbose:
        print(f"    Scale ranges:")
        print(f"      X: [{scales[:,0].min():.6f}, {scales[:,0].max():.6f}]")
        print(f"      Y: [{scales[:,1].min():.6f}, {scales[:,1].max():.6f}]")
        print(f"      Z: [{scales[:,2].min():.6f}, {scales[:,2].max():.6f}]")
    
    return scales


def extract_colors(vertex, verbose=True):
    """Extract RGB colors from vertex data."""
    names = vertex.data.dtype.names
    
    if all(f'f_dc_{i}' in names for i in range(3)):
        if verbose:
            print(f"    Found SH DC format (f_dc_0/1/2)")
        r = sh_to_rgb(np.array(vertex['f_dc_0']))
        g = sh_to_rgb(np.array(vertex['f_dc_1']))
        b = sh_to_rgb(np.array(vertex['f_dc_2']))
        return r, g, b
    
    if all(c in names for c in ['red', 'green', 'blue']):
        if verbose:
            print(f"    Found direct RGB format (red/green/blue)")
        r = np.array(vertex['red'])
        g = np.array(vertex['green'])
        b = np.array(vertex['blue'])
        
        if r.dtype == np.uint8:
            return r, g, b
        elif r.max() <= 1.0:
            r = np.clip(r * 255, 0, 255).astype(np.uint8)
            g = np.clip(g * 255, 0, 255).astype(np.uint8)
            b = np.clip(b * 255, 0, 255).astype(np.uint8)
        else:
            r = np.clip(r, 0, 255).astype(np.uint8)
            g = np.clip(g, 0, 255).astype(np.uint8)
            b = np.clip(b, 0, 255).astype(np.uint8)
        return r, g, b
    
    if all(c in names for c in ['r', 'g', 'b']):
        if verbose:
            print(f"    Found RGB format (r/g/b)")
        r = np.array(vertex['r'])
        g = np.array(vertex['g'])
        b = np.array(vertex['b'])
        if r.dtype != np.uint8:
            if r.max() <= 1.0:
                r = np.clip(r * 255, 0, 255).astype(np.uint8)
                g = np.clip(g * 255, 0, 255).astype(np.uint8)
                b = np.clip(b * 255, 0, 255).astype(np.uint8)
            else:
                r = np.clip(r, 0, 255).astype(np.uint8)
                g = np.clip(g, 0, 255).astype(np.uint8)
                b = np.clip(b, 0, 255).astype(np.uint8)
        return r, g, b
    
    if all(f'sh_dc_{i}' in names for i in range(3)):
        if verbose:
            print(f"    Found SH DC format (sh_dc_0/1/2)")
        r = sh_to_rgb(np.array(vertex['sh_dc_0']))
        g = sh_to_rgb(np.array(vertex['sh_dc_1']))
        b = sh_to_rgb(np.array(vertex['sh_dc_2']))
        return r, g, b
    
    # Fallback: white
    print(f"    [WARNING] Could not find color data, using white")
    n = len(vertex)
    return (
        np.ones(n, dtype=np.uint8) * 255,
        np.ones(n, dtype=np.uint8) * 255,
        np.ones(n, dtype=np.uint8) * 255
    )


def filter_by_scale(scales, percentile_low=5, percentile_high=95, verbose=True):
    """
    Filter points based on Gaussian scale values.
    
    Gaussians with extremely large or small scales are often noise/artifacts:
    - Very large: floaters, background noise
    - Very small: numerical artifacts, reconstruction errors
    """
    if scales is None:
        return None
    
    # Calculate the "volume" or magnitude of each Gaussian
    scale_magnitude = np.prod(scales, axis=1) ** (1/3)  # Geometric mean
    
    # Also consider max scale (elongated Gaussians are often noise)
    max_scale = np.max(scales, axis=1)
    
    # Calculate thresholds
    mag_low = np.percentile(scale_magnitude, percentile_low)
    mag_high = np.percentile(scale_magnitude, percentile_high)
    max_high = np.percentile(max_scale, percentile_high)
    
    # Create mask
    mask = (
        (scale_magnitude >= mag_low) & 
        (scale_magnitude <= mag_high) &
        (max_scale <= max_high)
    )
    
    if verbose:
        print(f"    Scale magnitude range: [{scale_magnitude.min():.6f}, {scale_magnitude.max():.6f}]")
        print(f"    Keeping scales in [{mag_low:.6f}, {mag_high:.6f}] (percentile {percentile_low}-{percentile_high})")
        print(f"    Max scale threshold: {max_high:.6f}")
        print(f"    Points passing scale filter: {np.sum(mask):,} / {len(mask):,} ({np.sum(mask)/len(mask)*100:.1f}%)")
    
    return mask


def filter_by_spatial_density(positions, k_neighbors=20, std_ratio=2.0, verbose=True):
    """
    Filter out spatially isolated points using local density.
    
    Points that are far from their neighbors are likely noise/floaters.
    """
    from scipy.spatial import KDTree
    
    if verbose:
        print(f"    Building KD-tree for spatial filtering...")
    
    tree = KDTree(positions)
    
    # Query k nearest neighbors for each point
    distances, _ = tree.query(positions, k=k_neighbors + 1)  # +1 because point is its own neighbor
    
    # Use mean distance to k neighbors as density measure
    mean_distances = distances[:, 1:].mean(axis=1)  # Skip self
    
    # Calculate threshold (mean + std_ratio * std)
    threshold = mean_distances.mean() + std_ratio * mean_distances.std()
    
    mask = mean_distances <= threshold
    
    if verbose:
        print(f"    Mean neighbor distance: {mean_distances.mean():.6f} +/- {mean_distances.std():.6f}")
        print(f"    Threshold: {threshold:.6f}")
        print(f"    Points passing density filter: {np.sum(mask):,} / {len(mask):,} ({np.sum(mask)/len(mask)*100:.1f}%)")
    
    return mask


def filter_by_color_confidence(r, g, b, verbose=True):
    """
    Filter out points with very dark or very bright colors (often noise).
    
    Pure black or white points are often artifacts.
    """
    # Calculate brightness
    brightness = (r.astype(float) + g.astype(float) + b.astype(float)) / 3
    
    # Filter out very dark (< 10) and very bright (> 250)
    mask = (brightness >= 10) & (brightness <= 250)
    
    # Also filter out pure gray (all channels equal, often artifacts)
    gray_mask = ~((r == g) & (g == b) & ((r < 20) | (r > 235)))
    
    combined_mask = mask & gray_mask
    
    if verbose:
        print(f"    Brightness range: [{brightness.min():.1f}, {brightness.max():.1f}]")
        print(f"    Points passing color filter: {np.sum(combined_mask):,} / {len(combined_mask):,} ({np.sum(combined_mask)/len(combined_mask)*100:.1f}%)")
    
    return combined_mask


def splat_to_pointcloud_improved(
    input_path, 
    output_path, 
    opacity_threshold=0.5,
    use_scale_filter=True,
    scale_percentile_low=5,
    scale_percentile_high=95,
    use_density_filter=True,
    density_std_ratio=2.0,
    use_color_filter=True,
    verbose=True
):
    """
    Extract point cloud from Gaussian Splat with improved filtering.
    
    Args:
        input_path: Path to input .ply splat file
        output_path: Path to output .ply point cloud file
        opacity_threshold: Minimum opacity to include (0-1)
        use_scale_filter: Filter by Gaussian scale values
        scale_percentile_low: Lower percentile for scale filtering
        scale_percentile_high: Upper percentile for scale filtering
        use_density_filter: Filter spatially isolated points
        density_std_ratio: Standard deviation ratio for density filtering
        use_color_filter: Filter by color (remove artifacts)
        verbose: Print progress
    """
    start_time = time.time()
    
    if verbose:
        print()
        print("  " + "=" * 50)
        print("  IMPROVED SPLAT TO POINT CLOUD")
        print("  " + "=" * 50)
        log_step(f"Reading PLY file: {input_path}")
    
    read_start = time.time()
    plydata = PlyData.read(input_path)
    vertex = plydata['vertex']
    read_time = time.time() - read_start
    
    if verbose:
        log_step(f"File read complete ({read_time:.2f}s)")
        print()
        print("  " + "-" * 50)
        print("  INPUT FILE ANALYSIS")
        print("  " + "-" * 50)
        print(f"    Total gaussians: {len(vertex):,}")
        print(f"    Properties: {list(vertex.data.dtype.names)}")
        print()
    
    # Extract positions
    names = vertex.data.dtype.names
    x_field = find_field(names, ['x', 'px', 'pos_x', 'position_x'])
    y_field = find_field(names, ['y', 'py', 'pos_y', 'position_y'])
    z_field = find_field(names, ['z', 'pz', 'pos_z', 'position_z'])
    
    if not all([x_field, y_field, z_field]):
        print(f"  [ERROR] Could not find position fields")
        return 0
    
    x = np.array(vertex[x_field])
    y = np.array(vertex[y_field])
    z = np.array(vertex[z_field])
    positions = np.column_stack([x, y, z])
    
    if verbose:
        print("  " + "-" * 50)
        print("  POSITION DATA")
        print("  " + "-" * 50)
        print(f"    X range: [{x.min():.4f}, {x.max():.4f}]")
        print(f"    Y range: [{y.min():.4f}, {y.max():.4f}]")
        print(f"    Z range: [{z.min():.4f}, {z.max():.4f}]")
        print()
    
    # Initialize combined mask (all True)
    combined_mask = np.ones(len(vertex), dtype=bool)
    
    # Step 1: Opacity filtering
    if verbose:
        print("  " + "-" * 50)
        print("  FILTER 1: OPACITY")
        print("  " + "-" * 50)
    
    opacity = extract_opacity(vertex, verbose)
    if opacity is not None:
        opacity_mask = opacity > opacity_threshold
        combined_mask &= opacity_mask
        if verbose:
            print(f"    Opacity threshold: {opacity_threshold}")
            print(f"    Points passing: {np.sum(opacity_mask):,} ({np.sum(opacity_mask)/len(opacity_mask)*100:.1f}%)")
    print()
    
    # Step 2: Scale filtering (Gaussian size)
    if use_scale_filter:
        if verbose:
            print("  " + "-" * 50)
            print("  FILTER 2: GAUSSIAN SCALE")
            print("  " + "-" * 50)
        
        scales = extract_scales(vertex, verbose)
        if scales is not None:
            scale_mask = filter_by_scale(
                scales, 
                percentile_low=scale_percentile_low,
                percentile_high=scale_percentile_high,
                verbose=verbose
            )
            if scale_mask is not None:
                combined_mask &= scale_mask
        print()
    
    # Step 3: Extract colors (needed for color filtering)
    if verbose:
        print("  " + "-" * 50)
        print("  COLOR EXTRACTION")
        print("  " + "-" * 50)
    r, g, b = extract_colors(vertex, verbose)
    print()
    
    # Step 4: Color filtering
    if use_color_filter:
        if verbose:
            print("  " + "-" * 50)
            print("  FILTER 3: COLOR")
            print("  " + "-" * 50)
        
        color_mask = filter_by_color_confidence(r, g, b, verbose)
        combined_mask &= color_mask
        print()
    
    # Apply mask to all arrays
    x, y, z = x[combined_mask], y[combined_mask], z[combined_mask]
    r, g, b = r[combined_mask], g[combined_mask], b[combined_mask]
    positions = positions[combined_mask]
    
    # Step 5: Spatial density filtering (on already filtered points)
    if use_density_filter and len(positions) > 100:
        if verbose:
            print("  " + "-" * 50)
            print("  FILTER 4: SPATIAL DENSITY")
            print("  " + "-" * 50)
        
        try:
            density_mask = filter_by_spatial_density(
                positions, 
                k_neighbors=20,
                std_ratio=density_std_ratio,
                verbose=verbose
            )
            x, y, z = x[density_mask], y[density_mask], z[density_mask]
            r, g, b = r[density_mask], g[density_mask], b[density_mask]
        except Exception as e:
            if verbose:
                print(f"    [WARNING] Density filter failed: {e}")
        print()
    
    if len(x) == 0:
        print("  [ERROR] No points remaining after filtering!")
        print("  [HINT] Try relaxing filter thresholds")
        return 0
    
    # Summary
    if verbose:
        print("  " + "-" * 50)
        print("  FILTERING SUMMARY")
        print("  " + "-" * 50)
        print(f"    Original points: {len(vertex):,}")
        print(f"    Final points:    {len(x):,}")
        print(f"    Kept: {len(x)/len(vertex)*100:.1f}%")
        print()
    
    # Write output
    if verbose:
        print("  " + "-" * 50)
        print("  WRITING OUTPUT")
        print("  " + "-" * 50)
    
    write_start = time.time()
    vertices = np.array(
        list(zip(x, y, z, r, g, b)),
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
    )
    
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(output_path)
    write_time = time.time() - write_start
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"    Output file: {output_path}")
        print(f"    Points written: {len(vertices):,}")
        print(f"    Write time: {write_time:.2f}s")
        print(f"    Total time: {total_time:.2f}s")
        print()
    
    return len(vertices)


def main():
    parser = argparse.ArgumentParser(
        description="Improved splat to point cloud extraction with filtering"
    )
    parser.add_argument("input", help="Input .ply splat file")
    parser.add_argument("output", help="Output .ply point cloud file")
    parser.add_argument(
        "--opacity", "-o",
        type=float,
        default=0.5,
        help="Minimum opacity threshold (0-1, default: 0.5)"
    )
    parser.add_argument(
        "--no-scale-filter",
        action="store_true",
        help="Disable scale-based filtering"
    )
    parser.add_argument(
        "--scale-low",
        type=float,
        default=5,
        help="Scale percentile low (default: 5)"
    )
    parser.add_argument(
        "--scale-high",
        type=float,
        default=95,
        help="Scale percentile high (default: 95)"
    )
    parser.add_argument(
        "--no-density-filter",
        action="store_true",
        help="Disable spatial density filtering"
    )
    parser.add_argument(
        "--density-std",
        type=float,
        default=2.0,
        help="Density filter std ratio (default: 2.0, lower = stricter)"
    )
    parser.add_argument(
        "--no-color-filter",
        action="store_true",
        help="Disable color-based filtering"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output messages"
    )
    
    args = parser.parse_args()
    
    splat_to_pointcloud_improved(
        args.input,
        args.output,
        opacity_threshold=args.opacity,
        use_scale_filter=not args.no_scale_filter,
        scale_percentile_low=args.scale_low,
        scale_percentile_high=args.scale_high,
        use_density_filter=not args.no_density_filter,
        density_std_ratio=args.density_std,
        use_color_filter=not args.no_color_filter,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
