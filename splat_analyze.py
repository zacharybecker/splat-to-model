"""
Gaussian Splat Analyzer

Analyzes a Gaussian Splat to identify potential quality issues:
1. Sparse regions (potential holes)
2. Floaters and outliers
3. Color/opacity distribution
4. Scale analysis
5. Multi-view coverage estimation

Usage:
    python splat_analyze.py input.ply [--output report.json]
"""

import argparse
import json
import numpy as np
from plyfile import PlyData
from scipy.spatial import KDTree, ConvexHull
from scipy.ndimage import gaussian_filter, binary_dilation
import sys


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def load_splat_data(ply_path):
    """Load essential data from a Gaussian splat PLY file."""
    print(f"Loading: {ply_path}")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    names = vertex.data.dtype.names
    
    data = {
        'n_gaussians': len(vertex),
        'properties': list(names),
    }
    
    # Positions
    data['positions'] = np.column_stack([
        np.array(vertex['x']),
        np.array(vertex['y']),
        np.array(vertex['z'])
    ])
    
    # Opacity
    if 'opacity' in names:
        opacity_raw = np.array(vertex['opacity'])
        if opacity_raw.min() < -0.5 or opacity_raw.max() > 1.5:
            data['opacity'] = sigmoid(opacity_raw)
        else:
            data['opacity'] = np.clip(opacity_raw, 0, 1)
    
    # Scales
    scale_fields = None
    if all(f'scale_{i}' in names for i in range(3)):
        scale_fields = ['scale_0', 'scale_1', 'scale_2']
    elif all(f'scaling_{i}' in names for i in range(3)):
        scale_fields = ['scaling_0', 'scaling_1', 'scaling_2']
    
    if scale_fields:
        scales = np.column_stack([np.array(vertex[f]) for f in scale_fields])
        if scales.min() < 0 or scales.max() < 1:
            scales = np.exp(scales)
        data['scales'] = scales
    
    # Colors (SH DC)
    if all(f'f_dc_{i}' in names for i in range(3)):
        sh_dc = np.column_stack([
            np.array(vertex['f_dc_0']),
            np.array(vertex['f_dc_1']),
            np.array(vertex['f_dc_2'])
        ])
        SH_C0 = 0.28209479177387814
        data['colors'] = np.clip(sh_dc * SH_C0 + 0.5, 0, 1)
    elif all(c in names for c in ['red', 'green', 'blue']):
        r = np.array(vertex['red'])
        g = np.array(vertex['green'])
        b = np.array(vertex['blue'])
        if r.dtype == np.uint8:
            data['colors'] = np.column_stack([r, g, b]) / 255.0
        elif r.max() > 1.0:
            data['colors'] = np.column_stack([r, g, b]) / 255.0
        else:
            data['colors'] = np.column_stack([r, g, b])
    
    return data


def analyze_geometry(data):
    """Analyze spatial distribution of Gaussians."""
    print("\n" + "=" * 60)
    print("GEOMETRY ANALYSIS")
    print("=" * 60)
    
    positions = data['positions']
    n = len(positions)
    
    # Bounding box
    bbox_min = np.min(positions, axis=0)
    bbox_max = np.max(positions, axis=0)
    bbox_size = bbox_max - bbox_min
    bbox_volume = np.prod(bbox_size)
    
    print(f"  Bounding box min:    [{bbox_min[0]:.4f}, {bbox_min[1]:.4f}, {bbox_min[2]:.4f}]")
    print(f"  Bounding box max:    [{bbox_max[0]:.4f}, {bbox_max[1]:.4f}, {bbox_max[2]:.4f}]")
    print(f"  Bounding box size:   [{bbox_size[0]:.4f}, {bbox_size[1]:.4f}, {bbox_size[2]:.4f}]")
    print(f"  Bounding box volume: {bbox_volume:.4f}")
    
    # Centroid and spread
    centroid = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    
    print(f"\n  Centroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")
    print(f"  Distance from centroid:")
    print(f"    Mean:   {np.mean(distances):.4f}")
    print(f"    Std:    {np.std(distances):.4f}")
    print(f"    Min:    {np.min(distances):.4f}")
    print(f"    Max:    {np.max(distances):.4f}")
    
    # Density estimate
    density = n / bbox_volume if bbox_volume > 0 else 0
    print(f"\n  Gaussian density: {density:.2f} per unit^3")
    
    # Convex hull (for more accurate volume)
    try:
        if n > 4:
            hull = ConvexHull(positions)
            hull_volume = hull.volume
            fill_ratio = hull_volume / bbox_volume if bbox_volume > 0 else 0
            print(f"  Convex hull volume: {hull_volume:.4f}")
            print(f"  Fill ratio (hull/bbox): {fill_ratio:.2%}")
    except Exception as e:
        print(f"  Could not compute convex hull: {e}")
    
    return {
        'bbox_min': bbox_min.tolist(),
        'bbox_max': bbox_max.tolist(),
        'bbox_size': bbox_size.tolist(),
        'bbox_volume': float(bbox_volume),
        'centroid': centroid.tolist(),
        'distance_mean': float(np.mean(distances)),
        'distance_std': float(np.std(distances)),
        'density': float(density),
    }


def analyze_density_grid(data, resolution=30):
    """Analyze density distribution using a 3D grid."""
    print("\n" + "=" * 60)
    print("DENSITY GRID ANALYSIS")
    print("=" * 60)
    
    positions = data['positions']
    n = len(positions)
    
    bbox_min = np.min(positions, axis=0)
    bbox_max = np.max(positions, axis=0)
    bbox_size = bbox_max - bbox_min
    cell_size = bbox_size / resolution
    
    print(f"  Grid resolution: {resolution}^3 = {resolution**3:,} cells")
    print(f"  Cell size: [{cell_size[0]:.4f}, {cell_size[1]:.4f}, {cell_size[2]:.4f}]")
    
    # Count Gaussians per cell
    cell_indices = ((positions - bbox_min) / cell_size).astype(int)
    cell_indices = np.clip(cell_indices, 0, resolution - 1)
    
    density_grid = np.zeros((resolution, resolution, resolution), dtype=np.int32)
    for idx in cell_indices:
        density_grid[idx[0], idx[1], idx[2]] += 1
    
    # Statistics
    non_empty = density_grid[density_grid > 0]
    empty_cells = np.sum(density_grid == 0)
    occupied_cells = np.sum(density_grid > 0)
    
    print(f"\n  Empty cells:    {empty_cells:,} ({100*empty_cells/resolution**3:.1f}%)")
    print(f"  Occupied cells: {occupied_cells:,} ({100*occupied_cells/resolution**3:.1f}%)")
    
    if len(non_empty) > 0:
        print(f"\n  Gaussians per occupied cell:")
        print(f"    Min:    {np.min(non_empty)}")
        print(f"    Max:    {np.max(non_empty)}")
        print(f"    Mean:   {np.mean(non_empty):.2f}")
        print(f"    Median: {np.median(non_empty):.2f}")
        print(f"    Std:    {np.std(non_empty):.2f}")
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\n  Density percentiles (occupied cells):")
        for p in percentiles:
            val = np.percentile(non_empty, p)
            print(f"    p{p:02d}: {val:.1f}")
    
    # Identify sparse regions (edge cells with low density)
    occupied_mask = density_grid > 0
    dilated = binary_dilation(occupied_mask, np.ones((3, 3, 3)), iterations=1)
    edge_mask = dilated & ~occupied_mask
    
    # Also find occupied cells with very low density
    if len(non_empty) > 0:
        sparse_threshold = np.percentile(non_empty, 10)
        sparse_occupied = (density_grid > 0) & (density_grid < sparse_threshold)
        
        print(f"\n  Sparse region analysis:")
        print(f"    Sparse threshold: {sparse_threshold:.1f} Gaussians/cell")
        print(f"    Empty edge cells (potential holes): {np.sum(edge_mask):,}")
        print(f"    Sparse occupied cells: {np.sum(sparse_occupied):,}")
    
    return {
        'resolution': resolution,
        'empty_cells': int(empty_cells),
        'occupied_cells': int(occupied_cells),
        'density_mean': float(np.mean(non_empty)) if len(non_empty) > 0 else 0,
        'density_std': float(np.std(non_empty)) if len(non_empty) > 0 else 0,
    }


def analyze_opacity(data):
    """Analyze opacity distribution."""
    if 'opacity' not in data:
        print("\n  No opacity data available")
        return {}
    
    print("\n" + "=" * 60)
    print("OPACITY ANALYSIS")
    print("=" * 60)
    
    opacity = data['opacity']
    
    print(f"  Range: [{np.min(opacity):.4f}, {np.max(opacity):.4f}]")
    print(f"  Mean:  {np.mean(opacity):.4f}")
    print(f"  Std:   {np.std(opacity):.4f}")
    
    # Distribution
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(opacity, bins=bins)
    
    print("\n  Opacity distribution:")
    for i in range(len(hist)):
        pct = 100 * hist[i] / len(opacity)
        bar = "#" * int(pct / 2)
        print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {bar} ({pct:.1f}%)")
    
    # Low opacity Gaussians (often noise)
    low_opacity = np.sum(opacity < 0.1)
    high_opacity = np.sum(opacity > 0.9)
    
    print(f"\n  Low opacity (<0.1):  {low_opacity:,} ({100*low_opacity/len(opacity):.1f}%)")
    print(f"  High opacity (>0.9): {high_opacity:,} ({100*high_opacity/len(opacity):.1f}%)")
    
    return {
        'min': float(np.min(opacity)),
        'max': float(np.max(opacity)),
        'mean': float(np.mean(opacity)),
        'std': float(np.std(opacity)),
        'low_opacity_count': int(low_opacity),
        'high_opacity_count': int(high_opacity),
    }


def analyze_scales(data):
    """Analyze Gaussian scale distribution."""
    if 'scales' not in data:
        print("\n  No scale data available")
        return {}
    
    print("\n" + "=" * 60)
    print("SCALE ANALYSIS")
    print("=" * 60)
    
    scales = data['scales']
    avg_scale = np.mean(scales, axis=1)
    max_scale = np.max(scales, axis=1)
    min_scale = np.min(scales, axis=1)
    
    print(f"  Average scale per Gaussian:")
    print(f"    Range:  [{np.min(avg_scale):.6f}, {np.max(avg_scale):.6f}]")
    print(f"    Mean:   {np.mean(avg_scale):.6f}")
    print(f"    Median: {np.median(avg_scale):.6f}")
    print(f"    Std:    {np.std(avg_scale):.6f}")
    
    print(f"\n  Max scale per Gaussian:")
    print(f"    Range:  [{np.min(max_scale):.6f}, {np.max(max_scale):.6f}]")
    print(f"    Mean:   {np.mean(max_scale):.6f}")
    
    # Percentiles
    percentiles = [50, 75, 90, 95, 99]
    print(f"\n  Scale percentiles (average):")
    for p in percentiles:
        val = np.percentile(avg_scale, p)
        print(f"    p{p:02d}: {val:.6f}")
    
    # Identify problematic scales
    very_small = np.sum(avg_scale < np.percentile(avg_scale, 1))
    very_large = np.sum(avg_scale > np.percentile(avg_scale, 99))
    
    print(f"\n  Potential issues:")
    print(f"    Very small (<p01): {very_small:,}")
    print(f"    Very large (>p99): {very_large:,}")
    
    # Aspect ratio (elongated Gaussians)
    aspect_ratio = max_scale / (min_scale + 1e-8)
    elongated = np.sum(aspect_ratio > 10)
    print(f"    Highly elongated (aspect > 10): {elongated:,}")
    
    return {
        'avg_scale_mean': float(np.mean(avg_scale)),
        'avg_scale_median': float(np.median(avg_scale)),
        'avg_scale_std': float(np.std(avg_scale)),
        'very_small_count': int(very_small),
        'very_large_count': int(very_large),
        'elongated_count': int(elongated),
    }


def analyze_outliers(data, std_threshold=3.0):
    """Identify potential floater/outlier Gaussians."""
    print("\n" + "=" * 60)
    print("OUTLIER ANALYSIS")
    print("=" * 60)
    
    positions = data['positions']
    centroid = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    outlier_threshold = mean_dist + std_threshold * std_dist
    outliers = np.sum(distances > outlier_threshold)
    
    print(f"  Outlier threshold: {outlier_threshold:.4f} ({std_threshold} std from mean)")
    print(f"  Position outliers: {outliers:,} ({100*outliers/len(positions):.2f}%)")
    
    # Combine with opacity if available
    if 'opacity' in data:
        opacity = data['opacity']
        low_opacity_far = np.sum((distances > outlier_threshold) & (opacity < 0.3))
        print(f"  Far + low opacity (likely floaters): {low_opacity_far:,}")
    
    # Isolated Gaussians (far from neighbors)
    print("\n  Checking for isolated Gaussians...")
    tree = KDTree(positions)
    k = min(6, len(positions))  # k nearest neighbors
    dists, _ = tree.query(positions, k=k)
    avg_neighbor_dist = np.mean(dists[:, 1:], axis=1)  # Skip self (dist=0)
    
    isolation_threshold = np.percentile(avg_neighbor_dist, 99)
    isolated = np.sum(avg_neighbor_dist > isolation_threshold)
    
    print(f"  Isolation threshold (p99): {isolation_threshold:.6f}")
    print(f"  Isolated Gaussians: {isolated:,}")
    
    return {
        'position_outliers': int(outliers),
        'outlier_threshold': float(outlier_threshold),
        'isolated_count': int(isolated),
        'isolation_threshold': float(isolation_threshold),
    }


def estimate_coverage(data, n_views=26):
    """
    Estimate view coverage by projecting Gaussians from multiple angles.
    Uses a simple projection (ignoring Gaussian shapes) to identify
    potential holes visible from certain angles.
    """
    print("\n" + "=" * 60)
    print("COVERAGE ESTIMATION")
    print("=" * 60)
    
    positions = data['positions']
    n = len(positions)
    
    # Center the positions
    centroid = np.mean(positions, axis=0)
    centered = positions - centroid
    
    # Generate view directions (roughly uniformly distributed)
    # Using golden spiral on sphere
    views = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
    
    for i in range(n_views):
        y = 1 - (i / float(n_views - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        views.append(np.array([x, y, z]))
    
    views = np.array(views)
    
    print(f"  Analyzing {n_views} view directions...")
    
    # For each view, project points and analyze 2D coverage
    resolution = 100
    min_coverage = float('inf')
    max_coverage = 0
    coverage_values = []
    
    for view_idx, view_dir in enumerate(views):
        # Create orthonormal basis for projection
        view_dir = view_dir / np.linalg.norm(view_dir)
        
        # Find perpendicular vectors
        if abs(view_dir[0]) < 0.9:
            up = np.array([1, 0, 0])
        else:
            up = np.array([0, 1, 0])
        
        right = np.cross(view_dir, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, view_dir)
        
        # Project points onto 2D plane
        proj_x = np.dot(centered, right)
        proj_y = np.dot(centered, up)
        
        # Normalize to grid
        px_min, px_max = np.min(proj_x), np.max(proj_x)
        py_min, py_max = np.min(proj_y), np.max(proj_y)
        
        if px_max - px_min < 1e-6 or py_max - py_min < 1e-6:
            continue
        
        px_norm = ((proj_x - px_min) / (px_max - px_min) * (resolution - 1)).astype(int)
        py_norm = ((proj_y - py_min) / (py_max - py_min) * (resolution - 1)).astype(int)
        
        # Create coverage map
        coverage_map = np.zeros((resolution, resolution), dtype=bool)
        for i in range(n):
            coverage_map[px_norm[i], py_norm[i]] = True
        
        # Dilate slightly (Gaussians have extent)
        coverage_map = binary_dilation(coverage_map, np.ones((3, 3)))
        
        # Calculate coverage
        coverage = np.sum(coverage_map) / (resolution * resolution)
        coverage_values.append(coverage)
        
        min_coverage = min(min_coverage, coverage)
        max_coverage = max(max_coverage, coverage)
    
    coverage_values = np.array(coverage_values)
    
    print(f"\n  Coverage statistics across views:")
    print(f"    Min coverage:  {min_coverage:.1%}")
    print(f"    Max coverage:  {max_coverage:.1%}")
    print(f"    Mean coverage: {np.mean(coverage_values):.1%}")
    print(f"    Std coverage:  {np.std(coverage_values):.1%}")
    
    # Flag potential problem views
    problem_threshold = np.mean(coverage_values) - 2 * np.std(coverage_values)
    problem_views = np.sum(coverage_values < problem_threshold)
    
    print(f"\n  Views with low coverage (< mean - 2*std):")
    print(f"    Count: {problem_views}")
    
    if problem_views > 0:
        print("    [!] Some view angles may show holes")
    else:
        print("    Coverage appears consistent across views")
    
    return {
        'n_views': n_views,
        'min_coverage': float(min_coverage),
        'max_coverage': float(max_coverage),
        'mean_coverage': float(np.mean(coverage_values)),
        'std_coverage': float(np.std(coverage_values)),
        'problem_views': int(problem_views),
    }


def generate_report(data, output_path=None):
    """Generate full analysis report."""
    print("\n" + "#" * 60)
    print("# GAUSSIAN SPLAT ANALYSIS REPORT")
    print("#" * 60)
    print(f"\n  Total Gaussians: {data['n_gaussians']:,}")
    print(f"  Properties: {len(data['properties'])}")
    
    report = {
        'n_gaussians': data['n_gaussians'],
        'properties': data['properties'],
    }
    
    report['geometry'] = analyze_geometry(data)
    report['density'] = analyze_density_grid(data)
    report['opacity'] = analyze_opacity(data)
    report['scales'] = analyze_scales(data)
    report['outliers'] = analyze_outliers(data)
    report['coverage'] = estimate_coverage(data)
    
    # Summary and recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = []
    
    if report['outliers'].get('position_outliers', 0) > data['n_gaussians'] * 0.01:
        recommendations.append("- Consider running floater removal (>1% outliers detected)")
    
    if report['opacity'].get('low_opacity_count', 0) > data['n_gaussians'] * 0.1:
        recommendations.append("- Many low-opacity Gaussians (>10%) - consider filtering")
    
    if report['density'].get('empty_cells', 0) > report['density'].get('occupied_cells', 1) * 2:
        recommendations.append("- High sparsity detected - consider densification")
    
    if report['coverage'].get('problem_views', 0) > 0:
        recommendations.append("- Some view angles show potential holes - check these areas")
    
    if report['scales'].get('elongated_count', 0) > data['n_gaussians'] * 0.05:
        recommendations.append("- Many elongated Gaussians (>5%) - may cause artifacts")
    
    if not recommendations:
        recommendations.append("- Splat appears healthy, no major issues detected")
    
    report['recommendations'] = recommendations
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n" + "#" * 60)
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved to: {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Gaussian Splat quality and identify potential issues"
    )
    
    parser.add_argument("input", help="Input .ply Gaussian Splat file")
    parser.add_argument("--output", "-o", help="Output JSON report file")
    parser.add_argument("--views", type=int, default=26,
                        help="Number of view directions for coverage analysis (default: 26)")
    
    args = parser.parse_args()
    
    data = load_splat_data(args.input)
    generate_report(data, args.output)


if __name__ == "__main__":
    main()
