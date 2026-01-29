# Original Splat-to-Mesh Pipeline

A straightforward two-stage pipeline that converts Gaussian Splat files to Unity-ready meshes.

---

## Overview

```
[Gaussian Splat PLY] --> [Point Cloud] --> [Triangle Mesh]
```

The original pipeline prioritizes simplicity and reliability. It uses well-established algorithms with sensible defaults that work for most cases.

---

## Stage 1: Splat to Point Cloud

**Script:** `splat_to_pointcloud.py`

### What It Does

Extracts the center positions and colors from each Gaussian in the splat file, filtering out low-confidence (low-opacity) data.

### The Process

1. **Parse Gaussian Data**
   - Reads PLY file containing millions of Gaussians
   - Each Gaussian has: position, color (as Spherical Harmonics), scale, rotation, opacity

2. **Extract Positions**
   - Takes the (x, y, z) center of each Gaussian
   - Handles multiple field naming conventions (x/y/z, px/py/pz, etc.)

3. **Extract Colors**
   - Converts Spherical Harmonics DC coefficients to RGB
   - Formula: `RGB = SH_DC * 0.28209 + 0.5` then scale to 0-255
   - Falls back to direct RGB if SH not present

4. **Opacity Filtering**
   - The key noise reduction step
   - Removes Gaussians below the opacity threshold
   - Auto-detects logit vs direct opacity format
   - Default threshold: 0.5 (keeps ~50-80% of points typically)

5. **Output Point Cloud**
   - Writes standard PLY with x, y, z, red, green, blue per vertex
   - Compatible with any point cloud processing tool

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--opacity` | 0.5 | Minimum opacity (0-1). Lower = more points, more noise |
| `--inspect` | - | Just analyze the PLY structure without converting |

### Example Usage

```bash
# Basic extraction
python splat_to_pointcloud.py model.ply pointcloud.ply

# More aggressive filtering (cleaner but fewer points)
python splat_to_pointcloud.py model.ply pointcloud.ply --opacity 0.6

# Keep more data (noisier but more complete)
python splat_to_pointcloud.py model.ply pointcloud.ply --opacity 0.2

# Inspect file structure first
python splat_to_pointcloud.py model.ply --inspect
```

---

## Stage 2: Point Cloud to Mesh

**Script:** `pointcloud_to_mesh.py`

### What It Does

Converts the point cloud into a solid, watertight triangle mesh using Poisson Surface Reconstruction.

### The Process

1. **Load Point Cloud**
   - Reads PLY into Open3D
   - Validates positions and colors

2. **Preprocessing**
   - **Voxel Downsampling** (optional): Reduces point density for faster processing
     - Auto-calculates voxel size as 0.5% of scene diagonal
     - Can be disabled with `--voxel-size 0`
   - **Outlier Removal**: Statistical filter removes isolated noise points
     - Uses 30 nearest neighbors
     - Points beyond std_ratio * standard deviation are removed

3. **Normal Estimation**
   - Required for Poisson reconstruction
   - Uses hybrid KD-tree search (radius + max neighbors)
   - Auto-calculates search radius as 1% of smallest dimension
   - Orients normals consistently using tangent plane method

4. **Poisson Surface Reconstruction**
   - Creates a continuous, watertight surface from oriented points
   - The `depth` parameter controls octree resolution
   - Also outputs "density" values indicating reconstruction confidence

5. **Mesh Cleaning**
   - Removes low-density vertices (Poisson artifacts at boundaries)
   - Removes degenerate triangles (zero area)
   - Removes duplicate geometry
   - Removes non-manifold edges

6. **Color Transfer**
   - Builds KD-tree from original point cloud
   - For each mesh vertex, finds nearest point cloud point
   - Copies that point's color to the vertex

7. **Simplification** (optional)
   - Quadric decimation to reduce triangle count
   - Preserves visual quality while reducing geometry

8. **Save Mesh**
   - Supports OBJ, PLY, STL, and other formats
   - OBJ recommended for Unity compatibility

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--depth` | 9 | Poisson octree depth (6-11). Higher = more detail |
| `--density-threshold` | 0.01 | Remove bottom X% low-confidence vertices |
| `--voxel-size` | auto | Downsample voxel size. 0 = disabled |
| `--simplify` | none | Target triangle count |
| `--outlier-std` | 2.0 | Outlier removal aggressiveness. Lower = more removal |

### Example Usage

```bash
# Basic mesh generation
python pointcloud_to_mesh.py pointcloud.ply mesh.obj

# Higher detail (more triangles)
python pointcloud_to_mesh.py pointcloud.ply mesh.obj --depth 10

# Mobile-optimized (fewer triangles)
python pointcloud_to_mesh.py pointcloud.ply mesh.obj --depth 8 --simplify 50000

# No downsampling (preserve all detail)
python pointcloud_to_mesh.py pointcloud.ply mesh.obj --voxel-size 0
```

---

## Combined Pipeline

**Script:** `run_pipeline.py`

Runs both stages with a single command.

### Example Usage

```bash
# Basic usage
python run_pipeline.py model.ply mesh.obj

# Quality preset examples
python run_pipeline.py model.ply mesh.obj --depth 7 --opacity 0.5 --simplify 20000  # Low
python run_pipeline.py model.ply mesh.obj --depth 8 --opacity 0.3 --simplify 50000  # Medium
python run_pipeline.py model.ply mesh.obj --depth 9 --opacity 0.2                   # High
python run_pipeline.py model.ply mesh.obj --depth 10 --opacity 0.1                  # Ultra

# Keep intermediate point cloud for inspection
python run_pipeline.py model.ply mesh.obj --keep-intermediate
```

---

## Understanding Poisson Depth

The `--depth` parameter is the most important quality control:

| Depth | Triangles | Quality | Speed | Use Case |
|-------|-----------|---------|-------|----------|
| 6 | ~10K | Low | Fast | Preview/testing |
| 7 | ~40K | Low-Med | Fast | Mobile games |
| 8 | ~150K | Medium | Medium | Standard games |
| 9 | ~600K | High | Slow | Desktop games (default) |
| 10 | ~2M | Very High | Very Slow | High-end/film |
| 11 | ~8M | Maximum | Extremely Slow | Archival quality |

**Note:** Actual triangle counts depend on scene complexity.

---

## Troubleshooting

### Problem: Mesh has large holes
- **Cause:** Not enough points in that area
- **Solution:** Lower opacity threshold (`--opacity 0.2`)

### Problem: Mesh has noise/bumps
- **Cause:** Too many low-quality points included
- **Solution:** Raise opacity threshold (`--opacity 0.5` or higher)

### Problem: Mesh is too smooth, lost detail
- **Cause:** Depth too low or too much downsampling
- **Solution:** Increase depth (`--depth 10`), disable downsampling (`--voxel-size 0`)

### Problem: Processing takes forever
- **Cause:** Too many points or depth too high
- **Solution:** Decrease depth (`--depth 8`), enable downsampling

### Problem: Colors look wrong
- **Cause:** SH to RGB conversion issue or file format mismatch
- **Solution:** Use `--inspect` to check field names, verify input format

---

## Output Formats

| Format | Extension | Colors | Unity Support | Notes |
|--------|-----------|--------|---------------|-------|
| Wavefront OBJ | .obj | Vertex colors | Good | Recommended |
| PLY | .ply | Vertex colors | Good | Good for point clouds |
| STL | .stl | No | Medium | Geometry only |
| GLTF | .gltf | Yes | Excellent | Modern format |

**For Unity:** Use OBJ format with a vertex color shader.
