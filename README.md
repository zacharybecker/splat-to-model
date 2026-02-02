# Gaussian Splat Pipeline for Unity

Convert Gaussian Splat PLY files (from Postshot/Jawset) to Unity with two approaches:

## Two Approaches

### 1. Mesh Conversion (Traditional)
```
Gaussian Splat (.ply) --> Point Cloud --> Mesh (.obj) --> Unity (Standard Renderer)
```
- Works with standard Unity workflow (physics, colliders, lightmaps)
- Best for: game objects, mobile, WebGL

### 2. Direct Splat Rendering (NEW)
```
Gaussian Splat (.ply) --> Enhanced Splat (.ply) --> Unity (Gaussian Splat Renderer)
```
- Preserves view-dependent effects (specularity, transparency)
- Requires [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting)
- Best for: photorealistic visualization, archviz, VR

## Prerequisites

- Docker Desktop

## Quick Start

1. Place your `.ply` file in the `data/` folder
2. Edit `docker-compose.yml` to set your input/output filenames
3. Run:

```powershell
docker compose up --build
```

## Configuration

All settings are configured via environment variables in `docker-compose.yml`:

```yaml
services:
  splat-to-mesh:
    environment:
      # Required
      - INPUT_FILE=model.ply      # Your input file in data/
      - OUTPUT_FILE=mesh.obj      # Output filename in data/
      
      # Point extraction
      - OPACITY_THRESHOLD=0.3     # 0-1, lower = more points
      # - MAX_SCALE=0.05          # Filter large Gaussians
      
      # Mesh generation  
      # - VOXEL_SIZE=0            # 0 = no downsampling
      # - TARGET_TRIANGLES=50000  # Simplify mesh
      - OUTLIER_STD_RATIO=2.0     # Lower = more aggressive cleanup
      - THIN_GEOMETRY=true        # Better for thin features
      
      # Mesh fixes
      - FLIP_NORMALS=false        # Fix inside-out mesh
      - DOUBLE_SIDED=false        # Visible from both sides
      - SMOOTH_FINAL=false        # Apply smoothing
      # - SMOOTH_ITERATIONS=5
      
      # Output
      - KEEP_INTERMEDIATE=false   # Keep point cloud file
      - VERBOSE=true
```

## Environment Variables

### Mesh Mode (`splat-to-mesh`)

| Variable | Description | Default |
|----------|-------------|---------|
| `INPUT_FILE` | Input Gaussian Splat PLY file (required) | - |
| `OUTPUT_FILE` | Output mesh file (required) | - |
| `OPACITY_THRESHOLD` | Minimum opacity for points (0-1). Lower = more points | 0.3 |
| `MAX_SCALE` | Maximum Gaussian scale to include | - |
| `VOXEL_SIZE` | Voxel size for downsampling (0 = disabled) | auto |
| `TARGET_TRIANGLES` | Target triangle count for simplification | - |
| `OUTLIER_STD_RATIO` | Outlier removal aggressiveness (lower = more) | 2.0 |
| `THIN_GEOMETRY` | Add extra radii for thin features | true |
| `FLIP_NORMALS` | Flip mesh normals if inside-out | false |
| `DOUBLE_SIDED` | Make mesh visible from both sides | false |
| `SMOOTH_FINAL` | Apply Taubin smoothing | false |
| `SMOOTH_ITERATIONS` | Number of smoothing iterations | 5 |
| `KEEP_INTERMEDIATE` | Keep intermediate point cloud | false |
| `VERBOSE` | Print progress information | true |

### Enhance Mode (`splat-enhance`)

| Variable | Description | Default |
|----------|-------------|---------|
| `INPUT_FILE` | Input Gaussian Splat PLY file (required) | - |
| `OUTPUT_FILE` | Output enhanced PLY file (required) | - |
| `REMOVE_FLOATERS` | Remove outlier Gaussians | true |
| `DENSIFY_SPARSE` | Fill sparse regions with interpolated Gaussians | true |
| `SPLIT_LARGE` | Split large/blobby Gaussians | true |
| `GRID_RESOLUTION` | 3D grid resolution for sparse detection | 50 |
| `DENSITY_THRESHOLD` | Percentile below which cells are "sparse" | 10 |
| `VERBOSE` | Print progress information | true |

## Quality Presets

Edit the environment variables in `docker-compose.yml`:

**Low Quality (Fast)**
```yaml
- OPACITY_THRESHOLD=0.5
- TARGET_TRIANGLES=20000
```

**Medium Quality**
```yaml
- OPACITY_THRESHOLD=0.3
- TARGET_TRIANGLES=50000
```

**High Quality**
```yaml
- OPACITY_THRESHOLD=0.2
- VOXEL_SIZE=0
```

**Ultra Quality**
```yaml
- OPACITY_THRESHOLD=0.1
- VOXEL_SIZE=0
```

## Example Usage

### Mesh Mode (Default)

```powershell
# Build and run with default settings
docker compose run --build splat-to-mesh

# Run with custom env vars (one-off)
docker compose run -e INPUT_FILE=scan.ply -e OUTPUT_FILE=result.obj splat-to-mesh
```

### Enhance Mode (For Direct Splat Rendering)

```powershell
# Enhance a splat for direct rendering in Unity
docker compose run --build splat-enhance

# Custom enhancement settings
docker compose run -e INPUT_FILE=scan.ply -e OUTPUT_FILE=enhanced.ply -e GRID_RESOLUTION=80 splat-enhance
```

See `SPLAT_RENDERING_GUIDE.md` for detailed documentation on direct splat rendering.

## Platform-Specific Triangle Counts

| Platform | Recommended `TARGET_TRIANGLES` |
|----------|-------------------------------|
| Mobile | 10,000 - 50,000 |
| Desktop | 50,000 - 200,000 |
| High-end Desktop | 200,000+ |

## Unity Import

1. Drag the `.obj` file into your Unity project
2. Create a material with a vertex color shader:
   - URP/HDRP: Create Shader Graph with Vertex Color node
   - Built-in: Use `Particles/Standard Unlit` shader

See `VIDEO_TO_UNITY_QUICKSTART.md` for detailed shader setup.

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Empty mesh output | Too few points extracted | Lower `OPACITY_THRESHOLD` (try 0.1) |
| Mesh has holes | Sparse point coverage | Lower `OPACITY_THRESHOLD` |
| See-through faces | Inconsistent normals | Set `DOUBLE_SIDED=true` |
| Mesh inside-out | Wrong normal direction | Set `FLIP_NORMALS=true` |
| Too many polygons | No simplification | Set `TARGET_TRIANGLES=50000` |
| No colors on mesh | Color transfer failed | Check input PLY has color data |

## File Structure

```
project/
    data/
        model.ply              # Input: Gaussian Splat from Postshot
        mesh.obj               # Output: Unity-ready mesh (mesh mode)
        enhanced.ply           # Output: Enhanced splat (enhance mode)
    
    # Pipeline scripts
    run_pipeline.py            # Main pipeline (reads env vars)
    splat_to_pointcloud.py     # Extract points from splat
    pointcloud_to_mesh.py      # Generate mesh (BPA/Poisson)
    splat_enhance.py           # Enhance splat quality
    splat_analyze.py           # Analyze splat for quality issues
    
    # Documentation
    SPLAT_RENDERING_GUIDE.md   # Guide for direct splat rendering
    VIDEO_TO_UNITY_QUICKSTART.md
    PIPELINE_PLAN.md
    
    # Docker files
    Dockerfile
    docker-compose.yml
    requirements.txt
```

## Full Workflow

1. **Capture video** of your object (30-90 seconds, orbit around it)
2. **Process in Postshot** (Jawset) to create Gaussian Splat
3. **Export as PLY** from Postshot
4. **Place PLY in `data/` folder**
5. **Edit `docker-compose.yml`** with your filenames
6. **Run `docker compose up --build`**
7. **Import mesh to Unity** and apply vertex color material

See `VIDEO_TO_UNITY_QUICKSTART.md` for detailed instructions.
