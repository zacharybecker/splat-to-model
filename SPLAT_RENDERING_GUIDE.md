# Direct Gaussian Splat Rendering in Unity

This guide covers the alternative approach of rendering Gaussian splats directly in Unity instead of converting them to meshes.

## Overview: Two Approaches

### Approach 1: Mesh Conversion (Traditional)

```
Splat PLY --> Point Cloud --> Mesh --> Unity (Standard Renderer)
```

**Pros:**
- Works with standard Unity workflow
- Compatible with physics, colliders, LOD
- Can bake lightmaps
- Smaller runtime footprint

**Cons:**
- Loses view-dependent effects (specularity, transparency variations)
- Surface reconstruction can introduce artifacts
- Holes and gaps require mesh repair

### Approach 2: Direct Splat Rendering (This Guide)

```
Splat PLY --> Enhanced Splat --> Unity (Gaussian Splat Renderer)
```

**Pros:**
- Preserves view-dependent effects
- No surface reconstruction artifacts
- Better visual quality for photorealistic captures
- Faster pipeline (no mesh generation)

**Cons:**
- Requires special renderer (aras-p/UnityGaussianSplatting)
- Higher GPU memory usage
- No standard physics/collision support
- Limited to D3D12, Metal, Vulkan (no DX11, WebGL)

---

## Setting Up Direct Splat Rendering

### 1. Install UnityGaussianSplatting

Clone or download from: https://github.com/aras-p/UnityGaussianSplatting

Requirements:
- Unity 2022.3+ recommended
- D3D12 (Windows), Metal (Mac), or Vulkan (Linux) - **DX11 will NOT work**
- GPU with decent memory (1GB+ for typical scenes)

Installation:
1. Open the project `projects/GaussianExample` or add the package to your project
2. Ensure graphics API is set to D3D12/Vulkan/Metal in Player Settings
3. Open Tools > Gaussian Splats > Create GaussianSplatAsset

### 2. Import Your Splat

The renderer supports:
- Standard 3DGS PLY files (`point_cloud/iteration_*/point_cloud.ply`)
- Scaniverse SPZ format

Import steps:
1. Go to Tools > Gaussian Splats > Create GaussianSplatAsset
2. Select your PLY file
3. Choose compression level (Very Low is ~8MB and still decent)
4. Click "Create Asset"

### 3. Add to Scene

1. Create a GameObject
2. Add `GaussianSplatRenderer` component
3. Assign your created asset
4. Adjust transform (most 3DGS models need -160 degree X rotation, Z mirror)

---

## Improving Splat Quality

### The Problem: Holes and Artifacts

Gaussian splats can have quality issues:
- **Holes**: Areas not captured in training images
- **Floaters**: Stray Gaussians in mid-air
- **Sparse regions**: Under-densified areas
- **Large blobby Gaussians**: Artifacts from optimization

### Solution: Splat Enhancement Pipeline

This repository provides tools to improve splat quality before rendering.

#### Analyze Your Splat

```bash
python splat_analyze.py model.ply --output report.json
```

This generates a report showing:
- Density distribution
- Potential holes (sparse regions)
- Outlier Gaussians (floaters)
- Coverage from multiple viewpoints

#### Enhance Your Splat

```bash
python splat_enhance.py input.ply output.ply
```

Enhancement operations:
1. **Floater Removal**: Removes outlier Gaussians far from the main geometry
2. **Sparse Region Densification**: Adds new Gaussians in under-represented areas
3. **Large Gaussian Splitting**: Breaks up blobby Gaussians into smaller ones

Options:
```bash
# Full enhancement (default)
python splat_enhance.py input.ply output.ply

# Only remove floaters
python splat_enhance.py input.ply output.ply --no-densify --no-split

# Aggressive hole filling
python splat_enhance.py input.ply output.ply --grid 30 --density-threshold 20

# Fine grid for detailed models
python splat_enhance.py input.ply output.ply --grid 100
```

#### Using Docker

```bash
# Set MODE=enhance for splat enhancement
docker compose run --rm \
  -e MODE=enhance \
  -e INPUT_FILE=model.ply \
  -e OUTPUT_FILE=enhanced.ply \
  -e REMOVE_FLOATERS=true \
  -e DENSIFY_SPARSE=true \
  -e SPLIT_LARGE=true \
  splat-pipeline
```

---

## Enhancement Algorithms

### 1. Floater Removal

Identifies and removes Gaussians that are likely artifacts:
- **Position outliers**: Gaussians > 3 standard deviations from centroid
- **Low opacity**: Gaussians with opacity < 5% (often noise)
- **Extreme scale**: Very large Gaussians (> 3x the 99th percentile)

### 2. Sparse Region Detection

Uses a 3D grid to find under-densified areas:
1. Divides bounding box into cells (default 50x50x50)
2. Counts Gaussians per cell
3. Identifies cells near the surface but with low counts
4. Returns center points of these sparse cells

### 3. Gaussian Densification

Fills sparse regions by interpolating from neighbors:
1. For each sparse region center, find K nearest Gaussians
2. Compute weighted average of their properties (position, color, scale, rotation)
3. Add small random jitter to avoid exact duplicates
4. Insert new Gaussian with interpolated properties

### 4. Large Gaussian Splitting

Breaks up oversized Gaussians:
1. Identify Gaussians above the 95th percentile in scale
2. Split each into two children along the largest axis
3. Position children at +/- 0.5 * scale along that axis
4. Reduce scale of children (halve in log-space)

---

## Advanced: Multi-View Hole Detection

For more sophisticated hole detection, analyze the splat from multiple camera angles:

```python
from splat_analyze import load_splat_data, estimate_coverage

data = load_splat_data("model.ply")
coverage = estimate_coverage(data, n_views=50)

if coverage['problem_views'] > 0:
    print("Some angles show potential holes")
```

This projects the splat from ~50 uniformly distributed view directions and identifies angles with lower-than-average coverage.

---

## Best Practices

### For Capture/Training

1. **More images = fewer holes**: Capture from many angles
2. **Overlap**: Ensure good overlap between views
3. **Lighting**: Consistent lighting reduces artifacts
4. **Train longer**: More iterations can fill sparse regions

### For Post-Processing

1. **Analyze first**: Run `splat_analyze.py` to understand issues
2. **Start conservative**: Default settings work for most cases
3. **Iterate**: Enhance, view in Unity, enhance again if needed
4. **Keep originals**: Enhancement modifies the splat permanently

### For Unity

1. **Compression**: Use Medium or Low quality for balance
2. **Transform**: Check rotation/scale - 3DGS models often need adjustment
3. **Performance**: Monitor GPU memory with large splats
4. **VR**: Works with some headsets (Quest 3, Vive), not all

---

## Comparison: When to Use Each Approach

| Use Case | Recommended Approach |
|----------|---------------------|
| Photorealistic visualization | Direct Splat |
| Game with physics | Mesh |
| AR/VR experience | Direct Splat (if supported) |
| WebGL deployment | Mesh (splats not supported) |
| Mobile | Mesh (more reliable) |
| Archviz/Product viz | Direct Splat |
| Interactive objects | Mesh |
| Background environments | Direct Splat |

---

## Troubleshooting

### "Black screen" in Unity
- Check graphics API is D3D12/Vulkan/Metal, not DX11
- Verify asset was created successfully

### Splat looks wrong after enhancement
- Try with only floater removal first (`--no-densify --no-split`)
- Check the analysis report for unusual statistics

### Large file after enhancement
- Densification adds Gaussians - expected
- Use `--no-densify` if you only want cleanup

### Holes still visible
- Increase grid resolution (`--grid 100`)
- Increase density threshold (`--density-threshold 20`)
- Some holes may be unfillable without new training data

---

## References

- [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) - Unity renderer
- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Original research
- [SuperSplat](https://playcanvas.com/supersplat/editor) - Web-based splat editor
- [Postshot](https://jawset.com/) - Commercial 3DGS tool
