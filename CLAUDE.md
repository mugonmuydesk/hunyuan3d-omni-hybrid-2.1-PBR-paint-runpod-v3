# Hunyuan3D Omni Hybrid Pipeline

## Overview

RunPod serverless endpoint for hybrid 3D generation with mode selection:
- **Quality mode** (default): Hunyuan3D-Omni (3.3B params, SiT architecture)
- **Fast mode** (`--fast`): Hunyuan3D-DiT-v2-mini-Fast (0.6B params)
- **Pose-conditioned** (`--skeleton`): Omni with skeleton/bone input
- **Textures**: Hunyuan3D-2.1 PaintPBR (PBR materials)

- **GitHub Repo**: `mugonmuydesk/hunyuan3d-omni-hybrid-2.1-PBR-paint-runpod-v3`
- **Endpoint ID**: `ixl24shhyt21s1`
- **Local Path**: `C:\dev\hunyuan3d-omni-hybrid-2.1-PBR-paint-runpod-v3` (or `/mnt/e/hunyuan3d-omni-hybrid-v3`)
- **GPU Requirements**: 24GB minimum, 48GB recommended

## Build

```bash
# Build locally in WSL2 (native Linux fs for speed)
cp -r /mnt/c/dev/hunyuan3d-omni-hybrid-2.1-PBR-paint-runpod-v3 ~/hunyuan3d-omni-hybrid-v3
cd ~/hunyuan3d-omni-hybrid-v3
docker build -t hunyuan3d-omni-hybrid-v3:latest .

# Or from E: drive (slower, Windows fs)
cd /mnt/e/hunyuan3d-omni-hybrid-v3
docker build -t hunyuan3d-omni-hybrid-v3:latest .
```

**Build time:** ~7 min locally, ~15 min on RunPod (30 min limit).
**Image size:** ~30 GB.

### Build optimizations applied
- Removed unused deps: open3d (447MB), cupy (104MB), pandas, pythreejs, torchaudio, fastapi, uvicorn
- Pinned bpy==4.4.0 (avoids downloading 7 candidate wheels)
- Removed --ignore-installed and Chinese mirror indexes
- Fixed numpy constraint to prevent runpod upgrading to numpy 2.x

## API

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_base64` | string | required | Base64 encoded input image |
| `fast_mode` | bool | false | Use Mini-Fast (0.6B) instead of Omni (3.3B) |
| `skeleton_base64` | string | - | Base64 encoded bone coordinates file (Omni only) |
| `skeleton_data` | array | - | JSON array of bone coords (Omni only) |
| `generate_texture` | bool | true | Generate PBR textures |
| `output_format` | string | "glb" | Output format: "glb" or "obj" |
| `num_views` | int | env | Texture views (1-6) |
| `texture_resolution` | int | env | Texture resolution px |

### Skeleton Format

For pose-conditioned generation (Omni only):
- Each bone: 6 values `[start_x, start_y, start_z, end_x, end_y, end_z]`
- Follows PoseMaster bone definition (body + hand bones)
- `skeleton_base64`: text file with M lines, 6 space-separated values each
- `skeleton_data`: JSON array of M arrays with 6 values each

## VRAM Requirements

| Mode | Pipeline | VRAM |
|------|----------|------|
| Quality shape | Omni 3.3B | ~10 GB |
| Fast shape | Mini-Fast 0.6B | ~3 GB |
| Texture | PaintPBR | ~21 GB |
| **Peak (sequential)** | | **~21 GB** |

Sequential loading with explicit VRAM unload between stages.

## Client Usage

```bash
# Set credentials
export RUNPOD_API_KEY="..."

# Quality mode (Omni - best quality)
python hunyuan3d_client.py image.png -e omni-hybrid -o output.glb

# Fast mode (Mini-Fast - ~2x faster)
python hunyuan3d_client.py image.png -e omni-hybrid --fast -o output.glb

# Pose-conditioned (skeleton file)
python hunyuan3d_client.py image.png -e omni-hybrid --skeleton pose.txt -o output.glb

# Shape only (skip textures)
python hunyuan3d_client.py image.png -e omni-hybrid --no-texture -o output.glb
```

## Models (Network Volume)

Models stored on RunPod Network Volume, not baked into image:
```
/runpod-volume/models/
  Hunyuan3D-Omni/      (~24GB) - Quality shape generation
  Hunyuan3D-2mini/     (~7GB)  - Fast shape generation
  Hunyuan3D-2.1/       (~7GB)  - PBR texture painting
```

## Pipeline Classes

- **Omni**: `Hunyuan3DOmniSiTFlowMatchingPipeline` (SiT architecture, supports skeleton)
- **Mini-Fast**: `Hunyuan3DDiTFlowMatchingPipeline` (DiT architecture)
- **PaintPBR**: `Hunyuan3DPaintPipeline`

## File Structure

```
handler.py              # RunPod handler with hybrid routing
Dockerfile              # Production build
requirements_inference.txt  # Trimmed deps for inference only
custom_rasterizer-*.whl # Pre-built CUDA extension wheel
mesh_utils_noblender.py # bpy subprocess wrapper
bpy_mesh_ops.py         # Blender mesh ops (runs in /opt/bpy-env)
onnx_upscaler.py        # ONNX-based RealESRGAN upscaler
schedulers.py           # Patched schedulers for numpy compat
patches/                # Patched upstream files
```

## Key Functions

- `generate_shape(job_input, image_path, temp_path)` - Routes to Omni or Mini-Fast
- `parse_skeleton_input(job_input, temp_path)` - Parses skeleton from base64 or JSON
- `load_omni_pipeline()` / `load_fast_pipeline()` - Lazy pipeline loaders
- `unload_shape_pipelines()` - VRAM cleanup between stages

## RunPod Build Constraints

| Limit | Value |
|-------|-------|
| Max build time | 30 minutes |
| GPU during build | **Not available** |
