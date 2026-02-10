# Hunyuan3D Omni Hybrid Pipeline

## Overview

RunPod serverless endpoint for hybrid 3D generation with mode selection:
- **Quality mode** (default): Hunyuan3D-Omni (3.3B params, SiT architecture)
- **Fast mode** (`--fast`): Hunyuan3D-DiT-v2-mini-Fast (0.6B params)
- **Pose-conditioned** (`--skeleton`): Omni with skeleton/bone input
- **Textures**: Hunyuan3D-2.1 PaintPBR (PBR materials)

- **GitHub Repo**: `mugonmuydesk/hunyuan3d-omni-hybrid-2.1-PBR-paint`
- **Local Path**: `C:\dev\hunyuan3d-omni-hybrid-2.1-PBR-paint`
- **GPU Requirements**: 24GB minimum, 48GB recommended

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
export RUNPOD_ENDPOINT_ID="..."

# Quality mode (Omni - best quality)
python hunyuan3d_client.py image.png -o output.glb

# Fast mode (Mini-Fast - ~2x faster)
python hunyuan3d_client.py image.png --fast -o output.glb

# Pose-conditioned (skeleton file)
python hunyuan3d_client.py image.png --skeleton pose.txt -o output.glb

# Shape only (skip textures)
python hunyuan3d_client.py image.png --no-texture -o output.glb
```

## Local Development

### Build in WSL2 (native Linux fs for speed)

```bash
# Copy to Linux fs
cp -r /mnt/c/dev/hunyuan3d-omni-hybrid-2.1-PBR-paint ~/hunyuan3d-omni-hybrid-2.1-PBR-paint
cd ~/hunyuan3d-omni-hybrid-2.1-PBR-paint

# Build
docker build -t hunyuan3d-omni-hybrid:test .
```

### Run Locally

```bash
docker run --gpus all -p 8000:8000 \
  -e MAX_NUM_VIEW=6 \
  -e TEXTURE_RESOLUTION=512 \
  hunyuan3d-omni-hybrid:test python -u handler.py --rp_serve_api --rp_api_host 0.0.0.0
```

## Deployment

Push to GitHub - RunPod pulls from GitHub repo:

```bash
git add -A && git commit -m "Fix: description"
git push origin master
# In RunPod console: trigger rebuild from GitHub
```

## Model Downloads (Selective)

| Model | Repo | allow_patterns | Est. Size |
|-------|------|----------------|-----------|
| Omni | `tencent/Hunyuan3D-Omni` | `hunyuan3d-omni-dit/*` | ~8 GB |
| Mini-Fast | `tencent/Hunyuan3D-2mini` | `hunyuan3d-dit-v2-mini-fast/*` | ~2 GB |
| PaintPBR | `tencent/Hunyuan3D-2.1` | `hunyuan3d-paintpbr-v2-1/*`, etc. | ~7 GB |

## Pipeline Classes

- **Omni**: `Hunyuan3DOmniSiTFlowMatchingPipeline` (SiT architecture, supports skeleton)
- **Mini-Fast**: `Hunyuan3DDiTFlowMatchingPipeline` (DiT architecture)
- **PaintPBR**: `Hunyuan3DPaintPipeline`

## File Structure

```
handler.py              # RunPod handler with hybrid routing
test_handler_mock.py    # Mock tests for routing logic
Dockerfile              # Production build with selective model downloads
hunyuan3d_client.py     # CLI client (in C:\dev\runpod\)
```

## Key Functions

- `generate_shape(job_input, image_path, temp_path)` - Routes to Omni or Mini-Fast
- `parse_skeleton_input(job_input, temp_path)` - Parses skeleton from base64 or JSON
- `load_omni_pipeline()` / `load_fast_pipeline()` - Lazy pipeline loaders
- `unload_shape_pipelines()` - VRAM cleanup between stages
