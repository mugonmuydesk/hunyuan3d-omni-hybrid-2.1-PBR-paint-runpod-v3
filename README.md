# Hunyuan3D Omni-Hybrid RunPod Serverless Endpoint

Deploy Tencent's Hunyuan3D as a RunPod serverless endpoint for image-to-3D generation with PBR textures.

## Architecture

This endpoint combines three model components:

| Component | Model Size | VRAM | Description |
|-----------|------------|------|-------------|
| **Omni** | 3.3B | ~10GB | Quality shape generation (default), supports pose/skeleton |
| **Mini-Fast** | 0.6B | ~5GB | Fast shape generation (`fast_mode=true`) |
| **Paint 2.1** | 2B | ~21GB | PBR texture synthesis |

## Key Fix

This deployment uses a **prebuilt Linux wheel** for the `custom_rasterizer` CUDA extension, avoiding compilation issues during Docker builds (no GPU available at build time).

**Requirements:**
- Python 3.12 (must match the wheel)
- CUDA 12.4
- PyTorch 2.5.1

## Requirements

- **GPU:** 48GB VRAM recommended (A40, A100, RTX A6000)
  - Shape generation: ~10GB VRAM
  - Texture synthesis: ~21GB VRAM
- **RunPod account** with API key

## Deployment Steps

RunPod clones your GitHub repo and builds the Dockerfile on their infrastructure.

### 1. Push to GitHub

```bash
# Create a new repo or use existing one
git init
git add Dockerfile handler.py test_input.json README.md
git commit -m "Hunyuan3D-2.1 RunPod serverless worker"
git remote add origin https://github.com/YOUR_USER/hunyuan3d-runpod.git
git push -u origin main
```

### 2. Connect GitHub to RunPod

1. Go to [RunPod Settings](https://www.runpod.io/console/settings)
2. Find **GitHub** under Connections
3. Click **Connect** and authorize RunPod

### 3. Deploy from GitHub

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Select **Import Git Repository**
4. Choose your repository and branch (main)
5. Dockerfile path: `Dockerfile` (or leave default if in root)
6. Configure endpoint:
   - **GPU:** 48GB+ (A40, A100-40GB, RTX A6000)
   - **Container Disk:** 50GB
   - **Idle Timeout:** 60s+
   - **Max Workers:** Based on budget
7. Click **Deploy Endpoint**

### 4. Monitor Build

Build stages:
- **Pending** → **Building** → **Uploading** → **Testing** → **Completed**

Check logs if build fails. Common issues:
- Dockerfile syntax errors
- Network timeouts downloading dependencies

### 5. Update Deployments

Changes to your repo don't auto-deploy. To update:
1. Create a new **GitHub Release** in your repo
2. RunPod will rebuild and redeploy

## Build Constraints (RunPod)

| Limit | Value |
|-------|-------|
| Max build time | 160 minutes |
| Max image size | 80 GB |
| GPU during build | **Not available** (why prebuilt wheel is essential) |

## Local Testing (Optional)

```bash
# Test with RunPod SDK local server
python handler.py --rp_serve_api --rp_log_level DEBUG

# Send test request
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

## Test the Deployed Endpoint

```bash
# Submit a job
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_base64": "'$(base64 -w0 input.png)'",
      "generate_texture": true,
      "output_format": "glb"
    }
  }'

# Check status
curl https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## API Reference

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_base64` | string | required | Base64 encoded input image |
| `fast_mode` | boolean | `false` | Use Mini-Fast (0.6B) instead of Omni (3.3B) for faster inference |
| `skeleton_base64` | string | optional | Base64 encoded skeleton/pose file (Omni only) |
| `skeleton_data` | array | optional | Bone coordinates as JSON array (Omni only) |
| `generate_texture` | boolean | `true` | Generate PBR textures |
| `output_format` | string | `"glb"` | Output format: `"glb"` or `"obj"` |
| `num_views` | integer | `6` | Number of views for texture synthesis |
| `texture_resolution` | integer | `512` | Texture map resolution |

### Output

```json
{
  "download_url": "https://s3api-eur-is-1.runpod.io/...",
  "s3_key": "outputs/job-id/model.glb",
  "format": "glb",
  "textured": true,
  "size_mb": 12.34
}
```

### Download the Output

```python
import requests

# From the API response
result = response.json()["output"]

# Download from presigned URL
model_data = requests.get(result["download_url"]).content
with open(f"output.{result['format']}", "wb") as f:
    f.write(model_data)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_NUM_VIEW` | `6` | Default number of texture views |
| `TEXTURE_RESOLUTION` | `512` | Default texture resolution |
| `HF_HOME` | `/models` | HuggingFace cache directory |

## Payload Limits

RunPod has payload size limits:
- `/run` endpoint: **10 MB** maximum
- `/runsync` endpoint: **20 MB** maximum

For larger images, consider:
1. Resize images before encoding (512x512 or 1024x1024 is usually sufficient)
2. Use JPEG encoding instead of PNG for smaller payloads
3. Upload to cloud storage and pass URL instead (requires handler modification)

## Troubleshooting

### "No module named 'custom_rasterizer'"

The prebuilt wheel wasn't installed correctly. Check:
1. Python version is 3.12 (must match the wheel)
2. The wheel file is present in the build context
3. Run: `python -c "import custom_rasterizer; print('OK')"`

### Out of Memory

Reduce texture settings:
```json
{
  "input": {
    "image_base64": "...",
    "num_views": 4,
    "texture_resolution": 256
  }
}
```

Or disable textures entirely: `"generate_texture": false`

### Slow Cold Starts

Models are loaded from Network Volume on first request. Cold starts typically take 30-60s for model loading. To reduce:
1. Keep Idle Timeout higher (60s+)
2. Use Active Workers to maintain minimum warm workers
3. Consider RunPod's FlashBoot feature

### Job Timeout

Default timeout is 600s. For high-resolution textures, increase via RunPod endpoint settings.

## Files

- `Dockerfile` - Container build configuration
- `handler.py` - RunPod serverless handler
- `test_input.json` - Sample test input for local testing
- `README.md` - This documentation

## Credits

- [Tencent Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) - Shape generation (Omni/Mini-Fast)
- [Tencent Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) - PBR texture synthesis
