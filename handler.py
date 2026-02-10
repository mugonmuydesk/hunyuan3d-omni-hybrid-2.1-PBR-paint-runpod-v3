"""
RunPod Serverless Handler for Hunyuan3D-2.1 (Image-to-3D)

Version: 1.6.0 - S3 upload for large outputs (bypass RunPod 20MB limit)

Generates high-fidelity 3D models with PBR materials from input images.

API:
  Input:
    - image_base64: Base64 encoded input image (required)
    - fast_mode: Use Mini-Fast (0.6B) instead of Omni (3.3B) for shape (default: false)
    - skeleton_base64: Base64 encoded bone coordinates file (Omni only, optional)
    - skeleton_data: JSON array of bone coords [[x1,y1,z1,x2,y2,z2],...] (Omni only, optional)
    - generate_texture: Whether to generate PBR textures (default: true)
    - output_format: 'glb' or 'obj' (default: 'glb')
    - num_views: Number of views for texture (default: from MAX_NUM_VIEW env)
    - texture_resolution: Texture resolution (default: from TEXTURE_RESOLUTION env)

  Skeleton Format (for pose-conditioned generation):
    - Each bone is 6 values: start_x, start_y, start_z, end_x, end_y, end_z
    - Follows PoseMaster bone definition (body + hand bones)
    - skeleton_base64: text file with M lines, 6 space-separated values each
    - skeleton_data: JSON array of M arrays with 6 values each

  Output:
    - download_url: Presigned S3 URL to download the model (valid 1 hour)
    - s3_key: S3 object key for direct access
    - format: Output format used
    - textured: Whether textures were generated
    - size_mb: File size in MB

Environment Variables (S3 - required):
  - S3_BUCKET: RunPod Network Volume bucket name
  - S3_ENDPOINT: RunPod S3 endpoint URL
  - S3_REGION: RunPod S3 region
  - AWS_ACCESS_KEY_ID: S3 access key
  - AWS_SECRET_ACCESS_KEY: S3 secret key

Environment Variables (VRAM Optimization):
  - MAX_NUM_VIEW: Max texture views (default: 3, lower = less VRAM)
  - TEXTURE_RESOLUTION: Texture resolution in pixels (default: 128)
  - ENABLE_CPU_OFFLOAD: Set to "1" to move models to CPU when idle (default: 0)
  - ENABLE_CACHE_CLEARING: Set to "1" to clear CUDA cache after each step (default: 0)
"""

import os
import sys
import base64
import tempfile
import traceback
import threading
import uuid
from pathlib import Path
from datetime import datetime

# Add Hunyuan3D paths
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/hy3dgen')
sys.path.insert(0, '/app/hy3dshape')
sys.path.insert(0, '/app/hy3dpaint')

import runpod
import torch
import numpy as np
import trimesh
import boto3
from botocore.config import Config as BotoConfig


# =============================================================================
# Model paths (RunPod Network Volume)
# =============================================================================
MODEL_BASE = os.environ.get('MODEL_BASE', '/runpod-volume/models')

# =============================================================================
# S3 Configuration (for large output files)
# =============================================================================
S3_BUCKET = os.environ.get('S3_BUCKET', 'df92r74hdc')
S3_ENDPOINT = os.environ.get('S3_ENDPOINT', 'https://s3api-eur-is-1.runpod.io')
S3_REGION = os.environ.get('S3_REGION', 'eur-is-1')

# Global S3 client (initialized lazily)
_s3_client = None


def get_s3_client():
    """Get or create S3 client."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT,
            region_name=S3_REGION,
            config=BotoConfig(signature_version='s3v4')
        )
    return _s3_client


def upload_to_s3(file_path: str, object_key: str) -> str:
    """Upload file to S3 and return presigned download URL."""
    client = get_s3_client()

    # Upload the file
    client.upload_file(file_path, S3_BUCKET, object_key)

    # Generate presigned URL (valid for 1 hour)
    presigned_url = client.generate_presigned_url(
        'get_object',
        Params={'Bucket': S3_BUCKET, 'Key': object_key},
        ExpiresIn=3600
    )

    return presigned_url

# =============================================================================
# FIX: Monkey-patch torch functions to handle numpy compatibility issues
#
# Two separate issues can occur with certain PyTorch/NumPy version combinations:
# 1. "expected np.ndarray (got numpy.ndarray)" - in torch.from_numpy()
# 2. "Could not infer dtype of numpy.int64" - in torch.tensor() with numpy scalars
#
# These patches pre-convert numpy types to avoid the errors entirely.
# Uses thread-local recursion guard to prevent infinite loops.
# =============================================================================

_patch_local = threading.local()

def _convert_numpy_to_native(data):
    """Convert numpy types to Python native types recursively."""
    if isinstance(data, np.ndarray):
        # Convert array to list of native types, then let torch handle it
        return data.tolist()
    elif isinstance(data, (np.integer,)):
        return int(data)
    elif isinstance(data, (np.floating,)):
        return float(data)
    elif isinstance(data, (np.bool_,)):
        return bool(data)
    elif isinstance(data, (list, tuple)):
        converted = [_convert_numpy_to_native(x) for x in data]
        return type(data)(converted)
    return data

_original_from_numpy = torch.from_numpy
_original_tensor = torch.tensor

def _patched_from_numpy(ndarray):
    """Wrapper for torch.from_numpy that handles type mismatch errors."""
    # Check recursion guard
    if getattr(_patch_local, 'in_from_numpy', False):
        return _original_from_numpy(ndarray)

    _patch_local.in_from_numpy = True
    try:
        return _original_from_numpy(ndarray)
    except TypeError as e:
        error_msg = str(e)
        if "expected np.ndarray" in error_msg or "Could not infer dtype" in error_msg:
            # Fallback: ensure contiguous array
            ndarray = np.ascontiguousarray(ndarray)
            return _original_from_numpy(ndarray)
        raise
    finally:
        _patch_local.in_from_numpy = False

def _patched_tensor(data, *args, **kwargs):
    """Wrapper for torch.tensor that handles numpy scalar type errors."""
    # Check recursion guard
    if getattr(_patch_local, 'in_tensor', False):
        return _original_tensor(data, *args, **kwargs)

    _patch_local.in_tensor = True
    try:
        # Pre-convert numpy scalars to native Python types
        if isinstance(data, (np.integer, np.floating, np.bool_)):
            data = _convert_numpy_to_native(data)
        return _original_tensor(data, *args, **kwargs)
    except (TypeError, RuntimeError) as e:
        error_msg = str(e)
        if "Could not infer dtype" in error_msg:
            # Full conversion as fallback
            converted_data = _convert_numpy_to_native(data)
            return _original_tensor(converted_data, *args, **kwargs)
        raise
    finally:
        _patch_local.in_tensor = False

torch.from_numpy = _patched_from_numpy
torch.tensor = _patched_tensor
print("Applied torch.from_numpy and torch.tensor monkey-patches for numpy compatibility (v1.3)")
# =============================================================================


# Global pipelines (loaded once on cold start)
shape_pipeline = None
paint_pipeline = None

# Hybrid mode pipelines (for quality/fast mode selection)
shape_pipeline_omni = None   # For quality mode (Omni 3.3B)
shape_pipeline_fast = None   # For fast mode (Mini-Fast 0.6B)


# =============================================================================
# VRAM Optimization toggles (all OFF by default for max speed on high-VRAM GPUs)
# =============================================================================
def should_enable_cpu_offload():
    """Check if CPU offload should be enabled (trades speed for VRAM)."""
    return os.environ.get('ENABLE_CPU_OFFLOAD', '0').lower() in ['1', 'true', 'yes']


def should_clear_cache():
    """Check if CUDA cache should be cleared after each generation."""
    return os.environ.get('ENABLE_CACHE_CLEARING', '0').lower() in ['1', 'true', 'yes']


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_omni_pipeline():
    """Load Omni pipeline for quality mode (3.3B params)

    Omni uses SiT (Scalable Interpolant Transformer) architecture with
    support for multi-modal control signals including skeleton/pose.
    """
    global shape_pipeline_omni
    from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
    shape_pipeline_omni = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(
        f'{MODEL_BASE}/Hunyuan3D-Omni',
        device=get_device()
    )


def load_fast_pipeline():
    """Load Mini-Fast pipeline for fast mode (0.6B params)"""
    global shape_pipeline_fast
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    shape_pipeline_fast = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        f'{MODEL_BASE}/Hunyuan3D-2mini',
        subfolder='hunyuan3d-dit-v2-mini-fast',
        device=get_device()
    )


def unload_shape_pipelines():
    """Free VRAM by unloading shape pipelines before texture generation"""
    global shape_pipeline_omni, shape_pipeline_fast
    if shape_pipeline_omni is not None:
        del shape_pipeline_omni
        shape_pipeline_omni = None
    if shape_pipeline_fast is not None:
        del shape_pipeline_fast
        shape_pipeline_fast = None
    torch.cuda.empty_cache()


def parse_skeleton_input(job_input, temp_path):
    """Parse skeleton/pose input from request.

    Supports two formats:
    - skeleton_base64: Base64 encoded text file with bone coordinates (M lines, 6 values each)
    - skeleton_data: JSON array of bone coordinates [[x1,y1,z1,x2,y2,z2], ...]

    Returns:
        torch.Tensor of shape [num_bones, 6] or None if no skeleton provided
    """
    import numpy as np

    skeleton_b64 = job_input.get("skeleton_base64")
    skeleton_data = job_input.get("skeleton_data")

    if skeleton_b64:
        # Decode base64 text file
        skeleton_text = base64.b64decode(skeleton_b64).decode('utf-8')
        skeleton_path = temp_path / "skeleton.txt"
        skeleton_path.write_text(skeleton_text)
        bone_points = torch.from_numpy(np.loadtxt(str(skeleton_path))).float()
        print(f"Loaded skeleton from base64: {bone_points.shape}")
        return bone_points
    elif skeleton_data:
        # Parse JSON array directly
        bone_points = torch.tensor(skeleton_data, dtype=torch.float32)
        print(f"Loaded skeleton from JSON: {bone_points.shape}")
        return bone_points

    return None


def generate_shape(job_input, image_path, temp_path=None):
    """Generate mesh using Omni (quality) or Mini-Fast (fast) pipeline.

    Args:
        job_input: Dict containing request parameters:
            - fast_mode: Use Mini-Fast instead of Omni (default: False)
            - skeleton_base64: Base64 encoded bone coordinates file (Omni only)
            - skeleton_data: JSON array of bone coordinates (Omni only)
        image_path: Path to the input image file
        temp_path: Temp directory for skeleton file (optional)

    Returns:
        Generated mesh object

    Note:
        Skeleton/pose input is only supported with Omni pipeline (quality mode).
        If skeleton is provided with fast_mode=True, fast_mode is ignored.
    """
    fast_mode = job_input.get("fast_mode", False)
    has_skeleton = "skeleton_base64" in job_input or "skeleton_data" in job_input

    # Skeleton input requires Omni (override fast_mode)
    if has_skeleton and fast_mode:
        print("WARNING: Skeleton input requires Omni pipeline. Ignoring fast_mode.")
        fast_mode = False

    if fast_mode:
        # Fast mode → Mini-Fast (0.6B, ~2x faster)
        print("Using Mini-Fast pipeline (fast mode)...")
        load_fast_pipeline()
        mesh = shape_pipeline_fast(image=str(image_path))[0]
    else:
        # Quality mode → Omni (3.3B, best quality)
        print("Using Omni pipeline (quality mode)...")
        load_omni_pipeline()

        # Check for skeleton/pose input
        bone_points = None
        if has_skeleton and temp_path:
            bone_points = parse_skeleton_input(job_input, temp_path)

        if bone_points is not None:
            # Pose-conditioned generation
            print(f"Generating with skeleton pose control ({bone_points.shape[0]} bones)...")
            mesh = shape_pipeline_omni(
                image=str(image_path),
                bone_points=bone_points.unsqueeze(0)  # Add batch dimension
            )[0]
        else:
            # Image-only generation
            mesh = shape_pipeline_omni(image=str(image_path))[0]

    return mesh


def load_pipelines():
    """Load the paint pipeline (shape pipeline loaded on-demand via generate_shape)."""
    global shape_pipeline, paint_pipeline

    device = get_device()
    print(f"Using device: {device}")

    # NOTE: Legacy shape_pipeline (Hunyuan3D-2.1) is no longer loaded here.
    # Shape generation now uses generate_shape() which loads Omni or Mini-Fast on-demand.

    if paint_pipeline is None:
        print("Loading paint pipeline...")
        try:
            from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
            config = Hunyuan3DPaintConfig(
                max_num_view=int(os.environ.get('MAX_NUM_VIEW', 6)),
                resolution=int(os.environ.get('TEXTURE_RESOLUTION', 512))
            )
            paint_pipeline = Hunyuan3DPaintPipeline(config)
            print("Paint pipeline loaded successfully.")
        except Exception as e:
            print(f"Error loading paint pipeline: {e}")
            traceback.print_exc()
            raise

    # Apply CPU offload if enabled (trades speed for VRAM)
    if should_enable_cpu_offload():
        print("VRAM optimization: Enabling CPU offload for paint pipeline...")
        try:
            if paint_pipeline is not None:
                paint_pipeline.enable_model_cpu_offload()
                print("  - Paint pipeline: CPU offload enabled")
        except Exception as e:
            print(f"  - Paint pipeline: CPU offload failed ({e})")

    return shape_pipeline, paint_pipeline


def save_base64_to_file(b64_data: str, output_path: str) -> str:
    """Decode base64 data and save to file."""
    # Handle data URI format
    if b64_data.startswith("data:"):
        b64_data = b64_data.split(",", 1)[1]

    decoded = base64.b64decode(b64_data)
    with open(output_path, "wb") as f:
        f.write(decoded)
    return output_path


def encode_file_to_base64(file_path: str) -> str:
    """Read file and encode to base64."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job: dict) -> dict:
    """
    RunPod serverless handler for Hunyuan3D-2.1.
    """
    job_input = job.get("input")

    # Health check - returns immediately without loading model
    if job_input == "health_check" or (isinstance(job_input, dict) and job_input.get("health_check")):
        return {
            "status": "healthy",
            "model_base": MODEL_BASE,
            "models_available": {
                "omni": os.path.exists(f"{MODEL_BASE}/Hunyuan3D-Omni"),
                "mini_fast": os.path.exists(f"{MODEL_BASE}/Hunyuan3D-2mini"),
                "paint_pbr": os.path.exists(f"{MODEL_BASE}/Hunyuan3D-2.1"),
            },
            "message": "Handler ready."
        }

    if not isinstance(job_input, dict):
        return {"error": "Invalid request: missing 'input' field"}

    # Validate required inputs
    if "image_base64" not in job_input:
        return {"error": "Missing required field: image_base64"}

    # Extract parameters
    image_b64 = job_input["image_base64"]
    generate_texture = job_input.get("generate_texture", True)
    output_format = job_input.get("output_format", "glb").lower()
    num_views = job_input.get("num_views", int(os.environ.get('MAX_NUM_VIEW', 6)))
    texture_resolution = job_input.get("texture_resolution", int(os.environ.get('TEXTURE_RESOLUTION', 512)))

    if output_format not in ["glb", "obj"]:
        return {"error": f"Invalid output_format: {output_format}. Use 'glb' or 'obj'."}

    # Load paint pipeline (shape pipeline loaded on-demand in generate_shape)
    try:
        _, paint_pipe = load_pipelines()
    except Exception as e:
        return {"error": f"Failed to load pipelines: {str(e)}"}

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save input image
        image_ext = ".png"
        if "image/jpeg" in image_b64[:50] or "image/jpg" in image_b64[:50]:
            image_ext = ".jpg"
        image_path = temp_path / f"input{image_ext}"

        try:
            save_base64_to_file(image_b64, str(image_path))
            print(f"Input image saved: {image_path}")
        except Exception as e:
            return {"error": f"Failed to decode input image: {str(e)}"}

        # Generate shape (untextured mesh) using hybrid routing
        try:
            fast_mode = job_input.get("fast_mode", False)
            mode_str = "fast" if fast_mode else "quality"
            runpod.serverless.progress_update(job, f"Generating 3D shape ({mode_str} mode)...")
            print("Generating 3D shape from image...")
            mesh = generate_shape(job_input, image_path, temp_path)
            print("Shape generation complete.")
            if should_clear_cache():
                torch.cuda.empty_cache()
        except Exception as e:
            if should_clear_cache():
                torch.cuda.empty_cache()
            traceback.print_exc()
            return {"error": f"Shape generation failed: {str(e)}"}

        # Free VRAM from shape pipeline before loading texture pipeline
        print("Unloading shape pipelines to free VRAM...")
        unload_shape_pipelines()

        # Generate texture if requested
        if generate_texture:
            try:
                runpod.serverless.progress_update(job, f"Generating textures ({num_views} views)...")
                print(f"Generating textures ({num_views} views, {texture_resolution}px)...")

                # Save untextured mesh to temp file (paint pipeline expects OBJ path for remeshing)
                untextured_mesh_path = temp_path / "untextured.obj"
                mesh.export(str(untextured_mesh_path), file_type='obj')
                print(f"Saved untextured mesh: {untextured_mesh_path}")

                # Update paint pipeline config
                paint_pipe.config.max_num_view = num_views
                paint_pipe.config.resolution = texture_resolution

                # Paint pipeline returns file path string, not Trimesh object
                output_mesh_path = paint_pipe(str(untextured_mesh_path), image_path=str(image_path))
                print(f"Texture generation complete: {output_mesh_path}")

                # Load the textured mesh from the output path
                mesh = trimesh.load(output_mesh_path)
                if should_clear_cache():
                    torch.cuda.empty_cache()
            except Exception as e:
                if should_clear_cache():
                    torch.cuda.empty_cache()
                traceback.print_exc()
                return {"error": f"Texture generation failed: {str(e)}"}

        runpod.serverless.progress_update(job, "Exporting mesh...")
        # Export mesh
        output_path = temp_path / f"output.{output_format}"
        try:
            if output_format == "glb":
                mesh.export(str(output_path))
            else:
                mesh.export(str(output_path), file_type='obj')
            print(f"Exported to {output_format.upper()}: {output_path}")
        except Exception as e:
            traceback.print_exc()
            return {"error": f"Failed to export mesh: {str(e)}"}

        # Upload to S3
        try:
            file_size_mb = os.path.getsize(str(output_path)) / (1024 * 1024)
            print(f"Output size: {file_size_mb:.2f} MB")

            # Generate unique object key with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_id = job.get("id", uuid.uuid4().hex[:8])
            object_key = f"outputs/{timestamp}_{job_id}.{output_format}"

            runpod.serverless.progress_update(job, "Uploading to S3...")
            print(f"Uploading to S3: {object_key}")
            download_url = upload_to_s3(str(output_path), object_key)
            print(f"Upload complete. URL valid for 1 hour.")

        except Exception as e:
            traceback.print_exc()
            return {"error": f"Failed to upload to S3: {str(e)}"}

        return {
            "download_url": download_url,
            "s3_key": object_key,
            "format": output_format,
            "textured": generate_texture,
            "size_mb": round(file_size_mb, 2)
        }


# Entry point
if __name__ == "__main__":
    # Check for RunPod flags (--rp_serve_api, --rp_log_level, etc.)
    has_runpod_flags = any(arg.startswith("--rp_") for arg in sys.argv[1:])

    # Local file test: first arg is a file path (no -- prefix, not a RunPod mode)
    local_test_file = None
    if not has_runpod_flags and len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        local_test_file = sys.argv[1]

    if local_test_file:
        # Local testing: python handler.py <image.png>
        with open(local_test_file, "rb") as f:
            test_image = base64.b64encode(f.read()).decode("utf-8")

        test_job = {
            "input": {
                "image_base64": test_image,
                "generate_texture": True,
                "output_format": "glb"
            }
        }
        result = handler(test_job)
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Success! Format: {result['format']}, Textured: {result['textured']}, Size: {result['size_mb']} MB")
            print(f"Download URL: {result['download_url']}")
            print(f"S3 Key: {result['s3_key']}")
            # Download from S3 URL
            import requests
            response = requests.get(result['download_url'])
            with open("test_output.glb", "wb") as f:
                f.write(response.content)
            print("Downloaded and saved to test_output.glb")
    else:
        # Production/API mode: RunPod serverless
        # Supports --rp_serve_api for local HTTP server
        runpod.serverless.start({"handler": handler})
