#!/usr/bin/env python3
"""
Client for Hunyuan3D-2.1-ONNX RunPod Serverless Endpoint

Converts images to 3D models (GLB/OBJ) with optional PBR textures.

Usage:
    python hunyuan3d_client.py input.png -o output.glb
    python hunyuan3d_client.py input.jpg --no-texture --format obj

Environment:
    RUNPOD_API_KEY: Your RunPod API key (required)
    RUNPOD_ENDPOINT_ID: Endpoint ID (default: ld5fnxyim0okwo)
"""

import os
import sys
import base64
import time
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# Configuration
DEFAULT_ENDPOINT_ID = "ld5fnxyim0okwo"
API_BASE = "https://api.runpod.ai/v2"
POLL_INTERVAL = 5  # seconds
MAX_WAIT_TIME = 600  # 10 minutes


def get_api_key() -> str:
    """Get API key from environment."""
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("Error: RUNPOD_API_KEY environment variable not set", file=sys.stderr)
        print("Set it with: export RUNPOD_API_KEY=your_key_here", file=sys.stderr)
        sys.exit(1)
    return key


def get_endpoint_id() -> str:
    """Get endpoint ID from environment or use default."""
    return os.environ.get("RUNPOD_ENDPOINT_ID", DEFAULT_ENDPOINT_ID)


def encode_image(image_path: Path) -> str:
    """Read and base64 encode an image file."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decode_model(model_b64: str) -> bytes:
    """Decode base64 model data."""
    return base64.b64decode(model_b64)


def api_request(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    """Make an API request to RunPod."""
    api_key = get_api_key()
    endpoint_id = get_endpoint_id()

    url = f"{API_BASE}/{endpoint_id}/{endpoint}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body = json.dumps(data).encode("utf-8") if data else None
    req = Request(url, data=body, headers=headers, method=method)

    try:
        with urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        print(f"HTTP Error {e.code}: {e.reason}", file=sys.stderr)
        if error_body:
            print(f"Response: {error_body}", file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print(f"Network error: {e.reason}", file=sys.stderr)
        sys.exit(1)


def submit_job(
    image_path: Path,
    generate_texture: bool = True,
    output_format: str = "glb",
    num_views: int = 6,
    texture_resolution: int = 512,
) -> str:
    """Submit a job and return the job ID."""
    image_b64 = encode_image(image_path)

    # Check payload size (RunPod limit: 10MB for /run)
    payload_size_mb = len(image_b64) / (1024 * 1024)
    if payload_size_mb > 9:
        print(f"Warning: Image payload is {payload_size_mb:.1f}MB (limit: 10MB)", file=sys.stderr)
        print("Consider resizing the image to reduce size.", file=sys.stderr)

    payload = {
        "input": {
            "image_base64": image_b64,
            "generate_texture": generate_texture,
            "output_format": output_format,
            "num_views": num_views,
            "texture_resolution": texture_resolution,
        }
    }

    print(f"Submitting job for: {image_path.name}")
    print(f"  Texture: {generate_texture}, Format: {output_format}")
    if generate_texture:
        print(f"  Views: {num_views}, Resolution: {texture_resolution}")

    result = api_request("run", method="POST", data=payload)
    job_id = result.get("id")

    if not job_id:
        print(f"Error: No job ID in response: {result}", file=sys.stderr)
        sys.exit(1)

    print(f"Job submitted: {job_id}")
    return job_id


def check_status(job_id: str) -> dict:
    """Check the status of a job."""
    return api_request(f"status/{job_id}")


def wait_for_completion(job_id: str) -> dict:
    """Poll until job completes or fails."""
    start_time = time.time()
    last_status = None

    while True:
        elapsed = time.time() - start_time
        if elapsed > MAX_WAIT_TIME:
            print(f"\nTimeout after {MAX_WAIT_TIME}s", file=sys.stderr)
            sys.exit(1)

        result = check_status(job_id)
        status = result.get("status")

        if status != last_status:
            print(f"Status: {status}")
            last_status = status

        if status == "COMPLETED":
            return result
        elif status == "FAILED":
            error = result.get("error", "Unknown error")
            print(f"Job failed: {error}", file=sys.stderr)
            sys.exit(1)
        elif status == "CANCELLED":
            print("Job was cancelled", file=sys.stderr)
            sys.exit(1)
        elif status in ("IN_QUEUE", "IN_PROGRESS"):
            # Show progress indicator
            print(f"  Waiting... ({int(elapsed)}s elapsed)", end="\r")
            time.sleep(POLL_INTERVAL)
        else:
            print(f"Unknown status: {status}", file=sys.stderr)
            time.sleep(POLL_INTERVAL)


def save_model(output: dict, output_path: Path) -> None:
    """Save the model from the API response."""
    model_b64 = output.get("model")
    if not model_b64:
        print(f"Error: No model in output: {output}", file=sys.stderr)
        sys.exit(1)

    model_data = decode_model(model_b64)

    with open(output_path, "wb") as f:
        f.write(model_data)

    size_mb = len(model_data) / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert images to 3D models using Hunyuan3D-2.1-ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python hunyuan3d_client.py photo.png
    python hunyuan3d_client.py photo.jpg -o model.glb
    python hunyuan3d_client.py photo.png --no-texture --format obj
    python hunyuan3d_client.py photo.png --views 4 --resolution 256

Environment variables:
    RUNPOD_API_KEY       Your RunPod API key (required)
    RUNPOD_ENDPOINT_ID   Endpoint ID (default: ld5fnxyim0okwo)
""",
    )

    parser.add_argument("image", type=Path, help="Input image file (PNG, JPG, etc.)")
    parser.add_argument("-o", "--output", type=Path, help="Output file path (default: input_name.glb)")
    parser.add_argument("--format", choices=["glb", "obj"], default="glb", help="Output format (default: glb)")
    parser.add_argument("--no-texture", action="store_true", help="Skip texture generation (faster)")
    parser.add_argument("--views", type=int, default=6, help="Number of views for texture (default: 6)")
    parser.add_argument("--resolution", type=int, default=512, help="Texture resolution (default: 512)")
    parser.add_argument("--status", metavar="JOB_ID", help="Check status of existing job")
    parser.add_argument("--sync", action="store_true", help="Use synchronous endpoint (blocks until done)")

    args = parser.parse_args()

    # Status check mode
    if args.status:
        result = check_status(args.status)
        print(json.dumps(result, indent=2))
        return

    # Validate input
    if not args.image.exists():
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.image.with_suffix(f".{args.format}")

    # Submit job
    job_id = submit_job(
        image_path=args.image,
        generate_texture=not args.no_texture,
        output_format=args.format,
        num_views=args.views,
        texture_resolution=args.resolution,
    )

    # Wait for completion
    print()
    result = wait_for_completion(job_id)

    # Save output
    output = result.get("output", {})
    save_model(output, output_path)

    # Print summary
    print()
    print("Done!")
    print(f"  Format: {output.get('format', args.format)}")
    print(f"  Textured: {output.get('textured', not args.no_texture)}")


if __name__ == "__main__":
    main()
