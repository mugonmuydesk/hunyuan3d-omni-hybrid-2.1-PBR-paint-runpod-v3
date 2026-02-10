#!/usr/bin/env python3
"""
Upload Hunyuan3D models to RunPod S3 Network Volume.

Downloads models from HuggingFace and uploads to S3 for use with RunPod Serverless.

Usage:
    python upload_models_to_s3.py [--local-cache /path/to/models]

Environment Variables (required):
    AWS_ACCESS_KEY_ID: S3 access key
    AWS_SECRET_ACCESS_KEY: S3 secret key

Environment Variables (optional):
    S3_BUCKET: Bucket name (default: df92r74hdc)
    S3_ENDPOINT: S3 endpoint (default: https://s3api-eur-is-1.runpod.io)
    S3_REGION: S3 region (default: eur-is-1)
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Check dependencies
try:
    import boto3
    from botocore.config import Config as BotoConfig
    from huggingface_hub import snapshot_download
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install boto3 huggingface_hub")
    sys.exit(1)


# S3 Configuration
S3_BUCKET = os.environ.get('S3_BUCKET', 'df92r74hdc')
S3_ENDPOINT = os.environ.get('S3_ENDPOINT', 'https://s3api-eur-is-1.runpod.io')
S3_REGION = os.environ.get('S3_REGION', 'eur-is-1')

# Model definitions
MODELS = {
    'Hunyuan3D-Omni': {
        'repo': 'tencent/Hunyuan3D-Omni',
        'allow_patterns': ['model/*', 'vae/*', 'cond_encoder/*', 'image_processor/*', 'scheduler/*', 'config.json'],
        'description': 'Quality shape generation (3.3B params, ~24GB)',
    },
    'Hunyuan3D-2mini': {
        'repo': 'tencent/Hunyuan3D-2mini',
        'allow_patterns': ['hunyuan3d-dit-v2-mini-fast/*', 'config.json'],
        'description': 'Fast shape generation (0.6B params, ~7GB)',
    },
    'Hunyuan3D-2.1': {
        'repo': 'tencent/Hunyuan3D-2.1',
        'allow_patterns': ['hunyuan3d-paintpbr-v2-1/*', 'hunyuan3d-vae-v2-1/*', 'hy3dpaint/*'],
        'description': 'PBR texture painting (~7GB)',
    },
}


def get_s3_client():
    """Create S3 client."""
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        region_name=S3_REGION,
        config=BotoConfig(signature_version='s3v4')
    )


def download_model(model_name: str, local_dir: Path) -> Path:
    """Download model from HuggingFace."""
    model_info = MODELS[model_name]
    local_path = local_dir / model_name

    print(f"\nDownloading {model_name}...")
    print(f"  Repo: {model_info['repo']}")
    print(f"  Description: {model_info['description']}")

    snapshot_download(
        model_info['repo'],
        local_dir=str(local_path),
        allow_patterns=model_info['allow_patterns'],
    )

    print(f"  Downloaded to: {local_path}")
    return local_path


def upload_file_to_s3(client, local_path: Path, s3_key: str):
    """Upload a single file to S3."""
    client.upload_file(str(local_path), S3_BUCKET, s3_key)
    return s3_key


def upload_directory_to_s3(client, local_dir: Path, s3_prefix: str, max_workers: int = 8):
    """Upload entire directory to S3 with parallel uploads."""
    files_to_upload = []

    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = Path(root) / file
            relative_path = local_path.relative_to(local_dir)
            s3_key = f"{s3_prefix}/{relative_path}"
            files_to_upload.append((local_path, s3_key))

    total = len(files_to_upload)
    print(f"\n  Uploading {total} files to s3://{S3_BUCKET}/{s3_prefix}/")

    uploaded = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(upload_file_to_s3, client, local_path, s3_key): s3_key
            for local_path, s3_key in files_to_upload
        }

        for future in as_completed(futures):
            s3_key = futures[future]
            try:
                future.result()
                uploaded += 1
                if uploaded % 10 == 0 or uploaded == total:
                    print(f"  Progress: {uploaded}/{total} files")
            except Exception as e:
                print(f"  ERROR uploading {s3_key}: {e}")

    print(f"  Completed: {uploaded}/{total} files uploaded")
    return uploaded


def main():
    parser = argparse.ArgumentParser(description='Upload Hunyuan3D models to RunPod S3')
    parser.add_argument('--local-cache', type=str, default='./hf_cache',
                        help='Local directory to cache downloaded models')
    parser.add_argument('--models', type=str, nargs='+', choices=list(MODELS.keys()),
                        default=list(MODELS.keys()),
                        help='Which models to upload (default: all)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download, use existing local cache')
    args = parser.parse_args()

    # Check credentials
    if not os.environ.get('AWS_ACCESS_KEY_ID') or not os.environ.get('AWS_SECRET_ACCESS_KEY'):
        print("ERROR: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables required")
        print("\nSet them with:")
        print("  export AWS_ACCESS_KEY_ID=user_37Gd9Du3X0vqIelXKB2lpKYW7RF")
        print("  export AWS_SECRET_ACCESS_KEY=rps_4HBXPTXG1DN0DGMXQOQ7THZ2Z603EGTAHN0JH2OV1176lt")
        sys.exit(1)

    local_cache = Path(args.local_cache)
    local_cache.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Hunyuan3D Model Uploader")
    print("=" * 60)
    print(f"S3 Bucket: {S3_BUCKET}")
    print(f"S3 Endpoint: {S3_ENDPOINT}")
    print(f"Local cache: {local_cache.absolute()}")
    print(f"Models to upload: {', '.join(args.models)}")

    client = get_s3_client()

    for model_name in args.models:
        print(f"\n{'=' * 60}")
        print(f"Processing: {model_name}")
        print("=" * 60)

        local_path = local_cache / model_name

        # Download if needed
        if not args.skip_download or not local_path.exists():
            local_path = download_model(model_name, local_cache)
        else:
            print(f"  Using cached: {local_path}")

        # Upload to S3
        s3_prefix = f"models/{model_name}"
        upload_directory_to_s3(client, local_path, s3_prefix)

    print(f"\n{'=' * 60}")
    print("DONE! Models uploaded to S3.")
    print("=" * 60)
    print("\nNetwork Volume path: /runpod-volume/models/")
    print("Models available:")
    for model_name in args.models:
        print(f"  - /runpod-volume/models/{model_name}/")


if __name__ == '__main__':
    main()
