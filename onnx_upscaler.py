"""
ONNX-based RealESRGAN upscaler - drop-in replacement for realesrgan package.

Avoids basicsr dependency issues on Python 3.12 by using pre-exported ONNX model.
"""

import numpy as np
from PIL import Image
import os
from typing import Optional, Union

# Lazy import to avoid loading at module level
_ort_session = None


def get_onnx_session(model_path: str, use_gpu: bool = True):
    """Load ONNX model with appropriate execution provider."""
    import onnxruntime as ort

    providers = []
    if use_gpu:
        available = ort.get_available_providers()
        # Skip TensorRT - causes CUDA init failures on some GPU configurations
        # TensorRT provider tries to init CUDA immediately when added to list
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')

    session = ort.InferenceSession(model_path, providers=providers)
    actual_provider = session.get_providers()[0]
    print(f"[ONNX Upscaler] Loaded model with provider: {actual_provider}")
    return session


def upscale_image(
    image: Union[Image.Image, np.ndarray],
    model_path: str = None,
    scale: int = 4,
    tile_size: int = 512,
    tile_overlap: int = 32,
    use_gpu: bool = True
) -> Image.Image:
    """
    Upscale an image using RealESRGAN ONNX model.

    Args:
        image: Input PIL Image or numpy array (HWC, RGB, uint8)
        model_path: Path to ONNX model file
        scale: Upscaling factor (default 4 for RealESRGAN_x4plus)
        tile_size: Process image in tiles to manage memory (0 = no tiling)
        tile_overlap: Overlap between tiles to avoid seams
        use_gpu: Use CUDA if available

    Returns:
        Upscaled PIL Image
    """
    global _ort_session

    # Default model path
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "weights", "realesrgan_x4plus.onnx")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    # Load session (cached)
    if _ort_session is None:
        _ort_session = get_onnx_session(model_path, use_gpu)

    # Convert to numpy if PIL Image
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert('RGB'))
    else:
        img_np = image

    h, w = img_np.shape[:2]

    # Process with tiling for large images
    if tile_size > 0 and (h > tile_size or w > tile_size):
        output = _upscale_tiled(img_np, _ort_session, scale, tile_size, tile_overlap)
    else:
        output = _upscale_single(img_np, _ort_session)

    return Image.fromarray(output)


def _upscale_single(img_np: np.ndarray, session) -> np.ndarray:
    """Upscale a single image without tiling."""
    # Preprocess: HWC uint8 -> NCHW float32 [0,1]
    img = img_np.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, 0)  # CHW -> NCHW

    # Run inference
    output = session.run(None, {'input': img})[0]

    # Postprocess: NCHW float32 -> HWC uint8
    output = output[0]  # Remove batch dim
    output = np.transpose(output, (1, 2, 0))  # CHW -> HWC
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

    return output


def _upscale_tiled(
    img_np: np.ndarray,
    session,
    scale: int,
    tile_size: int,
    tile_overlap: int
) -> np.ndarray:
    """Upscale image using tiles to manage memory."""
    h, w, c = img_np.shape
    out_h, out_w = h * scale, w * scale
    output = np.zeros((out_h, out_w, c), dtype=np.float32)
    weights = np.zeros((out_h, out_w, c), dtype=np.float32)

    # Calculate tile positions
    step = tile_size - tile_overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            # Extract tile with bounds checking
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = img_np[y_start:y_end, x_start:x_end]

            # Upscale tile
            tile_out = _upscale_single(tile, session).astype(np.float32)

            # Place in output with blending weights
            out_y_start = y_start * scale
            out_y_end = y_end * scale
            out_x_start = x_start * scale
            out_x_end = x_end * scale

            output[out_y_start:out_y_end, out_x_start:out_x_end] += tile_out
            weights[out_y_start:out_y_end, out_x_start:out_x_end] += 1.0

    # Average overlapping regions
    output = output / np.maximum(weights, 1e-8)
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output


# Convenience class matching realesrgan API style
class RealESRGANUpscaler:
    """Drop-in replacement for RealESRGANer class."""

    def __init__(
        self,
        model_path: str = None,
        scale: int = 4,
        tile_size: int = 512,
        tile_overlap: int = 32,
        use_gpu: bool = True
    ):
        self.model_path = model_path
        self.scale = scale
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.use_gpu = use_gpu

        # Pre-load session
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "weights", "realesrgan_x4plus.onnx")
        self.session = get_onnx_session(model_path, use_gpu)

    def enhance(self, img: np.ndarray, outscale: int = None) -> tuple:
        """
        Enhance image (matches RealESRGANer.enhance signature).

        Args:
            img: Input image as numpy array (HWC, BGR, uint8) - note: BGR not RGB!
            outscale: Output scale (ignored, uses model's native scale)

        Returns:
            Tuple of (output_image, None) - second element for API compatibility
        """
        # RealESRGAN uses BGR, convert to RGB
        img_rgb = img[:, :, ::-1]

        result = upscale_image(
            img_rgb,
            model_path=self.model_path,
            scale=self.scale,
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            use_gpu=self.use_gpu
        )

        # Convert back to BGR numpy
        result_bgr = np.array(result)[:, :, ::-1]
        return result_bgr, None
