# Tencent is pleased to support the open source community by making Hunyuan3D available.
# Copyright (c) 2025 Tencent. All rights reserved.
#
# This file is modified to use ONNX-based RealESRGAN upscaler for Python 3.12 compatibility.
# Original file imported from basicsr and realesrgan which don't support Python 3.12.

import numpy as np
from PIL import Image
import sys
import os

# Add /app to path to find onnx_upscaler
sys.path.insert(0, '/app')


class imageSuperNet:
    """Image super-resolution using ONNX-based RealESRGAN upscaler.

    Drop-in replacement for the original basicsr/realesrgan implementation.
    """

    def __init__(self, config):
        from onnx_upscaler import RealESRGANUpscaler

        # ONNX model path (relative to /app)
        onnx_path = os.path.join('/app', 'weights', 'realesrgan_x4plus.onnx')
        if not os.path.exists(onnx_path):
            # Fallback to relative path
            onnx_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'weights', 'realesrgan_x4plus.onnx')

        self.upscaler = RealESRGANUpscaler(model_path=onnx_path)

    def __call__(self, image):
        """Upscale an image by 4x.

        Args:
            image: PIL Image or numpy array (HWC, RGB, uint8)

        Returns:
            PIL Image: Upscaled image
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # Convert RGB to BGR for RealESRGAN (expects BGR input)
        img_bgr = img_array[:, :, ::-1]

        # Upscale
        output_bgr, _ = self.upscaler.enhance(img_bgr)

        # Convert BGR back to RGB
        output_rgb = output_bgr[:, :, ::-1]

        return Image.fromarray(output_rgb)
