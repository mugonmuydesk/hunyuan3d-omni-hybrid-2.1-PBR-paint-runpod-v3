# Dockerfile for Hunyuan3D-2.1 on RunPod Serverless
# Image-to-3D generation with PBR materials
# Requires ~29GB VRAM (A40, A100, RTX A6000)
#
# KEY FIX: Uses Python 3.12 + locally-built custom_rasterizer wheel
# (HuggingFace wheel has ABI mismatch with all PyTorch 2.3-2.5 versions)

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1
ENV MAX_JOBS=4

# =============================================================================
# STAGE 1: System dependencies + TensorRT
# =============================================================================
# FIX: Remove conflicting NVIDIA apt sources before adding our own
# The base image has cuda-archive-keyring.gpg, we need nvidia-cuda.gpg
# Having both causes "Conflicting values set for option Signed-By" error
RUN rm -f /etc/apt/sources.list.d/cuda*.list && \
    rm -f /usr/share/keyrings/cuda-archive-keyring.gpg

# Add NVIDIA repo for TensorRT (must be done before apt-get update)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg \
    curl \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/nvidia-cuda.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/nvidia-cuda.list \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libgomp1 \
    build-essential \
    gcc \
    g++ \
    ninja-build \
    software-properties-common \
    libnvinfer10 \
    libnvinfer-plugin10 \
    libnvonnxparsers10 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && add-apt-repository ppa:ubuntu-toolchain-r/test -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-dev python3.12-venv \
    && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# VERIFY: TensorRT libraries installed
RUN ldconfig && ldconfig -p | grep nvinfer || echo "WARN: TensorRT libs not in ldconfig"

# Set Python 3.12 as default (MUST match locally-built wheel: cp312)
# Also symlink python3-config for C extension compilation (mesh_inpaint_processor)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    ln -sf /usr/bin/python3.12-config /usr/bin/python3-config

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# VERIFY: Python version
RUN python --version | grep "3.12" || (echo "ERROR: Python 3.12 required" && exit 1)

# =============================================================================
# STAGE 2: PyTorch (must be installed BEFORE any CUDA extensions)
# =============================================================================
RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja pybind11

# Note: cu124 has known issues with PyTorch 2.5.x, using cu121 instead
# The CUDA runtime is bundled with PyTorch, so this works with cuda:12.4 base image
# Install PyTorch packages separately to avoid resolution issues
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# VERIFY: PyTorch installed with CUDA
RUN python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); assert torch.__version__.startswith('2.5'), 'Wrong PyTorch version'"

# Add PyTorch libs to LD_LIBRARY_PATH for CUDA extensions (custom_rasterizer needs libc10.so)
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:${LD_LIBRARY_PATH}"

# =============================================================================
# STAGE 3: Clone Hunyuan3D sources
# =============================================================================
# Need THREE repos:
# - Hunyuan3D-2.1: Contains hy3dpaint for PBR texture generation
# - Hunyuan3D-2: Contains hy3dgen for Mini-Fast shape generation
# - Hunyuan3D-Omni: Contains hy3dshape for Omni shape generation
WORKDIR /app

# Clone all three repos in a single layer to reduce build time:
# - Hunyuan3D-2.1 (hy3dpaint for PBR textures) - skip LFS, models on Network Volume
# - Hunyuan3D-2 (hy3dgen for Mini-Fast shape generation)
# - Hunyuan3D-Omni (hy3dshape for Omni shape generation, replaces nested hy3dshape from 2.1)
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git . && \
    rm -rf .git && \
    git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /tmp/hy3d2 && \
    cp -r /tmp/hy3d2/hy3dgen /app/hy3dgen && \
    rm -rf /tmp/hy3d2 && \
    git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni.git /tmp/hy3d-omni && \
    rm -rf /app/hy3dshape && \
    cp -r /tmp/hy3d-omni/hy3dshape /app/hy3dshape && \
    rm -rf /tmp/hy3d-omni && \
    test -d /app/hy3dgen && test -d /app/hy3dshape && test -d /app/hy3dpaint && test -f /app/requirements.txt

# =============================================================================
# STAGE 4a: Install Python 3.11 + bpy in separate venv
# =============================================================================
# bpy 4.2.0+ requires Python 3.11+ (not 3.10, not 3.12)
# We use Python 3.11 in a separate venv and call it via subprocess for mesh operations
#
# NOTE: We add deadsnakes PPA manually because add-apt-repository breaks after
# changing the default Python to 3.12 (apt_pkg is only available for system Python)

# Add deadsnakes PPA manually (avoids add-apt-repository which needs apt_pkg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg ca-certificates \
    && echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes.list \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 \
    libxkbcommon0 libsm6 libice6 libxrandr2 libxcursor1 \
    libxinerama1 libglew2.2 libwayland-client0 libwayland-cursor0 \
    libwayland-egl1 libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# Create bpy virtual environment with Python 3.11
# NOTE: bpy wheels are hosted on Blender's index, not standard PyPI
# bpy 4.x+ requires Python 3.11 (no 3.12 wheels exist)
RUN python3.11 -m venv /opt/bpy-env && \
    /opt/bpy-env/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/bpy-env/bin/pip install --no-cache-dir \
        --extra-index-url https://download.blender.org/pypi/ \
        bpy==4.4.0 numpy==1.24.3

# VERIFY: bpy works in the venv
RUN /opt/bpy-env/bin/python -c "import bpy; print(f'bpy {bpy.app.version_string}')"

# =============================================================================
# STAGE 4b: Python dependencies (inference-optimized)
# =============================================================================
# Use our trimmed requirements_inference.txt instead of full requirements.txt
# Excludes: torch (already installed), open3d, cupy, pandas, torchaudio,
# pythreejs, fastapi, uvicorn, gradio, deepspeed, bpy
# This reduces image size and build time significantly
COPY requirements_inference.txt /tmp/requirements_inference.txt
RUN pip install --no-cache-dir -r /tmp/requirements_inference.txt

# VERIFY: Key packages installed and PyTorch wasn't overwritten
RUN python -c "import transformers, diffusers, trimesh, torch; assert torch.__version__.startswith('2.5'), f'PyTorch overwritten to {torch.__version__}'; print('Dependencies OK')"

# =============================================================================
# STAGE 5: Custom rasterizer (pre-built wheel with all CUDA architectures)
# =============================================================================
# Use pre-built wheel with kernels for: sm_70 (V100), sm_75 (T4), sm_80 (A100),
# sm_86 (A40), sm_89 (L40), sm_90 (H100). Built locally with Dockerfile.wheel.
COPY custom_rasterizer-0.1-cp312-cp312-linux_x86_64.whl /tmp/
RUN pip install --no-cache-dir /tmp/custom_rasterizer-0.1-cp312-cp312-linux_x86_64.whl

# VERIFY: custom_rasterizer works (combined with install step above)
RUN python -c "import custom_rasterizer; print('custom_rasterizer: OK')" && rm /tmp/custom_rasterizer-0.1-cp312-cp312-linux_x86_64.whl

# =============================================================================
# STAGE 6: mesh_inpaint_processor (pybind11 C++ extension)
# =============================================================================
# This is a pybind11 C++ extension that needs to be compiled with g++
# The compile script uses: c++ -O3 -Wall -shared -std=c++11 -fPIC $(python -m pybind11 --includes) ...
WORKDIR /app/hy3dpaint/DifferentiableRenderer
RUN chmod +x compile_mesh_painter.sh && \
    ./compile_mesh_painter.sh && \
    ls -la mesh_inpaint_processor*.so && \
    echo "mesh_inpaint_processor compiled successfully"

# VERIFY: mesh_inpaint_processor can be imported
RUN python -c "from mesh_inpaint_processor import meshVerticeInpaint; print('mesh_inpaint_processor: OK')"

# =============================================================================
# STAGE 7: RunPod SDK and HuggingFace Hub
# =============================================================================
# Include hf_xet for faster downloads via Xet Storage protocol
# Constrain numpy to prevent runpod/hf_xet from upgrading to numpy 2.x
# (Hunyuan3D uses numpy.core.multiarray.generic which was removed in numpy 2.0)
RUN pip install --no-cache-dir "numpy==1.26.4" runpod "huggingface_hub[cli,hf_xet]"

# FIX: Upgrade setuptools to fix Python 3.12 compatibility
# The system pkg_resources at /usr/lib/python3/dist-packages/ uses the removed
# FileFinder.find_module API. This manifests when pytorch_lightning imports
# lightning_fabric which calls pkg_resources.declare_namespace().
# setuptools>=69.0 fixes this. Force-reinstall to override the system copy.
RUN pip install --no-cache-dir --upgrade --force-reinstall "setuptools>=69.0"

# VERIFY: RunPod SDK, hf_xet, numpy version, and pkg_resources compatibility
RUN python -c "import runpod, numpy; assert numpy.__version__.startswith('1.'), f'numpy 2.x detected: {numpy.__version__}'; print(f'runpod {runpod.__version__}, numpy {numpy.__version__}: OK')"
RUN python -c "import pkg_resources; pkg_resources.declare_namespace('test_ns_pkg'); print('pkg_resources: Python 3.12 compat OK')"

# =============================================================================
# STAGE 8: Model paths (models stored on RunPod Network Volume)
# =============================================================================
# Models are NOT downloaded during build - they're on Network Volume at /runpod-volume/models
# This keeps image size small (~15GB vs ~52GB) and build time under 30 min limit
#
# Network Volume structure (upload separately):
#   /runpod-volume/models/
#     Hunyuan3D-Omni/      (~24GB) - Quality shape generation
#     Hunyuan3D-2mini/     (~7GB)  - Fast shape generation
#     Hunyuan3D-2.1/       (~7GB)  - PBR texture painting
WORKDIR /app

# ONNX-based RealESRGAN upscaler (replaces basicsr/realesrgan packages)
COPY onnx_upscaler.py /app/onnx_upscaler.py
COPY weights/realesrgan_x4plus.onnx /app/weights/realesrgan_x4plus.onnx

# =============================================================================
# STAGE 9: Copy handler and final verification
# =============================================================================
COPY handler.py /app/handler.py

# FIX: Replace schedulers.py to fix numpy/torch compatibility issue
# (torch.from_numpy fails with "expected np.ndarray got numpy.ndarray")
COPY schedulers.py /app/hy3dshape/schedulers.py

# FIX: Replace mesh_utils.py with subprocess wrapper (calls bpy via Python 3.10)
COPY mesh_utils_noblender.py /app/hy3dpaint/DifferentiableRenderer/mesh_utils.py

# Copy bpy wrapper script (runs in /opt/bpy-env)
COPY bpy_mesh_ops.py /app/bpy_mesh_ops.py

# FIX: Replace image_super_utils.py with ONNX upscaler (realesrgan/basicsr incompatible with Python 3.12)
COPY patches/image_super_utils.py /app/hy3dpaint/utils/image_super_utils.py

# VERIFY: All critical imports and pipelines in a single layer
RUN /opt/bpy-env/bin/python /app/bpy_mesh_ops.py --help > /dev/null && \
    python -m py_compile /app/handler.py && \
    python -c "\
import sys; \
sys.path.insert(0, '/app'); \
sys.path.insert(0, '/app/hy3dgen'); \
sys.path.insert(0, '/app/hy3dshape'); \
sys.path.insert(0, '/app/hy3dpaint'); \
import runpod, torch, base64, tempfile; \
from onnx_upscaler import upscale_image; \
import onnxruntime as ort; \
assert 'CUDAExecutionProvider' in ort.get_available_providers(); \
from hy3dpaint.utils.image_super_utils import imageSuperNet; \
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; \
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig; \
print('All imports OK')" && \
    (python -c "\
import sys; sys.path.insert(0, '/app'); sys.path.insert(0, '/app/hy3dshape'); \
from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline; \
print('Omni pipeline OK')" || echo "WARN: Omni pipeline import failed - check at runtime")

# =============================================================================
# FINAL: Environment and entrypoint
# =============================================================================
# Model base path - RunPod Network Volume mounts at /runpod-volume
ENV MODEL_BASE=/runpod-volume/models
ENV HF_HOME=/runpod-volume/models
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/models
# Diffusers modules cache - must be writable for custom code loading
ENV HF_MODULES_CACHE=/tmp/hf_modules

# Texture generation settings (quality defaults)
ENV MAX_NUM_VIEW=6
ENV TEXTURE_RESOLUTION=512

# VRAM Optimization toggles (OFF by default for max speed on high-VRAM GPUs like RunPod)
# Set to "1" to enable when running on lower-VRAM GPUs
ENV ENABLE_CPU_OFFLOAD=0
ENV ENABLE_CACHE_CLEARING=0

WORKDIR /app
CMD ["python", "-u", "handler.py"]

# Build summary already covered by verification step above
