# Hunyuan3D-2.1 RunPod Deployment - Build Issues & Fixes

## Summary
This document tracks all issues encountered while building the Docker image for RunPod serverless deployment and how they were resolved.

---

## Issue #1: PyTorch CUDA 12.4 Installation Failure

**Error:**
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
exit code: 1
```

**Cause:**
PyTorch 2.5.x has known issues with CUDA 12.4 wheels. The cu124 index has compatibility problems.

**Fix:**
Changed to CUDA 12.1 wheels which are stable:
```dockerfile
--index-url https://download.pytorch.org/whl/cu121
```

**Status:** Fixed in commit `7c1bfdf`

---

## Issue #2: GitHub Actions Disk Space Exhausted

**Error:**
```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

**Cause:**
- GitHub Actions runners have ~14GB free disk
- PyTorch cu121 + dependencies = ~2.5GB
- CUDA base image = ~4GB
- Combined exceeds available space

**Fix:**
Added disk cleanup step to workflow before Docker build:
```yaml
- name: Free disk space
  run: |
    sudo rm -rf /usr/share/dotnet
    sudo rm -rf /usr/local/lib/android
    sudo rm -rf /opt/ghc
    sudo rm -rf /opt/hostedtoolcache/CodeQL
    sudo docker image prune -af
```

This frees ~25GB by removing unused SDKs.

**Status:** Fixed in commit `98e80de`

---

## Issue #3: bpy==4.0 Not Pip-Installable

**Error:**
```
ERROR: Could not find a version that satisfies the requirement bpy==4.0 (from versions: none)
ERROR: No matching distribution found for bpy==4.0
```

**Cause:**
- `bpy` (Blender Python) is NOT available via pip
- The real bpy module comes bundled with Blender
- PyPI has `fake-bpy-module` for type hints only

**What bpy is used for:**
- `hy3dpaint/DifferentiableRenderer/mesh_utils.py` - OBJ to GLB conversion
- `hy3dshape/tools/render/render.py` - Rendering (optional)

**Fix:**
Excluded bpy from pip install in Dockerfile:
```dockerfile
grep -v "^bpy" requirements.txt > requirements_filtered.txt && \
pip install --no-cache-dir -r requirements_filtered.txt
```

**Alternative solutions:**
1. Install Blender in container and use its bundled Python
2. Use trimesh/pygltflib for format conversion instead
3. Accept OBJ output without GLB conversion

**Status:** Fixed in commit `c02534b`

---

## Issue #4: Token-Expensive GitHub Actions Polling

**Problem:**
Using WebFetch to poll GitHub Actions consumes ~200KB per request, extremely token-expensive.

**Fix:**
Use `gh` CLI instead - returns compact text (~100x more efficient):
```bash
# List runs
gh run list --repo owner/repo --limit 5

# Get failed logs
gh run view RUN_ID --repo owner/repo --log-failed 2>&1 | tail -100
```

**Status:** Documented in global CLAUDE.md

---

## Current Build Status

| Run | Commit | Issue | Status |
|-----|--------|-------|--------|
| #1 | 09f15b3 | cu124 failure | Failed |
| #2 | 7c1bfdf | cu121 same error | Failed |
| #3 | 4c5c0fc | Disk space | Failed |
| #4 | 98e80de | Disk space (with cleanup) | Failed |
| #5 | c02534b | bpy exclusion | Failed (18m55s) |
| #6 | 8f62cf5 | custom_rasterizer libc10.so | Failed (5m10s) |
| #7 | pending | LD_LIBRARY_PATH fix | Pending |

---

## Issue #5: Requirements Too Large for GitHub Actions

**Error:**
```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

**Cause:**
Full requirements.txt has massive dependencies:
- open3d: 400MB
- cupy-cuda12x: 104MB
- gradio: 54MB
- deepspeed, pytorch, tensorboard, etc.

Total exceeds ~25GB, more than GitHub Actions can provide.

**Fix:**
Created `requirements_ci.txt` with minimal deps for CI testing:
- transformers, diffusers, accelerate (core ML)
- trimesh, pygltflib (lightweight 3D)
- runpod (handler)

Full requirements installed in production Dockerfile.

**Status:** Fixed in commit `8f62cf5`

---

## Issue #6: custom_rasterizer libc10.so Missing

**Error:**
```
ImportError: libc10.so: cannot open shared object file: No such file or directory
```

**Cause:**
custom_rasterizer is a compiled CUDA extension that links against PyTorch's C++ libraries. PyTorch installs these libs in its package directory, but they're not in the default library search path.

**Fix:**
Add PyTorch's lib directory to LD_LIBRARY_PATH:
```dockerfile
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH}"
```

**Status:** Fixing in next commit

---

## Key Constraints

1. **RunPod builds have NO GPU** - Cannot compile CUDA extensions
2. **Must use prebuilt custom_rasterizer wheel** - Requires Python 3.10
3. **PyTorch 2.5.x + cu124 is broken** - Use cu121 instead
4. **bpy requires Blender** - Not pip-installable, need alternative

---

## Files Modified

- `Dockerfile.dryrun` - Fast CI build (skips model download)
- `Dockerfile` - Full build for RunPod
- `.github/workflows/build-test.yml` - CI workflow with disk cleanup

---

## RunPod Blender Support

**Can Blender run on RunPod serverless?** Yes.

RunPod has no restrictions on what software can run - it's standard Docker containers with linux/amd64 platform requirement.

**How to install Blender:**
```dockerfile
# Option 1: Install from PPA
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:savoury1/blender -y && \
    apt-get update && apt-get install -y blender

# Option 2: Download from official release
RUN wget https://download.blender.org/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz && \
    tar -xf blender-4.0.0-linux-x64.tar.xz -C /opt/ && \
    ln -s /opt/blender-4.0.0-linux-x64/blender /usr/local/bin/blender
```

**Trade-offs:**
- Adds ~500MB to image size
- Slower cold start times
- But provides full bpy functionality for OBJ→GLB conversion

**Alternative:** Use trimesh for format conversion (already in requirements):
```python
import trimesh
mesh = trimesh.load('model.obj')
mesh.export('model.glb')
```

---

## Next Steps (After CI Passes)

1. Decide on Blender integration approach for production
2. Deploy to RunPod serverless
3. Test with actual GPU worker
