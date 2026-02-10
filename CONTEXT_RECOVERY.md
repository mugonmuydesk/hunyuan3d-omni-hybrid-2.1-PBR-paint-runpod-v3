# Context Recovery - Hunyuan3D-2.1 RunPod Deployment

## Current State
- **Working dir:** `/mnt/c/dev/hunyuan3d-runpod`
- **Branch:** `main`
- **Remote:** `https://github.com/mugonmuydesk/Hunyuan3D-2.1`
- **Task:** Fix Docker build until GitHub Actions CI passes

## Goal
Deploy Hunyuan3D-2.1 (image-to-3D with textures) as a RunPod serverless endpoint.

## Critical Constraint
- **RunPod builds have NO GPU access** - cannot compile CUDA extensions
- **Solution:** Use prebuilt `custom_rasterizer` wheel from Tencent HuggingFace:
  ```
  https://huggingface.co/spaces/tencent/Hunyuan3D-2.1/resolve/main/custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl
  ```
- **Requires Python 3.10** to match wheel (cp310)

## Files
```
/mnt/c/dev/hunyuan3d-runpod/
├── Dockerfile              # Full build (with 7GB model download)
├── Dockerfile.dryrun       # Fast CI build (skips model download) ← GitHub Actions uses this
├── handler.py              # RunPod serverless handler
├── test_input.json         # Sample test input
├── README.md               # Deployment docs
├── .gitignore
├── .github/workflows/build-test.yml  # CI workflow
└── CONTEXT_RECOVERY.md     # This file
```

## GitHub Actions
- **Workflow:** Build Test (Dry Run)
- **Actions URL:** https://github.com/mugonmuydesk/Hunyuan3D-2.1/actions
- **Triggers on:** Push to `main`
- **Uses:** `Dockerfile.dryrun`

## Current Issue Being Fixed
PyTorch 2.5.x + CUDA 12.4 has known installation issues.
- **Fix applied:** Changed `cu124` → `cu121` in both Dockerfiles
- **Build #2 currently running**

## Iteration Loop
```
1. Check build status:
   WebFetch https://github.com/mugonmuydesk/Hunyuan3D-2.1/actions

2. If FAILED - get error details:
   WebFetch https://github.com/mugonmuydesk/Hunyuan3D-2.1/actions/runs/LATEST_RUN_ID

3. Fix the issue in Dockerfile.dryrun (and Dockerfile if same issue)

4. Push fix:
   cd /mnt/c/dev/hunyuan3d-runpop && git add -A && git commit -m "Fix: description" && git push

5. Wait 2 min, repeat from step 1
```

## Build Stages to Pass
1. System dependencies (apt-get, Python 3.10)
2. PyTorch 2.5.1 + CUDA ← **Currently fixing**
3. Clone Hunyuan3D-2.1 repo
4. Python requirements (transformers, diffusers, trimesh, etc.)
5. custom_rasterizer wheel install
6. mesh_inpaint_processor (optional)
7. RunPod SDK
8. Handler verification (imports, syntax)

## Key Dependencies
- **Base image:** `nvidia/cuda:12.4.0-devel-ubuntu22.04`
- **Python:** 3.10 (required for prebuilt wheel)
- **PyTorch:** 2.5.1 with cu121 (cu124 broken)
- **custom_rasterizer:** prebuilt wheel from HuggingFace

## Quick Commands
```bash
# Check git status
cd /mnt/c/dev/hunyuan3d-runpod && git status

# View recent commits
cd /mnt/c/dev/hunyuan3d-runpod && git log --oneline -5

# Edit and push fix
cd /mnt/c/dev/hunyuan3d-runpod && git add -A && git commit -m "Fix: ..." && git push

# Check GitHub Actions (use WebFetch)
# https://github.com/mugonmuydesk/Hunyuan3D-2.1/actions
```

## After Build Passes
1. Go to RunPod Serverless Console
2. New Endpoint → Import Git Repository
3. Select `mugonmuydesk/Hunyuan3D-2.1`, branch `main`
4. Use full `Dockerfile` (not dryrun)
5. GPU: 48GB (A40/A100)
6. Deploy

## Reference Links
- [RunPod GitHub Integration Docs](https://docs.runpod.io/serverless/workers/github-integration)
- [PyTorch cu124 Issue](https://github.com/pytorch/pytorch/issues/138324)
- [Tencent Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
- [Prebuilt wheel location](https://huggingface.co/spaces/tencent/Hunyuan3D-2.1)
