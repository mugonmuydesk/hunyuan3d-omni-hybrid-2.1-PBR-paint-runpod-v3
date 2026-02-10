"""
Mesh utilities for Hunyuan3D - replaces original mesh_utils.py

This version replicates the original API but uses subprocess for Blender operations
since bpy doesn't support Python 3.12.

The Blender operations run in /opt/bpy-env/ via bpy_mesh_ops.py
"""

import os
import subprocess
import cv2
import numpy as np
import trimesh
from io import StringIO
from typing import Any, Dict, Optional, Tuple


# =============================================================================
# Blender subprocess wrapper (replaces direct bpy calls)
# =============================================================================

BPY_PYTHON = "/opt/bpy-env/bin/python"
BPY_SCRIPT = "/app/bpy_mesh_ops.py"


def convert_obj_to_glb(
    obj_path: str,
    glb_path: str,
    shade_type: str = "SMOOTH",
    auto_smooth_angle: float = 60,
    merge_vertices: bool = False
) -> bool:
    """
    Convert OBJ file to GLB format using Blender via subprocess.

    Args:
        obj_path: Path to input OBJ file
        glb_path: Path for output GLB file
        shade_type: Shading mode - "SMOOTH", "FLAT", or "auto"
        auto_smooth_angle: Angle for auto-smooth (degrees)
        merge_vertices: Whether to merge duplicate vertices

    Returns:
        True if successful
    """
    # Map shade_type to bpy_mesh_ops format
    shading_map = {"SMOOTH": "smooth", "FLAT": "flat", "AUTO": "auto"}
    shading = shading_map.get(shade_type.upper(), "auto")

    cmd = [
        BPY_PYTHON,
        BPY_SCRIPT,
        "convert_obj_to_glb",
        obj_path,
        glb_path,
        f"--shading={shading}",
        f"--angle={auto_smooth_angle}"
    ]

    if merge_vertices:
        cmd.append("--merge-verts")
    else:
        cmd.append("--no-merge-verts")

    print(f"Running bpy mesh conversion: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300
    )

    if result.returncode != 0:
        print(f"bpy stderr: {result.stderr}")
        raise RuntimeError(f"bpy mesh conversion failed: {result.stderr}")

    print(f"bpy stdout: {result.stdout}")

    if not os.path.exists(glb_path):
        raise RuntimeError(f"Output file not created: {glb_path}")

    return True


# =============================================================================
# Helper functions (from original mesh_utils.py)
# =============================================================================

def _safe_extract_attribute(obj: Any, attr_path: str, default: Any = None) -> Any:
    """Safely extract nested attribute from object."""
    try:
        result = obj
        for attr in attr_path.split('.'):
            result = getattr(result, attr)
        return result
    except AttributeError:
        return default


def _convert_to_numpy(data: Any, dtype: np.dtype) -> Optional[np.ndarray]:
    """Convert data to numpy array with specified dtype."""
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return data.astype(dtype)
    return np.array(data, dtype=dtype)


def _get_base_path_and_name(mesh_path: str) -> Tuple[str, str]:
    """Get base path without extension and mesh name."""
    base_path = os.path.splitext(mesh_path)[0]
    name = os.path.basename(base_path)
    return base_path, name


def _save_texture_map(
    texture: np.ndarray,
    base_path: str,
    suffix: str = "",
    image_format: str = ".jpg",
    color_convert: Optional[int] = None,
) -> str:
    """Save texture map with optional color conversion."""
    path = f"{base_path}{suffix}{image_format}"
    processed_texture = (texture * 255).astype(np.uint8)

    if color_convert is not None:
        processed_texture = cv2.cvtColor(processed_texture, color_convert)
        cv2.imwrite(path, processed_texture)
    else:
        # BGR to RGB conversion for cv2
        cv2.imwrite(path, processed_texture[..., ::-1])

    return os.path.basename(path)


def _write_mtl_properties(f, properties: Dict[str, Any]):
    """Write material properties to MTL file."""
    for key, value in properties.items():
        if isinstance(value, (list, tuple)):
            f.write(f"{key} {' '.join(map(str, value))}\n")
        else:
            f.write(f"{key} {value}\n")


def _create_obj_content(
    vtx_pos: np.ndarray,
    vtx_uv: np.ndarray,
    pos_idx: np.ndarray,
    uv_idx: np.ndarray,
    name: str
) -> str:
    """Create OBJ file content."""
    buffer = StringIO()

    buffer.write(f"mtllib {name}.mtl\no {name}\n")
    np.savetxt(buffer, vtx_pos, fmt="v %.6f %.6f %.6f")
    np.savetxt(buffer, vtx_uv, fmt="vt %.6f %.6f")
    buffer.write("s 0\nusemtl Material\n")

    pos_idx_plus1 = pos_idx + 1
    uv_idx_plus1 = uv_idx + 1
    face_format = np.frompyfunc(lambda *x: f"{int(x[0])}/{int(x[1])}", 2, 1)
    faces = face_format(pos_idx_plus1, uv_idx_plus1)
    face_strings = [f"f {' '.join(face)}" for face in faces]
    buffer.write("\n".join(face_strings) + "\n")

    return buffer.getvalue()


def _create_mtl_file(base_path: str, texture_maps: Dict[str, str], is_pbr: bool):
    """Create MTL material file."""
    mtl_path = f"{base_path}.mtl"

    with open(mtl_path, "w") as f:
        f.write("newmtl Material\n")

        if is_pbr:
            properties = {
                "Kd": [0.800, 0.800, 0.800],
                "Ke": [0.000, 0.000, 0.000],
                "Ni": 1.500,
                "d": 1.0,
                "illum": 2,
                "map_Kd": texture_maps["diffuse"],
            }
            _write_mtl_properties(f, properties)

            map_configs = [
                ("metallic", "map_Pm"),
                ("roughness", "map_Pr"),
                ("normal", "map_Bump -bm 1.0")
            ]

            for texture_key, mtl_key in map_configs:
                if texture_key in texture_maps:
                    f.write(f"{mtl_key} {texture_maps[texture_key]}\n")
        else:
            properties = {
                "Ns": 250.000000,
                "Ka": [0.200, 0.200, 0.200],
                "Kd": [0.800, 0.800, 0.800],
                "Ks": [0.500, 0.500, 0.500],
                "Ke": [0.000, 0.000, 0.000],
                "Ni": 1.500,
                "d": 1.0,
                "illum": 3,
                "map_Kd": texture_maps["diffuse"],
            }
            _write_mtl_properties(f, properties)


# =============================================================================
# Main API functions (matching original mesh_utils.py signatures)
# =============================================================================

def load_mesh(mesh):
    """Load mesh data including vertices, faces, UV coordinates and texture.

    Args:
        mesh: A trimesh.Trimesh object

    Returns:
        Tuple of (vtx_pos, pos_idx, vtx_uv, uv_idx, texture_data)
    """
    # Extract vertex positions and face indices
    vtx_pos = _safe_extract_attribute(mesh, "vertices")
    pos_idx = _safe_extract_attribute(mesh, "faces")

    # Extract UV coordinates
    vtx_uv = _safe_extract_attribute(mesh, "visual.uv")
    uv_idx = pos_idx  # Reuse face indices for UV mapping

    # Convert to numpy arrays with appropriate dtypes
    vtx_pos = _convert_to_numpy(vtx_pos, np.float32)
    pos_idx = _convert_to_numpy(pos_idx, np.int32)
    vtx_uv = _convert_to_numpy(vtx_uv, np.float32)
    uv_idx = _convert_to_numpy(uv_idx, np.int32)

    texture_data = None
    return vtx_pos, pos_idx, vtx_uv, uv_idx, texture_data


def save_obj_mesh(
    mesh_path,
    vtx_pos,
    pos_idx,
    vtx_uv,
    uv_idx,
    texture,
    metallic=None,
    roughness=None,
    normal=None
):
    """Save mesh as OBJ file with textures and material."""
    vtx_pos = _convert_to_numpy(vtx_pos, np.float32)
    vtx_uv = _convert_to_numpy(vtx_uv, np.float32)
    pos_idx = _convert_to_numpy(pos_idx, np.int32)
    uv_idx = _convert_to_numpy(uv_idx, np.int32)

    base_path, name = _get_base_path_and_name(mesh_path)

    # Write OBJ file
    obj_content = _create_obj_content(vtx_pos, vtx_uv, pos_idx, uv_idx, name)
    with open(mesh_path, "w") as obj_file:
        obj_file.write(obj_content)

    # Save texture maps
    texture_maps = {}
    texture_maps["diffuse"] = _save_texture_map(texture, base_path)

    if metallic is not None:
        texture_maps["metallic"] = _save_texture_map(
            metallic, base_path, "_metallic", color_convert=cv2.COLOR_RGB2GRAY
        )
    if roughness is not None:
        texture_maps["roughness"] = _save_texture_map(
            roughness, base_path, "_roughness", color_convert=cv2.COLOR_RGB2GRAY
        )
    if normal is not None:
        texture_maps["normal"] = _save_texture_map(normal, base_path, "_normal")

    # Create MTL file
    _create_mtl_file(base_path, texture_maps, metallic is not None)


def save_mesh(
    mesh_path,
    vtx_pos,
    pos_idx,
    vtx_uv,
    uv_idx,
    texture,
    metallic=None,
    roughness=None,
    normal=None
):
    """Save mesh using OBJ format with textures and material.

    This is the main entry point for saving textured meshes.
    """
    save_obj_mesh(
        mesh_path,
        vtx_pos,
        pos_idx,
        vtx_uv,
        uv_idx,
        texture,
        metallic=metallic,
        roughness=roughness,
        normal=normal
    )
