import bpy
import numpy as np

from itertools import chain
from typing import Tuple


def np2mesh(context, V: np.ndarray, F: np.ndarray, name: str) -> bpy.types.Object:

    if not isinstance(V, np.ndarray) or not isinstance(F, np.ndarray):
        raise ValueError("V and F must be NumPy arrays.")
    if V.shape[1] != 3:
        raise ValueError("V must have shape (n, 3).")
    if F.shape[1] not in (3, 4):
        raise ValueError("F must have shape (m, 3) or (m, 4).")

    # Create a new mesh
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    # Link object to the active collection
    collection = context.collection
    collection.objects.link(obj)
    # Create the mesh from the vertex and face data
    mesh.from_pydata(V.tolist(), [], F.tolist())
    # Update mesh with calculated normals and topology
    mesh.update()

    return obj


def mesh2np(mesh_obj: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a Blender mesh object to numpy arrays of vertices and triangulated faces.
    Uses the base mesh only (no modifiers) and is fully vectorized.
    """
    if mesh_obj.type != "MESH":
        raise TypeError("Input must be a mesh object")

    mesh = mesh_obj.data
    mesh.calc_loop_triangles()  # ensure triangulated faces

    V = np.array([v.co for v in mesh.vertices], dtype=np.float32)
    F = np.array([tri.vertices[:] for tri in mesh.loop_triangles], dtype=np.int32)

    return V, F


def vg2np(mesh: bpy.types.Object, group_name: str) -> Tuple[np.ndarray, np.ndarray]:
    if mesh.type != "MESH":
        raise ValueError("Object must be a mesh.")

    vg = mesh.vertex_groups.get(group_name)
    if vg is None:
        raise ValueError(f"Vertex group '{group_name}' not found.")

    group_idx = vg.index
    n_verts = len(mesh.data.vertices)

    # Flatten all vertex groups info into numpy arrays
    all_vertex_indices = np.array(
        list(
            chain.from_iterable([[v.index] * len(v.groups) for v in mesh.data.vertices])
        ),
        dtype=np.int32,
    )
    all_group_indices = np.array(
        list(
            chain.from_iterable(
                [[g.group for g in v.groups] for v in mesh.data.vertices]
            )
        ),
        dtype=np.int32,
    )
    all_weights = np.array(
        list(
            chain.from_iterable(
                [[g.weight for g in v.groups] for v in mesh.data.vertices]
            )
        ),
        dtype=np.float32,
    )

    # Mask only the entries for the target group
    mask = all_group_indices == group_idx
    selected_indices = all_vertex_indices[mask]
    selected_weights = all_weights[mask]
    nz_mask = selected_weights > 0.0
    return selected_weights[nz_mask], selected_indices[nz_mask]
