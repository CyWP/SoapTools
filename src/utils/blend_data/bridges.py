import bpy
import numpy as np
import torch

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


def tensor2mesh(
    context, V: torch.Tensor, F: torch.Tensor, name: str
) -> bpy.types.Object:
    return np2mesh(context, V.cpu().numpy(), F.cpu().numpy(), name)


def mesh2np(mesh_obj: bpy.types.Object) -> Tuple[np.ndarray, np.ndarray]:
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


def mesh2tensor(
    mesh_obj: bpy.types.Object, device: torch.device, dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    V, F = mesh2np(mesh_obj)
    return torch.tensor(V, device=device, dtype=dtype), torch.tensor(
        F, device=device, dtype=torch.long
    )


def vg2np(mesh_obj: bpy.types.Object, group_name: str) -> Tuple[np.ndarray, np.ndarray]:
    if mesh_obj.type != "MESH":
        raise ValueError("Object must be a mesh.")

    vg = mesh_obj.vertex_groups.get(group_name)
    if vg is None:
        raise ValueError(f"Vertex group '{group_name}' not found.")

    group_idx = vg.index
    n_verts = len(mesh_obj.data.vertices)

    # Flatten all vertex groups info into numpy arrays
    all_vertex_indices = np.array(
        list(
            chain.from_iterable(
                [[v.index] * len(v.groups) for v in mesh_obj.data.vertices]
            )
        ),
        dtype=np.int32,
    )
    all_group_indices = np.array(
        list(
            chain.from_iterable(
                [[g.group for g in v.groups] for v in mesh_obj.data.vertices]
            )
        ),
        dtype=np.int32,
    )
    all_weights = np.array(
        list(
            chain.from_iterable(
                [[g.weight for g in v.groups] for v in mesh_obj.data.vertices]
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


def vg2tensor(
    mesh_obj: bpy.types.Object,
    group_name: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    W, idx = vg2np(mesh_obj, group_name)
    nV = len(mesh_obj.data.vertices)
    weights = torch.zeros((nV,), device=device, dtype=dtype)
    weights[idx] = torch.tensor(W, device=device, dtype=dtype)
    return weights, torch.tensor(idx, device=device, dtype=torch.long)


def vn2np(mesh_obj: bpy.types.Object) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return vertex normals from a Blender mesh as NumPy arrays.
    Uses the base mesh only (no modifiers).
    """
    if mesh_obj.type != "MESH":
        raise TypeError("Input must be a mesh object")
    mesh = mesh_obj.data
    mesh.calc_loop_triangles()
    # Vertex normals
    N_v = np.array([v.normal for v in mesh.vertices], dtype=np.float32)
    return N_v


def vn2tensor(
    mesh_obj: bpy.types.Object, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    return torch.tensor(vn2np(mesh_obj), device=device, dtype=dtype)


def fn2np(mesh_obj: bpy.types.Object) -> torch.Tensor:
    """
    Return face normals from a Blender mesh as NumPy arrays.
    Uses the base mesh only (no modifiers).
    """
    if mesh_obj.type != "MESH":
        raise TypeError("Input must be a mesh object")

    mesh = mesh_obj.data
    mesh.calc_normals_split()  # ensure normals are up to date
    mesh.calc_loop_triangles()
    # Face (triangle) normals
    N_f = np.array([tri.normal for tri in mesh.loop_triangles], dtype=np.float32)

    return N_f


def fn2tensor(
    mesh_obj: bpy.types.Object, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    return torch.tensor(fn2np(mesh_obj), device=device, dtype=dtype)


def e2np(mesh_obj: bpy.types.Object) -> np.ndarray:
    """
    Return triangulated mesh edges as a NumPy array of vertex index pairs.
    Each edge comes from the mesh's triangulated faces.
    """
    if mesh_obj.type != "MESH":
        raise TypeError("Input must be a mesh object")

    mesh = mesh_obj.data
    mesh.calc_loop_triangles()  # ensure triangulated faces

    # Collect all triangle edges
    tris = np.array([tri.vertices[:] for tri in mesh.loop_triangles], dtype=np.int32)
    edges = np.concatenate(
        [
            tris[:, [0, 1]],
            tris[:, [1, 2]],
            tris[:, [2, 0]],
        ],
        axis=0,
    )

    # Sort each edge and remove duplicates
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    return edges


def e2tensor(
    mesh_obj: bpy.types.Object, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    return torch.tensor(e2np(mesh_obj), device=device, dtype=dtype)
