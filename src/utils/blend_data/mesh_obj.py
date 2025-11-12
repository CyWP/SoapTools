import bpy
import bmesh
import numpy as np

from bpy.types import Object, Context
from typing import List

from .vertex_groups import harden_vertex_group


def apply_first_n_modifiers(
    obj: Object,
    n: int,
    strict_vgs: List[str] = None,
):
    """
    Apply the first n modifiers of a mesh object directly.
    strict_vgs keep their initial sharpness using a
    custom 'strict' subdivision that preserves vertex groups on original vertices only.
    """
    if obj.type != "MESH":
        raise ValueError("Object must be a mesh")

    # Store the current active object
    prev_active = bpy.context.view_layer.objects.active
    bpy.context.view_layer.objects.active = obj

    try:
        for mod in list(obj.modifiers)[:n]:
            try:
                is_subsurf = mod.type == "SUBSURF"
                bpy.ops.object.modifier_apply(modifier=mod.name)
                if is_subsurf:
                    for vg in strict_vgs:
                        harden_vertex_group(obj, vg)
            except RuntimeError as e:
                print(f"Failed to apply modifier {mod.name}: {e}")  # TODO: add logger
    finally:
        # Restore previous active object
        bpy.context.view_layer.objects.active = prev_active


def modifier_items(caller, context: Context):
    obj = context.active_object
    if obj and obj.type == "MESH" and obj.modifiers:
        data = [(str(i + 1), mod.name, "") for i, mod in enumerate(obj.modifiers)]
        return [("0", "None", ""), *data]
    return [("0", "Modifier", "")]


def update_mesh_vertices(obj: Object, V: np.ndarray):
    """Update the vertex positions of an existing mesh object."""
    if not isinstance(obj, Object) or obj.type != "MESH":
        raise TypeError("obj must be a Blender mesh object")
    if V.shape[1] != 3:
        raise ValueError("V must have shape (n, 3)")

    mesh = obj.data
    n_verts = len(mesh.vertices)
    if V.shape[0] != n_verts:
        raise ValueError(
            f"Vertex count mismatch: mesh has {n_verts}, array has {V.shape[0]}"
        )

    # Ensure float32 for speed
    if V.dtype != np.float32:
        V = V.astype(np.float32)

    # Fast update via foreach_set
    mesh.vertices.foreach_set("co", V.ravel())
    mesh.update()


def select_boundary(obj: Object):
    if obj is None or obj.type != "MESH":
        raise TypeError("Active object is not a mesh.")

    # Ensure we're in Edit Mode and have an up-to-date mesh
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)

    selected_verts = [v for v in bm.verts if v.select]
    for v in selected_verts:
        v.select = False
    selected_verts = set(selected_verts)

    def is_inside_boundary(e) -> bool:
        nonlocal selected_verts
        return e.vertices[0] in selected_verts and e.vertices[1] in selected_verts

    for v in selected_verts:
        if not all(is_inside_boundary(e) for e in v.link_edges):
            v.select = True

    # Update the edit mesh so the viewport updates
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
