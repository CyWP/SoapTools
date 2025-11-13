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
        raise TypeError("Active object must be a mesh")

    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    selected_verts = {v for v in bm.verts if v.select}
    # selected_edges = {e for e in bm.verts if e.select}

    for v in bm.verts:
        v.select = False
    for e in bm.edges:
        e.select = False
    for f in bm.faces:
        f.select = False

    select_verts = set()
    select_edges = set()

    for v in selected_verts:
        # A vertex is boundary if at least one of its edges connects to an unselected vertex
        linked_edges = v.link_edges
        edges_selected = [e.other_vert(v) in selected_verts for e in linked_edges]
        if sum(edges_selected) < len(edges_selected):
            select_verts.add(v)
            for i, on_edge in enumerate(edges_selected):
                if on_edge:
                    select_edges.add(linked_edges[i])

    for v in selected_verts:
        if v in select_verts:
            continue
        linked_edges = v.link_edges
        edges_selected = [e in select_edges for e in linked_edges]
        num_connected = sum(edges_selected)
        if num_connected <= 1 or num_connected >= len(linked_edges):
            continue
        for i, is_connected in enumerate(edges_selected):
            select_verts.add(v)
            if is_connected:
                select_edges.add(linked_edges[i])

    for v in select_verts:
        v.select = True
        for e in v.link_edges:
            other = e.other_vert(v)
            if other not in select_verts:
                continue

            e.select = True

    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
