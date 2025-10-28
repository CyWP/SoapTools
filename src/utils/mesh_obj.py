import bpy
import numpy as np

from .vertex_groups import harden_vertex_group


def apply_first_n_modifiers(
    obj: bpy.types.Object, n: int, vertex_group: str, preserve_vg: bool = False
):
    """
    Apply the first n modifiers of a mesh object directly.
    If strict=True, Subdivision Surface modifiers are applied using a
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
                if preserve_vg and is_subsurf:
                    harden_vertex_group(obj, vertex_group)
            except RuntimeError as e:
                print(f"Failed to apply modifier {mod.name}: {e}")
    finally:
        # Restore previous active object
        bpy.context.view_layer.objects.active = prev_active


def update_mesh_vertices(obj: bpy.types.Object, V: np.ndarray):
    """Update the vertex positions of an existing mesh object."""
    if not isinstance(obj, bpy.types.Object) or obj.type != "MESH":
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
