import bpy
import bmesh


def vertex_group_items(self, context):
    obj = context.active_object
    if obj and obj.type == "MESH" and obj.vertex_groups:
        return [(vg.name, vg.name, "") for vg in obj.vertex_groups]
    return [("NONE", "None", "")]


def get_vertex_group_copy(
    obj: bpy.types.Object, vertex_group: str, new_name: str, caller=None
):
    vg = obj.vertex_groups[vertex_group]
    if new_name in obj.vertex_groups and caller:
        caller.report(
            {"WARNING"},
            f"Vertex group '{new_name}' already exists. Overwriting.",
        )
        new_vg = obj.vertex_groups[new_name]
    else:
        new_vg = obj.vertex_groups.new(name=new_name)

    # Copy weights from original to new
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == vg.index:
                new_vg.add([v.index], g.weight, "REPLACE")
    return new_vg


def harden_vertex_group(obj: bpy.types.Object, vertex_group: str = None):
    """
    Sets vertex weights to 0 unless they are >= all their neighbors' weights.
    Operates in place on the given vertex group.
    """
    if obj.type != "MESH":
        raise ValueError("Object must be a mesh")

    vg = obj.vertex_groups.get(vertex_group)
    if vg is None:
        raise ValueError(
            f"Vertex group '{vertex_group}' not found on object '{obj.name}'"
        )

    # Create BMesh for topological access
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    # Cache vertex weights
    weights = {}
    for v in bm.verts:
        try:
            weights[v.index] = vg.weight(v.index)
        except RuntimeError:
            weights[v.index] = 0.0

    # Determine which vertices to keep
    new_weights = [0.0] * len(bm.verts)
    for v in bm.verts:
        w = weights[v.index]
        if all(w >= weights[e.other_vert(v).index] for e in v.link_edges):
            new_weights[v.index] = w  # keep original weight
        # else: stays 0.0

    # Write back weights
    vg.add(range(len(new_weights)), 0.0, "REPLACE")
    for i, w in enumerate(new_weights):
        if w > 0:
            vg.add([i], w, "REPLACE")

    bm.free()
