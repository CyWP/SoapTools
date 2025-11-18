import bpy
import bmesh
import torch

from bpy.types import Object


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


def soften_vertex_group_inwards(obj: Object, vg_name: str, num_rings: int):
    """
    Applies cosine smoothing to n outermost rings of a vertex groups. Inner vertices set to 1.
    """
    if obj.type != "MESH":
        raise ValueError("Object must be a mesh")

    if vg_name not in obj.vertex_groups:
        raise ValueError(
            f"{vg_name} is not a valid vertex group for object {obj.name}."
        )

    dat = obj.data
    weights = (-(torch.cos(torch.linspace(0, torch.pi, num_rings + 2)))[1:-1] + 1) * 0.5
    smoothed_vg = obj.vertex_groups[vg_name]

    vgi = obj.vertex_groups[vg_name].index
    vi = set([v.index for v in dat.vertices if vgi in [vg.group for vg in v.groups]])

    edges = dat.edges
    for i in range(num_rings):
        iv = set(vi)
        etemp = []
        ev = []
        for edge in edges:
            if edge.vertices[0] in vi:
                if edge.vertices[1] not in vi:
                    ev.append(edge.vertices[0])
                    iv.discard(edge.vertices[0])
                else:
                    etemp.append(edge)
            elif edge.vertices[1] in vi:
                ev.append(edge.vertices[1])
                iv.discard(edge.vertices[1])
        ev = list(set(ev))
        edges = etemp
        vi = iv
        smoothed_vg.add(ev, weights[i], "REPLACE")
    smoothed_vg.add(list(vi), 1, "REPLACE")


def soften_vertex_group_outwards(obj: Object, vg_name: str, num_rings: int):
    """
    Sets all current vertex weights to 1, and does cosine smoothing based on the number of outer rings on the vertex weights.
    """
    if obj.type != "MESH":
        raise ValueError("Object must be a mesh")

    if vg_name not in obj.vertex_groups:
        raise ValueError(
            f"{vg_name} is not a valid vertex group for object {obj.name}."
        )

    dat = obj.data
    weights = (-(torch.cos(torch.linspace(torch.pi, 0, num_rings + 2)))[1:-1] + 1) * 0.5
    smoothed_vg = obj.vertex_groups[vg_name]

    vgi = obj.vertex_groups[vg_name].index
    vi = set(
        [
            v.index
            for v in dat.vertices
            if vgi in [vg.group for vg in v.groups]
            and smoothed_vg.weight(v.index) > 0.05
        ]
    )
    smoothed_vg.add(list(vi), 1, "REPLACE")

    edges = dat.edges
    for i in range(num_rings):
        etemp = []
        vi_temp = set()
        ev = set()
        for edge in edges:
            if edge.vertices[0] in vi:
                if edge.vertices[1] not in vi:
                    ev.add(edge.vertices[1])
                    vi_temp.add(edge.vertices[1])
            elif edge.vertices[1] in vi:
                ev.add(edge.vertices[0])
                vi_temp.add(edge.vertices[0])
            else:
                etemp.append(edge)
        edges = etemp
        vi = vi | vi_temp
        smoothed_vg.add(list(ev), weights[i], "REPLACE")
