import bpy

from bpy.types import Object, Image
from contextlib import contextmanager
from typing import List

from .mesh_obj import apply_first_n_modifiers, safe_delete


def duplicate_mesh_object(obj: bpy.types.Object, deep: bool = True) -> bpy.types.Object:
    """
    Fully duplicate a mesh object so that it has its own independent mesh datablock,
    not linked to the original.
    """
    if obj.type != "MESH":
        raise ValueError("Only mesh objects can be duplicated")

    # Duplicate the object
    new_obj = obj.copy()

    if deep:
        # Duplicate the mesh datablock
        new_obj.data = obj.data.copy()
        new_obj.data.use_fake_user = False  # ensure it's not preserved unnecessarily

    # Clear animation data, constraints, etc. if needed
    new_obj.animation_data_clear()

    return new_obj


def link_to_same_scene_collections(
    original: bpy.types.Object, new_obj: bpy.types.Object, scene=None
):
    """Link new_obj to the same collections in the scene as original."""
    if scene is None:
        scene = bpy.context.scene

    # Remove from all current collections first
    for coll in new_obj.users_collection:
        coll.objects.unlink(new_obj)

    # Link to collections where original exists
    for coll in original.users_collection:
        # Ensure the collection belongs to the scene
        if coll.name in scene.collection.children or coll == scene.collection:
            coll.objects.link(new_obj)


@contextmanager
def temp_copy(obj: Object, apply_after: int = 0, strict_vgs: List[str] = None):
    new_obj = None
    try:
        new_obj = duplicate_mesh_object(obj, deep=True)
        if apply_after > 0:
            link_to_same_scene_collections(obj, new_obj)
            apply_first_n_modifiers(
                new_obj,
                apply_after,
                strict_vgs if strict_vgs is not None else [],
            )
        yield new_obj
    finally:
        if new_obj is not None:
            safe_delete(new_obj)


def transfer_object_state(src: bpy.types.Object, dst: bpy.types.Object):
    """
    Transfer all object-level state from src to dst,
    excluding mesh datablock and object name.
    """
    dst.matrix_world = src.matrix_world.copy()
    dst.matrix_parent_inverse = src.matrix_parent_inverse.copy()
    dst.parent = src.parent
    dst.parent_type = src.parent_type
    dst.parent_bone = src.parent_bone
    dst.hide_viewport = src.hide_viewport
    dst.hide_render = src.hide_render
    dst.hide_select = src.hide_select
    dst.visible_camera = src.visible_camera
    dst.visible_diffuse = src.visible_diffuse
    dst.visible_glossy = src.visible_glossy
    dst.visible_shadow = src.visible_shadow
    dst.visible_transmission = src.visible_transmission
    dst.visible_volume_scatter = src.visible_volume_scatter
    dst.display_type = src.display_type
    dst.show_bounds = src.show_bounds
    dst.display_bounds_type = src.display_bounds_type
    dst.show_name = src.show_name
    dst.show_axis = src.show_axis
    dst.show_wire = src.show_wire
    dst.show_all_edges = src.show_all_edges
    dst.pass_index = src.pass_index
    dst.color = src.color[:]
    dst.instance_type = src.instance_type
    dst.use_instance_faces_scale = src.use_instance_faces_scale
    dst.constraints.clear()
    for c in src.constraints:
        nc = dst.constraints.new(type=c.type)
        for prop in c.bl_rna.properties:
            if prop.is_readonly or prop.identifier == "rna_type":
                continue
            setattr(nc, prop.identifier, getattr(c, prop.identifier))
    dst.modifiers.clear()
    for m in src.modifiers:
        nm = dst.modifiers.new(name=m.name, type=m.type)
        for prop in m.bl_rna.properties:
            if prop.is_readonly or prop.identifier in {"name", "rna_type"}:
                continue
            setattr(nm, prop.identifier, getattr(m, prop.identifier))
    dst.vertex_groups.clear()
    for vg in src.vertex_groups:
        dst.vertex_groups.new(name=vg.name)
    for k, v in src.items():
        try:
            dst[k] = v
        except:
            pass
    if src.animation_data:
        dst.animation_data_create()
        dst.animation_data.action = src.animation_data.action


def bake_material(
    obj: Object,
    uv_map: str,
    material_name: str,
    width: int = 512,
    height: int = 512,
    bake_type: str = "EMIT",
) -> Image:

    # Ensure object is active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Set the active UV map
    uv_layer = obj.data.uv_layers.get(uv_map)
    if uv_layer is None:
        raise ValueError(f"UV map '{uv_map}' not found on object '{obj.name}'")
    obj.data.uv_layers.active = uv_layer

    # Get the material
    mat = bpy.data.materials.get(material_name)
    if mat is None:
        raise ValueError(f"Material '{material_name}' not found")

    # Assign material if not already
    if obj.active_material != mat:
        # Ensure material exists in slots
        if mat.name not in obj.data.materials:
            obj.data.materials.append(mat)

    # Make it active
    obj.active_material = mat

    # Create internal image
    img_name = f"Bake_{material_name}_{bake_type}"
    img = bpy.data.images.new(
        name=img_name, width=width, height=height, alpha=True, float_buffer=True
    )

    # Add temporary Image Texture node and make it active
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = img
    nodes.active = tex_node

    # Switch to object mode
    bpy.ops.object.mode_set(mode="OBJECT")

    # Set bake engine to Cycles for baking
    bpy.context.scene.render.engine = "CYCLES"

    # Bake settings
    cycles_settings = bpy.context.scene.cycles
    cycles_settings.bake_type = bake_type
    if bake_type == "DIFFUSE":
        cycles_settings.use_bake_direct = False
        cycles_settings.use_bake_indirect = False
        cycles_settings.cycles.use_bake_color = True

    try:
        # Perform bake
        bpy.ops.object.bake(type=bake_type)
    except:
        cycles_settings.device = "CPU"
        bpy.ops.object.bake(type=bake_type)

    # Remove temporary node
    nodes.remove(tex_node)

    # internally save image
    img.pack()
    # Return the baked image
    return img
