import bpy

from bpy.types import Context, Object, Image
from typing import List, Tuple


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

    # Return the baked image
    return img
