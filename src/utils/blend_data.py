import bpy


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
