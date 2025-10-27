import bpy

from .operators import MESH_OT_MinimalSurface, MESH_OT_HardenVGroup


classes = [MESH_OT_MinimalSurface, MESH_OT_HardenVGroup]


def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            bpy.utils.unregister_class(cls)
            bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
