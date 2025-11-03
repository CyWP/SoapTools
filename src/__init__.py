import bpy

from .operators import MESH_OT_MinimalSurface, MESH_OT_HardenVGroup, MESH_OT_Inflation
from .properties import GlobalSettings


classes = [
    MESH_OT_MinimalSurface,
    MESH_OT_HardenVGroup,
    MESH_OT_Inflation,
    GlobalSettings,
]


def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            bpy.utils.unregister_class(cls)
            bpy.utils.register_class(cls)
    bpy.types.Scene.soap_settings = bpy.props.PointerProperty(type=GlobalSettings)


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
    del bpy.types.Scene.soap_settings


if __name__ == "__main__":
    register()
    mp.set_start_method("spawn", force=True)
