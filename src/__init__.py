import bpy

from .operators import MESH_OT_MinimalSurface, MESH_OT_HardenVGroup, MESH_OT_Inflation
from .properties import (
    GlobalSettings,
    ScalarVertexMapSettings,
    RemappingMode,
    RemappingStack,
    REMAP_UL_ModeList,
    REMAP_OT_RemoveModeOperator,
    REMAP_OT_AddModeOperator,
)


classes = [
    MESH_OT_MinimalSurface,
    MESH_OT_HardenVGroup,
    MESH_OT_Inflation,
    GlobalSettings,
    ScalarVertexMapSettings,
    RemappingMode,
    RemappingStack,
    REMAP_UL_ModeList,
    REMAP_OT_RemoveModeOperator,
    REMAP_OT_AddModeOperator,
]


def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            bpy.utils.unregister_class(cls)
            bpy.utils.register_class(cls)
    bpy.types.Scene.soap_settings = bpy.props.PointerProperty(type=GlobalSettings)
    #
    bpy.types.Scene.soap_inflate_disp_map = bpy.props.PointerProperty(
        type=ScalarVertexMapSettings
    )


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
    del bpy.types.Scene.soap_settings
    del bpy.types.Scene.soap_inflate_disp_map


if __name__ == "__main__":
    register()
