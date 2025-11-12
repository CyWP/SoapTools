import bpy

from bpy.props import PointerProperty

from .operators import (
    MESH_OT_MinimalSurface,
    MESH_OT_HardenVGroup,
    MESH_OT_SoftenVGroup,
    MESH_OT_Inflation,
)
from .properties import (
    GlobalSettings,
    MinSrfSettings,
    FlationSettings,
    SolverSettings,
    SoftenVertexGroupSettings,
    HardenVertexGroupSettings,
    ScalarVertexMapSettings,
    SimpleVertexGroup,
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
    MESH_OT_SoftenVGroup,
    RemappingMode,
    RemappingStack,
    HardenVertexGroupSettings,
    SoftenVertexGroupSettings,
    ScalarVertexMapSettings,
    SolverSettings,
    SimpleVertexGroup,
    REMAP_UL_ModeList,
    REMAP_OT_RemoveModeOperator,
    REMAP_OT_AddModeOperator,
    MinSrfSettings,
    FlationSettings,
    GlobalSettings,
]


def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            bpy.utils.unregister_class(cls)
            bpy.utils.register_class(cls)
    bpy.types.Scene.soap_settings = PointerProperty(type=GlobalSettings)


def unregister():
    del bpy.types.Scene.soap_settings
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
