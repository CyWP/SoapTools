import bpy

from bpy.props import PointerProperty

from .operators import (
    MESH_OT_MinimalSurface,
    MESH_OT_HardenVGroup,
    MESH_OT_SoftenVGroup,
    MESH_OT_Inflation,
    MESH_OT_SelectBoundary,
    MESH_OT_BakeChannel,
    MESH_OT_ImageToVG,
    MESH_OT_RemapVGroup,
)
from .properties import (
    GlobalSettings,
    RemapVertexGroupSettings,
    MinSrfSettings,
    FlationSettings,
    ImageSettings,
    ImageMappingSettings,
    SolverSettings,
    SoftenVertexGroupSettings,
    HardenVertexGroupSettings,
    ScalarVertexMapSettings,
    SimpleVertexGroup,
    RemappingMode,
    RemappingStack,
    BakingSettings,
    REMAP_UL_ModeList,
    REMAP_OT_RemoveModeOperator,
    REMAP_OT_AddModeOperator,
)


classes = [
    MESH_OT_MinimalSurface,
    MESH_OT_HardenVGroup,
    MESH_OT_Inflation,
    MESH_OT_SoftenVGroup,
    MESH_OT_SelectBoundary,
    MESH_OT_BakeChannel,
    MESH_OT_ImageToVG,
    MESH_OT_RemapVGroup,
    BakingSettings,
    ImageSettings,
    ImageMappingSettings,
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
    RemapVertexGroupSettings,
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
