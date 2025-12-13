from .dev import deps

# ONLY USED FOR DEV, IGNORED BY BUILT RELEASES
if not deps.check_deps():
    deps.install_deps_dev()

import bpy

from bpy.props import PointerProperty

from .operators import (
    SOAP_OT_MinimalSurface,
    SOAP_OT_HardenVGroup,
    SOAP_OT_SoftenVGroup,
    SOAP_OT_TransferVGroup,
    SOAP_OT_RemapVGroup,
    SOAP_OT_Inflation,
    SOAP_OT_SelectBoundary,
    SOAP_OT_BakeChannel,
    SOAP_OT_ImageToVG,
    SOAP_OT_OperateMaps,
    SOAP_OT_Interpolate,
    SOAP_OT_ChangeDevice,
    SOAP_OT_CudaTorch,
)
from .properties import (
    GlobalSettings,
    InterpolationSettings,
    InterpolationTarget,
    RemapVertexGroupSettings,
    MinSrfSettings,
    FlationSettings,
    ImageSettings,
    ImageMappingSettings,
    SolverSettings,
    SoftenVertexGroupSettings,
    HardenVertexGroupSettings,
    TransferVertexGroupSettings,
    InterpolationSettings,
    SymbolicExpression,
    MapOperationSettings,
    ScalarVertexMapSettings,
    SimpleVertexGroup,
    RemappingMode,
    RemappingStack,
    BakingSettings,
    TorchDevice,
    SOAP_UL_ModeList,
    SOAP_OT_RemoveModeOperator,
    SOAP_OT_AddModeOperator,
    SOAP_OT_AddMapVariable,
    SOAP_OT_RemoveMapVariable,
    SOAP_OT_AddInterpolationVariable,
    SOAP_OT_RemoveInterpolationVariable,
)
from .panels import VIEW3D_PT_NPanel, VIEW3D_PT_MapOps, VIEW3D_PT_MeshOps


classes = [
    SOAP_OT_CudaTorch,
    SOAP_OT_ChangeDevice,
    SOAP_OT_Interpolate,
    SOAP_OT_MinimalSurface,
    SOAP_OT_HardenVGroup,
    SOAP_OT_TransferVGroup,
    SOAP_OT_Inflation,
    SOAP_OT_SoftenVGroup,
    SOAP_OT_SelectBoundary,
    SOAP_OT_BakeChannel,
    SOAP_OT_ImageToVG,
    SOAP_OT_RemapVGroup,
    SOAP_OT_OperateMaps,
    TorchDevice,
    BakingSettings,
    SymbolicExpression,
    SimpleVertexGroup,
    ImageSettings,
    ImageMappingSettings,
    RemappingMode,
    RemappingStack,
    HardenVertexGroupSettings,
    TransferVertexGroupSettings,
    SoftenVertexGroupSettings,
    ScalarVertexMapSettings,
    InterpolationTarget,
    MapOperationSettings,
    SolverSettings,
    SOAP_UL_ModeList,
    SOAP_OT_RemoveModeOperator,
    SOAP_OT_AddModeOperator,
    SOAP_OT_AddMapVariable,
    SOAP_OT_RemoveMapVariable,
    SOAP_OT_AddInterpolationVariable,
    SOAP_OT_RemoveInterpolationVariable,
    InterpolationSettings,
    MinSrfSettings,
    FlationSettings,
    RemapVertexGroupSettings,
    InterpolationSettings,
    GlobalSettings,
    VIEW3D_PT_NPanel,
    VIEW3D_PT_MapOps,
    VIEW3D_PT_MeshOps,
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
