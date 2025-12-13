import bpy
import torch

from bpy.props import PointerProperty, EnumProperty, IntProperty, BoolProperty
from bpy.types import PropertyGroup, Object

from .baking import BakingSettings
from .img import ImageMappingSettings
from .lerp import InterpolationSettings
from .map_ops import MapOperationSettings
from .solver import SolverSettings
from .svm import ScalarVertexMapSettings, RemappingStack
from .v_group import SimpleVertexGroup
from ..utils.blend_data.enums import BlendEnums
from .device import TorchDevice


class MinSrfSettings(PropertyGroup):
    solver: PointerProperty(type=SolverSettings)
    apply_after: EnumProperty(
        name="Apply after",
        description="Applies modifier and all prior ones in stack before transforming, preserves ones after.",
        items=lambda self, context: BlendEnums.modifiers(self, context),
    )
    fixed_verts: PointerProperty(type=SimpleVertexGroup)


class FlationSettings(PropertyGroup):
    solver: PointerProperty(type=SolverSettings)
    apply_after: EnumProperty(
        name="Apply after",
        description="Applies modifier and all prior ones in stack before transforming, preserves ones after.",
        items=lambda self, context: BlendEnums.modifiers(self, context),
    )
    active_constraint: EnumProperty(
        name="Constraint",
        description="Constraint to be currently edited",
        items=[
            ("DISPLACEMENT", "Displacement", ""),
            ("LAPLACIAN", "Smoothness", ""),
            ("ALPHA", "Normal", ""),
            ("BETA", "Tangent", ""),
        ],
    )
    fixed_verts: PointerProperty(type=SimpleVertexGroup)
    displacement: PointerProperty(type=ScalarVertexMapSettings)
    laplacian: PointerProperty(type=ScalarVertexMapSettings)
    alpha: PointerProperty(type=ScalarVertexMapSettings)
    beta: PointerProperty(type=ScalarVertexMapSettings)


class SoftenVertexGroupSettings(PropertyGroup):
    group: EnumProperty(
        name="Vertex Group",
        items=lambda self, context: BlendEnums.vertex_groups(self, context),
    )
    rings: IntProperty(
        name="Rings", description="Topologoical smoothing distance", min=1, default=5
    )
    copy: BoolProperty(name="Copy", description="Apply to copy", default=True)
    direction: EnumProperty(
        name="Direction", items=[("IN", "Inwards", ""), ("OUT", "Outwards", "")]
    )


class HardenVertexGroupSettings(PropertyGroup):
    group: EnumProperty(
        name="Vertex Group",
        items=lambda self, context: BlendEnums.vertex_groups(self, context),
    )
    copy: BoolProperty(name="Copy", description="Apply to copy", default=True)


class RemapVertexGroupSettings(PropertyGroup):
    remap: PointerProperty(type=RemappingStack)
    group: PointerProperty(type=SimpleVertexGroup)


class TransferVertexGroupSettings(PropertyGroup):
    target: PointerProperty(type=Object)
    group: PointerProperty(type=SimpleVertexGroup)
    apply_after: EnumProperty(
        name="Apply after",
        description="Applies modifier and all prior ones in stack before transforming, preserves ones after.",
        items=lambda self, context: BlendEnums.modifiers(self, context),
    )


class GlobalSettings(PropertyGroup):
    device: PointerProperty(type=TorchDevice)
    minsrf: PointerProperty(type=MinSrfSettings)
    flation: PointerProperty(type=FlationSettings)
    vghard: PointerProperty(type=HardenVertexGroupSettings)
    vgsoft: PointerProperty(type=SoftenVertexGroupSettings)
    vgtransfer: PointerProperty(type=TransferVertexGroupSettings)
    vgremap: PointerProperty(type=RemapVertexGroupSettings)
    bake: PointerProperty(type=BakingSettings)
    imgmap: PointerProperty(type=ImageMappingSettings)
    mapops: PointerProperty(type=MapOperationSettings)
    lerp: PointerProperty(type=InterpolationSettings)
