import bpy
import torch

from bpy.props import PointerProperty, EnumProperty, IntProperty, BoolProperty
from bpy.types import PropertyGroup

from .baking import BakingSettings
from .img import ImageMappingSettings
from .solver import SolverSettings
from .svm import ScalarVertexMapSettings, RemappingStack
from .v_group import SimpleVertexGroup
from ..utils.blend_data.enums import BlendEnums


class MinSrfSettings(PropertyGroup):
    solver: PointerProperty(type=SolverSettings)  # type:ignore
    apply_after: EnumProperty(
        name="Apply after",
        description="Applies modifier and all prior ones in stack before transforming, preserves ones after.",
        items=BlendEnums.modifiers,
    )  # type:ignore
    fixed_verts: PointerProperty(type=SimpleVertexGroup)  # type: ignore


class FlationSettings(PropertyGroup):
    solver: PointerProperty(type=SolverSettings)  # type: ignore
    apply_after: EnumProperty(
        name="Apply after",
        description="Applies modifier and all prior ones in stack before transforming, preserves ones after.",
        items=BlendEnums.modifiers,
    )  # type:ignore
    active_constraint: EnumProperty(
        name="Constraint",
        description="Constraint to be currently edited",
        items=[
            ("DISPLACEMENT", "Displacement", ""),
            ("LAPLACIAN", "Smoothness", ""),
            ("ALPHA", "Normal", ""),
            ("BETA", "Tangent", ""),
        ],
    )  # type:ignore
    fixed_verts: PointerProperty(type=SimpleVertexGroup)  # type: ignore
    displacement: PointerProperty(type=ScalarVertexMapSettings)  # type:ignore
    laplacian: PointerProperty(type=ScalarVertexMapSettings)  # type:ignore
    alpha: PointerProperty(type=ScalarVertexMapSettings)  # type:ignore
    beta: PointerProperty(type=ScalarVertexMapSettings)  # type:ignore


class SoftenVertexGroupSettings(PropertyGroup):
    group: EnumProperty(
        name="Vertex Group", items=BlendEnums.vertex_groups
    )  # type:ignore
    rings: IntProperty(
        name="Rings", description="Topologoical smoothing distance", min=1, default=5
    )  # type:ignore
    copy: BoolProperty(
        name="Copy", description="Apply to copy", default=True
    )  # type:ignore
    direction: EnumProperty(
        name="Direction", items=[("IN", "Inwards", ""), ("OUT", "Outwards", "")]
    )  # type:ignore


class HardenVertexGroupSettings(PropertyGroup):
    group: EnumProperty(
        name="Vertex Group", items=BlendEnums.vertex_groups
    )  # type:ignore
    copy: BoolProperty(
        name="Copy", description="Apply to copy", default=True
    )  # type:ignore


class RemapVertexGroupSettings(PropertyGroup):
    remap: PointerProperty(type=RemappingStack)  # type:ignore
    group: PointerProperty(type=SimpleVertexGroup)  # type:ignore


class GlobalSettings(PropertyGroup):
    minsrf: PointerProperty(type=MinSrfSettings)  # type: ignore
    flation: PointerProperty(type=FlationSettings)  # type: ignore
    vghard: PointerProperty(type=HardenVertexGroupSettings)  # type:ignore
    vgsoft: PointerProperty(type=SoftenVertexGroupSettings)  # type:ignore
    bake: PointerProperty(type=BakingSettings)  # type:ignore
    imgmap: PointerProperty(type=ImageMappingSettings)  # type:ignore
    vgremap: PointerProperty(type=RemapVertexGroupSettings)  # type:ignore
