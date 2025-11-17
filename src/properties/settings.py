import bpy
import torch

from bpy.props import PointerProperty, EnumProperty, IntProperty, BoolProperty
from bpy.types import PropertyGroup

from .baking import BakingSettings
from .solver import SolverSettings
from .svm import ScalarVertexMapSettings
from .v_group import SimpleVertexGroup
from ..utils.blend_data.mesh_obj import modifier_items
from ..utils.blend_data.vertex_groups import vertex_group_items


class MinSrfSettings(PropertyGroup):
    solver: PointerProperty(type=SolverSettings)  # type:ignore
    apply_after: EnumProperty(
        name="Apply_after",
        description="Applies modifier and all prior ones in stack before transforming, preserves ones after.",
        items=modifier_items,
    )  # type:ignore
    fixed_verts: PointerProperty(type=SimpleVertexGroup)  # type: ignore


class FlationSettings(PropertyGroup):
    solver: PointerProperty(type=SolverSettings)  # type: ignore
    apply_after: EnumProperty(
        name="Apply_after",
        description="Applies modifier and all prior ones in stack before transforming, preserves ones after.",
        items=modifier_items,
    )  # type:ignore
    active_constraint: EnumProperty(
        name="Constraint",
        description="Constraint to be currently edited",
        items=[
            ("DISPLACEMENT", "Displacement", ""),
            ("LAPLACIAN", "Laplacian", ""),
            ("ALPHA", "Alpha", ""),
            ("BETA", "Beta", ""),
        ],
    )  # type:ignore
    fixed_verts: PointerProperty(type=SimpleVertexGroup)  # type: ignore
    displacement: PointerProperty(type=ScalarVertexMapSettings)  # type:ignore
    laplacian: PointerProperty(type=ScalarVertexMapSettings)  # type:ignore
    alpha: PointerProperty(type=ScalarVertexMapSettings)  # type:ignore
    beta: PointerProperty(type=ScalarVertexMapSettings)  # type:ignore


class SoftenVertexGroupSettings(PropertyGroup):
    group: EnumProperty(name="Vertex Group", items=vertex_group_items)  # type:ignore
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
    group: EnumProperty(name="Vertex Group", items=vertex_group_items)  # type:ignore
    copy: BoolProperty(
        name="Copy", description="Apply to copy", default=True
    )  # type:ignore


class GlobalSettings(PropertyGroup):
    minsrf: PointerProperty(type=MinSrfSettings)  # type: ignore
    flation: PointerProperty(type=FlationSettings)  # type: ignore
    vghard: PointerProperty(type=HardenVertexGroupSettings)  # type:ignore
    vgsoft: PointerProperty(type=SoftenVertexGroupSettings)  # type:ignore
    bake: PointerProperty(type=BakingSettings)  # type:ignore
