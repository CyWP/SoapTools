import bpy
import torch

from bpy.props import PointerProperty, EnumProperty
from bpy.types import PropertyGroup

from .solver import SolverSettings
from .svm import ScalarVertexMapSettings
from .v_group import SimpleVertexGroup
from ..utils.blend_data.mesh_obj import modifier_items


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


class GlobalSettings(PropertyGroup):
    minsrf: PointerProperty(type=MinSrfSettings)  # type: ignore
    flation: PointerProperty(type=FlationSettings)  # type: ignore
