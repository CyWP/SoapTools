import bpy
import torch

from bpy.props import PointerProperty, EnumProperty, IntProperty
from bpy.types import PropertyGroup
from typing import Dict

from .svm import ScalarVertexMapSettings
from .v_group import SimpleVertexGroup
from ..utils.blend_data.mesh_obj import modifier_items


def get_torch_devices():
    """Dynamically list available torch devices."""
    devices = [("cpu", "CPU", "Use the CPU for computation.")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            devices.append((f"cuda:{i}", f"CUDA:{i}", f"GPU {i}: {name}"))
    return devices


class GlobalSettings(PropertyGroup):
    device: bpy.props.EnumProperty(
        name="Device",
        description="Compute device used for torch operations",
        items=get_torch_devices(),
    )  # type: ignore

    def get_device(self) -> torch.device:
        return torch.device(self.device)


class SolverSettings(PropertyGroup):
    device: bpy.props.EnumProperty(
        name="Device",
        description="Compute device used for torch operations",
        items=get_torch_devices(),
    )  # type: ignore

    solver: bpy.props.EnumProperty(
        name="Solver",
        description="Solver used for linear system.",
        items=[],
    )  # type: ignore

    precond: bpy.props.EnumProperty(
        name="Precond",
        description="Preconditionning for linear system solver.",
        items=[],
    )  # type: ignore

    iters: bpy.props.IntProperty(
        name="Iterations",
        description="Solver iterations.",
        default=100,
    )  # type: ignore

    tolerance: bpy.porps.FloatProperty(
        name="Tolerance",
        description="Forces solver to cease iterating once error is below tolerance.",
        default=0,
        precision=8,
    )  # type: ignore

    def get_device(self) -> torch.device:
        return torch.device(self.device)

    def get_dict(self) -> Dict:
        return

    def get_config(self) -> SolverConfig:
        return SolverConfig(self.get_dict())


class MinSrfSettings(PropertyGroup):
    apply_after: EnumProperty(
        name="Apply_after",
        description="Applies modifier and all prior ones in stack before transforming, preserves ones after.",
        items=modifier_items,
    )  # type:ignore
    fixed_verts: PointerProperty(type=SimpleVertexGroup)  # type: ignore


class FlationSettings(PropertyGroup):
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
