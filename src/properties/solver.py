import bpy
import torch

from bpy.props import EnumProperty, IntProperty, FloatProperty
from bpy.types import PropertyGroup
from typing import List, Tuple

from ..utils.math.solvers import Solver, SolverConfig


def get_torch_devices():
    """Dynamically list available torch devices."""
    devices = [("CPU", "CPU", "Use the CPU for computation.")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            devices.append((f"cuda:{i}", f"GPU", f"GPU {i}: {name}"))
    return devices


class SolverSettings(PropertyGroup):
    device: EnumProperty(
        name="Device",
        description="Compute device used for torch operations",
        items=lambda self, context: get_torch_devices(),
        default=0,
    )  # type: ignore

    solver: EnumProperty(
        name="Solver",
        description="Solver used for linear system.",
        items=lambda self, context: self.get_solver_options(),
    )  # type: ignore

    precond: EnumProperty(
        name="Precond",
        description="Preconditionning for linear system solver.",
        items=lambda self, context: self.get_precond_options(),
    )  # type: ignore

    iters: IntProperty(
        name="Iterations",
        description="Solver iterations.",
        default=100,
        min=1,
    )  # type: ignore

    tolerance: FloatProperty(
        name="Tolerance",
        description="Forces solver to cease iterating once error is below tolerance.",
        default=1e-6,
        precision=6,
        min=1e-6,
    )  # type: ignore

    def get_solver_options(self) -> List[Tuple]:
        options = [(name, name, "") for name, cls in Solver()._solver_classes.items()]
        return [("AUTO", "Auto", ""), *options]

    def get_precond_options(self) -> List[Tuple]:
        is_direct = self.solver == "Direct"
        if is_direct:
            return [("NONE", "None", "")]
        options = [(name, name, "") for name, cls in Solver()._precond_classes.items()]
        return [("AUTO", "Auto", ""), *options]

    def get_device(self) -> torch.device:
        return torch.device(self.device if self.device else "cpu")

    def get_config(self) -> SolverConfig:
        return SolverConfig(
            solver=self.solver,
            precond=self.precond,
            iters=self.iters,
            tolerance=self.tolerance,
            device=self.get_device(),
        )

    def draw(self, layout):
        solver = self.solver
        if len(get_torch_devices()) > 1:
            row = layout.row()
            row.prop(self, "device", expand=True)
        row = layout.row()
        row.prop(self, "solver")
        if solver not in ("AUTO", "Direct"):
            row.prop(self, "precond")
        if solver in ("AUTO", "Conjugate Gradient", "Biconjugate Gradient", "BiCGSTAB"):
            row = layout.row()
            row.prop(self, "iters")
            row.prop(self, "tolerance")
