import bpy
import torch

from bpy.props import EnumProperty, IntProperty, FloatProperty
from bpy.types import PropertyGroup

from ..utils.math.solvers import Solver, SolverConfig


def get_torch_devices():
    """Dynamically list available torch devices."""
    devices = [("cpu", "CPU", "Use the CPU for computation.")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            devices.append((f"cuda:{i}", f"CUDA:{i}", f"GPU {i}: {name}"))
    return devices


class SolverSettings(PropertyGroup):
    device: EnumProperty(
        name="Device",
        description="Compute device used for torch operations",
        items=get_torch_devices(),
    )  # type: ignore

    solver: EnumProperty(
        name="Solver",
        description="Solver used for linear system.",
        items=lambda self, context: Solver().get_solver_options(self.get_config()),
    )  # type: ignore

    precond: EnumProperty(
        name="Precond",
        description="Preconditionning for linear system solver.",
        items=lambda self, context: Solver().get_precond_options(self.get_config()),
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
        default=0,
        precision=8,
        min=0,
    )  # type: ignore

    block_size: IntProperty(
        name="Block Size",
        description="Block size, should match DOF for optimization.",
        default=3,
        min=1,
    )  # type: ignore

    def get_device(self) -> torch.device:
        return torch.device(self.device)

    def get_config(self) -> SolverConfig:
        return SolverConfig(
            solver=self.solver,
            precond=self.precond,
            iters=self.iters,
            tolerance=self.tolerance,
            block_size=self.block_sze,
            device=self.get_device(),
        )

    def draw(self, layout):
        solver = self.solver
        if len(get_torch_devices()) > 1:
            row = layout.row()
            row.prop(self, "device", expand=True)
        row = layout.row()
        row.prop(self, "solver")
        if solver != "AUTO":
            row.prop(self, "precond")
        if solver in ("AUTO", "Conjugate Gradient", "Biconjugate Gradient", "BiCGSTAB"):
            row = layout.row()
            row.prop(self, "iters")
            row.prop(self, "tolerance")
