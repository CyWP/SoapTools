import bpy
import torch

from bpy.props import EnumProperty, IntProperty, FloatProperty, PointerProperty
from bpy.types import PropertyGroup
from typing import List, Tuple

from ..utils.math.solvers import Solver, SolverConfig


class SolverSettings(PropertyGroup):
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

    def get_config(self, device: torch.device) -> SolverConfig:
        return SolverConfig(
            solver=self.solver,
            precond=self.precond,
            iters=self.iters,
            tolerance=self.tolerance,
            device=device,
        )

    def draw(self, layout):
        box = layout.box()
        row = box.row()
        row.alignment = "CENTER"
        row.enabled = False
        row.label(text="Solver")
        solver = self.solver
        row = box.row()
        row.prop(self, "solver")
        if solver not in ("AUTO", "Direct"):
            row.prop(self, "precond")
        if solver in ("AUTO", "Conjugate Gradient", "Biconjugate Gradient", "BiCGSTAB"):
            row = box.row()
            row.prop(self, "iters")
            row.prop(self, "tolerance")
