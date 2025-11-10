import torch

from typing import Dict, Type

from .config import SolverConfig
from .preconds import (
    Preconditioner,
    JacobiPreconditioner,
    LeftScalingPreconditioner,
)
from .solvers import (
    Result,
    SystemSolver,
    DirectSparseSolver,
    ConjugateGradientSolver,
    BiConjugateGradientSolver,
    BiConjugateGradientStabilizedSolver,
)
from ...singleton import Singleton
from ..sparse_ops import SparseTensor


class Solver(Singleton):

    _solver_classes = {
        "Direct": DirectSparseSolver,
        "Conjugate Gradient": ConjugateGradientSolver,
        "Biconjugate Gradient": BiConjugateGradientSolver,
        "BiCGSTAB": BiConjugateGradientStabilizedSolver,
    }

    _precond_classes = {
        "NONE": Preconditioner,
        "Jacobi": JacobiPreconditioner,
        "Left Scaling": LeftScalingPreconditioner,
    }

    def initialize(self, *args, **kwargs):
        pass

    def solve(self, A: torch.Tensor, b: torch.Tensor, config: SolverConfig) -> Result:
        A = SparseTensor(A)
        solver = self.get_solver(A, b, config)
        return solver.solve()

    def get_solver(
        self, A: SparseTensor, b: torch.Tensor, config: SolverConfig
    ) -> SystemSolver:
        s_name = config.solver
        if s_name == "AUTO":
            return self.derive_solver(A, b, config)
        elif s_name == "Direct":
            return DirectSparseSolver(A, b)
        else:
            solver_cls = Solver._solver_classes.get(s_name, None)
            if solver_cls is None:
                raise ValueError(f"'{s_name}' is not a valid solver class name.")
            precond = self.get_precond(A, b, config, solver_cls)
            return solver_cls(A, b, precond, config.iters, config.tolerance)

    def get_precond(
        self,
        A: SparseTensor,
        b: torch.Tensor,
        config: SolverConfig,
        solver_cls: Type[SystemSolver],
    ) -> Preconditioner:
        p_name = config.precond
        if p_name == "NONE":
            Preconditioner()
        if p_name == "AUTO":
            return self.derive_precond(A, b, config, solver_cls)
        precond_cls = Solver._precond_classes.get(p_name, None)
        if precond_cls is None:
            raise ValueError(f"'{p_name}' is not a valid Preconditioner class.")
        return precond_cls()

    def derive_solver(
        self, A: SparseTensor, b: torch.Tensor, config: SolverConfig
    ) -> SystemSolver:
        symmetric = A.is_symmetric()
        spd = A.is_spd()

        if symmetric and spd:
            precond = self.derive_precond(A, b, ConjugateGradientSolver, config)
            return ConjugateGradientSolver(
                A, b, precond, iters=config.iters, tolerance=config.tolerance
            )
        else:
            precond = self.derive_precond(
                A, b, BiConjugateGradientStabilizedSolver, config
            )
            return BiConjugateGradientStabilizedSolver(
                A, b, precond, iters=config.iters, tolerance=config.tolerance
            )

    def derive_precond(
        self,
        A: SparseTensor,
        b: torch.Tensor,
        solver_cls: Type[SystemSolver],
        config: SolverConfig,
    ) -> Preconditioner:
        if solver_cls == DirectSparseSolver:
            return Preconditioner()
        # Would be nice: find a way to calculate efficiently condition number of A in case it doesn't need a Preconditioner
        return Preconditioner()
