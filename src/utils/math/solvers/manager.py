import torch

from typing import List, Tuple, Type

from .config import SolverConfig
from .preconds import (
    Preconditioner,
    JacobiPreconditioner,
    BlockJacobiPreconditioner,
    LeftScalingPreconditioner,
    IterativePreconditioner,
    SymmetricScalingPreconditioner,
    ApproximateInversePreconditioner,
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


class Solver(Singleton):

    _solver_classes = {
        "Direct": DirectSparseSolver,
        "Conjugate Gradient": ConjugateGradientSolver,
        "Biconjugate Gradient": BiConjugateGradientSolver,
        "BiCGSTAB": BiConjugateGradientStabilizedSolver,
    }

    _precond_classes = {
        "None": Preconditioner,
        "Jacobi": JacobiPreconditioner,
        "Block Jacobi": BlockJacobiPreconditioner,
        "Left Scaling": LeftScalingPreconditioner,
        "Iterative": IterativePreconditioner,
        "Symmetric": SymmetricScalingPreconditioner,
        "Approx Inverse": ApproximateInversePreconditioner,
    }

    def get_solver_options(self, config: SolverConfig) -> List[Tuple]:
        options = [(name, name, "") for name, cls in Solver._solver_classes.items()]
        return [("AUTO", "Auto", ""), *options]

    def get_precond_options(self, config: SolverConfig) -> List[Tuple]:
        is_direct = config.solver == "Direct"
        if is_direct:
            return [("NONE", "None", "")]
        options = [(name, name, "") for name, cls in Solver._solver_classes.items()]
        return [("AUTO", "Auto", ""), *options]

    def solve(self, A: torch.Tensor, b: torch.Tensor, config: SolverConfig) -> Result:
        solver = self.get_solver(A, b, config)
        return solver.solve()

    def get_solver(
        self, A: torch.Tensor, b: torch.Tensor, config: SolverConfig
    ) -> SystemSolver:
        s_name = config.solver
        if s_name == "AUTO":
            return self.derive_solver(A, b, config)
        elif s_name == "DIRECT":
            return DirectSparseSolver(A, b)
        else:
            solver_cls = Solver._solver_classes.get(s_name, None)
            if solver_cls is None:
                raise ValueError(f"'{s_name}' is not a valid solver class name.")
            precond = self.get_precond(A, b, config, solver_cls)
            return solver_cls(A, b, precond, config.iters, config.tolerance)

    def get_precond(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        config: SolverConfig,
        solver_cls: Type[SystemSolver],
    ) -> Preconditioner:
        p_name = config.precond
        if p_name == "AUTO":
            return self.derive_precond(A, b, config, solver_cls)
        elif p_name == "Block Jacobi":
            return BlockJacobiPreconditioner(config.block_size)
        precond_cls = Solver._precond_classes.get(p_name, None)
        if precond_cls is None:
            raise ValueError(f"'{p_name}' is not a valid Preconditioner class.")
        return precond_cls()

    def derive_solver(
        self, A: torch.Tensor, b: torch.Tensor, config: SolverConfig
    ) -> SystemSolver:
        symmetric = torch.allclose(A, A.T)
        eig_min = torch.linalg.eigvals(A).real.min()

        if symmetric and eig_min > 0:
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
        A: torch.Tensor,
        b: torch.Tensor,
        solver_cls: Type[SystemSolver],
        config: SolverConfig,
    ) -> Preconditioner:
        if solver_cls == DirectSparseSolver:
            return Preconditioner()
        cond = torch.linalg.cond(A).real
        if cond < 1e3:
            return Preconditioner()
        sparse = A.is_sparse
        return (
            BlockJacobiPreconditioner(config.block_size)
            if sparse
            else JacobiPreconditioner()
        )
