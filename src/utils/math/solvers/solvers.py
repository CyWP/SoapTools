import torch

from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as spla
from typing import Tuple

from .preconds import Preconditioner
from ..dense_ops import batched_dot
from ...easydict import EasyDict


class Result(EasyDict):

    _defaults = dict(result=None, converged=False, err=None)

    def __init__(self, **kwargs):
        data = dict(Result._defaults)
        data.update(**kwargs)
        super().__init__(**data)


class SystemSolver:

    name: str = "Solver"

    def __init__(self):
        self._result = Result()
        self.validate()
        self.setup()

    def solve(self) -> Result:
        try:
            res, converged, err = self.solve_system()
            return Result(result=res, converged=converged, err=err)
        except Exception as e:
            return Result(result=None, converged=False, err=e)

    def validate(
        self,
    ) -> bool:
        return True

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def solve_system(self) -> Tuple[torch.Tensor, bool, torch.Tensor]:
        pass


class DirectSparseSolver(SystemSolver):
    """
    Direct sparse solver using SciPy's spsolve.
    Supports torch.sparse_coo and torch.sparse_csr inputs.
    """

    name = "Direct"

    def __init__(self, A: torch.Tensor, b: torch.Tensor, device=None):
        self.A = A
        self.b = b
        self.device = device if device is not None else b.device
        super().__init__()

    def setup(self):
        # No preconditioner needed for direct solver
        pass

    def solve_system(self) -> Tuple[torch.Tensor, bool, float]:
        # Convert A to CSR
        if self.A.layout == torch.sparse_coo:
            row, col = self.A.indices()
            data = self.A.values()
            A_csr = csr_matrix(
                (data.cpu().numpy(), (row.cpu().numpy(), col.cpu().numpy())),
                shape=self.A.shape,
            )
        elif self.A.layout == torch.sparse_csr:
            data = self.A.values().cpu().numpy()
            indices = self.A.col_indices().cpu().numpy()
            indptr = self.A.crow_indices().cpu().numpy()
            A_csr = csr_matrix((data, indices, indptr), shape=self.A.shape)
        else:
            raise ValueError(f"Unsupported sparse layout {self.A.layout}")

        # Convert b
        b_cpu = self.b.cpu().numpy()

        # Solve
        try:
            x_cpu = spla.spsolve(A_csr, b_cpu)
        except Exception as e:
            return torch.zeros_like(self.b), False, float("inf")

        x = torch.tensor(x_cpu, device=self.device, dtype=self.b.dtype)
        # Compute residual norm
        r = self.A @ x.unsqueeze(1)
        r = r.squeeze(1) - self.b
        err = torch.linalg.norm(r)
        return x, True, err


class ConjugateGradientSolver(SystemSolver):

    name: str = "Conjugate Gradient"

    def __init__(
        self,
        A: torch.Tensor,
        b: torch.tensor,
        precond: Preconditioner,
        iters: int = 100,
        tolerance: float = 0.0,
    ):
        self.A = A
        self.b = b
        self.precond = precond
        self.iters = iters
        self.tolerance = tolerance
        super().__init__()

    def setup(self):
        self.A, self.b = self.precond.setup(self.A, self.b)
        self.x = torch.zeros_like(self.b)
        self.r = self.b - (self.A @ self.x.unsqueeze(1)).squeeze(1)
        self.A, self.r = self.precond.apply(self.A, self.r)
        self.p = self.r.clone()
        self.rs_old = batched_dot(self.r, self.r)

    def solve_system(self) -> Tuple[torch.Tensor, bool, torch.Tensor]:
        rs_new = self.rs_old
        for _ in range(self.iters):
            Ap = (self.A @ self.p.unsqueeze(1)).squeeze(1)
            alpha = self.rs_old / batched_dot(self.p, Ap)
            self.x += alpha * self.p
            self.r -= alpha * Ap
            self.A, self.r = self.precond.apply(self.A, self.r)
            rs_new = batched_dot(self.r, self.r)
            if torch.sqrt(rs_new) < self.tolerance:
                return self.x, True, torch.sqrt(rs_new).detach()
            self.p = self.r + (rs_new / self.rs_old) * self.p
            self.rs_old = rs_new

        return self.x, False, torch.sqrt(rs_new).detach()


class BiConjugateGradientSolver(SystemSolver):
    name: str = "BiConjugate Gradient"

    def __init__(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        precond: Preconditioner,
        iters: int = 100,
        tolerance: float = 0.0,
    ):
        self.A = A
        self.b = b
        self.precond = precond
        self.iters = iters
        self.tolerance = tolerance
        super().__init__()

    def setup(self):
        # Preconditioning setup
        self.A, self.b = self.precond.setup(self.A, self.b)

        # Initial values
        self.x = torch.zeros_like(self.b)
        self.r = self.b - (self.A @ self.x.unsqueeze(1)).squeeze(1)
        self.r_hat = self.r.clone()  # arbitrary, often equal to r₀
        self.A, self.r = self.precond.apply(self.A, self.r)
        self.A, self.r_hat = self.precond.apply(self.A, self.r_hat)

        self.p = self.r.clone()
        self.p_hat = self.r_hat.clone()
        self.rho_old = batched_dot(self.r_hat, self.r)

    def solve_system(self) -> Tuple[torch.Tensor, bool, float]:
        for _ in range(self.iters):
            Ap = (self.A @ self.p.unsqueeze(1)).squeeze(1)
            ATp_hat = (self.A.transpose(-1, -2) @ self.p_hat.unsqueeze(1)).squeeze(1)

            denom = batched_dot(self.p_hat, Ap)
            # prevent divide by zero
            denom = torch.where(
                denom == 0, torch.tensor(1e-20, device=denom.device), denom
            )
            alpha = self.rho_old / denom

            self.x += alpha * self.p
            self.r -= alpha * Ap
            self.r_hat -= alpha * ATp_hat

            self.A, self.r = self.precond.apply(self.A, self.r)
            self.A, self.r_hat = self.precond.apply(self.A, self.r_hat)

            rho_new = batched_dot(self.r_hat, self.r)
            if torch.sqrt(rho_new.abs()) < self.tolerance:
                return self.x, True, torch.sqrt(rho_new.abs())

            beta = rho_new / self.rho_old
            self.p = self.r + beta * self.p
            self.p_hat = self.r_hat + beta * self.p_hat
            self.rho_old = rho_new

        return self.x, False, torch.sqrt(rho_new.abs())


class BiConjugateGradientStabilizedSolver(SystemSolver):
    name: str = "BiCGSTAB"

    def __init__(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        precond: Preconditioner,
        iters: int = 100,
        tolerance: float = 1e-6,
    ):
        self.A = A
        self.b = b
        self.precond = precond
        self.iters = iters
        self.tolerance = tolerance
        super().__init__()

    def setup(self):
        # Setup preconditioner (may modify A, b)
        self.A, self.b = self.precond.setup(self.A, self.b)

        # Initialize
        self.x = torch.zeros_like(self.b)
        self.r = self.b - (self.A @ self.x.unsqueeze(1)).squeeze(1)
        self.r_hat = self.r.clone()  # fixed shadow residual
        self.rho_old = torch.ones_like(batched_dot(self.r_hat, self.r))
        self.alpha = torch.zeros_like(self.rho_old)
        self.omega = torch.ones_like(self.rho_old)
        self.v = torch.zeros_like(self.b)
        self.p = torch.zeros_like(self.b)

    def solve_system(self) -> Tuple[torch.Tensor, bool, float]:
        for _ in range(self.iters):
            rho_new = batched_dot(self.r_hat, self.r)
            if torch.any(rho_new == 0):
                break

            beta = (rho_new / self.rho_old) * (self.alpha / self.omega)
            self.p = self.r + beta * (self.p - self.omega * self.v)

            # Apply preconditioner
            self.A, self.p = self.precond.apply(self.A, self.p)
            self.v = (self.A @ self.p.unsqueeze(1)).squeeze(1)

            denom = batched_dot(self.r_hat, self.v)
            denom = torch.where(
                denom == 0, torch.tensor(1e-20, device=denom.device), denom
            )
            self.alpha = rho_new / denom

            s = self.r - self.alpha * self.v
            if torch.linalg.norm(s) < self.tolerance:
                self.x += self.alpha * self.p
                return self.x, True, torch.linalg.norm(s)

            # Apply preconditioner again
            self.A, s = self.precond.apply(self.A, s)
            t = (self.A @ s.unsqueeze(1)).squeeze(1)

            t_dot_t = batched_dot(t, t)
            omega = batched_dot(t, s) / t_dot_t

            self.x += self.alpha * self.p + omega * s
            self.r = s - omega * t

            err = torch.linalg.norm(self.r)
            if err < self.tolerance or omega == 0:
                return self.x, True, err

            self.rho_old = rho_new
            self.omega = omega

        return self.x, False, torch.linalg.norm(self.r)
