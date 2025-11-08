import torch

from typing import Tuple


class Preconditioner:

    def setup(
        self, A: torch.Tensor, b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return A, b

    def apply(
        self, A: torch.Tensor, r: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return A, r


class JacobiPreconditioner(Preconditioner):
    def setup(self, A, b):
        self.inv_diag = 1.0 / A.diagonal()
        return A, b

    def apply(self, A, r):
        return A, self.inv_diag * r


class BlockJacobiPreconditioner(Preconditioner):
    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size

    def setup(self, A, b):
        n = A.shape[0]
        self.blocks = []
        for i in range(0, n, self.block_size):
            end = min(i + self.block_size, n)
            self.blocks.append(torch.linalg.inv(A[i:end, i:end]))
        return A, b

    def apply(self, A, r):
        out = torch.zeros_like(r)
        for i, B_inv in enumerate(self.blocks):
            s = i * self.block_size
            e = s + B_inv.shape[0]
            out[s:e] = B_inv @ r[s:e]
        return A, out


class LeftScalingPreconditioner(Preconditioner):
    def setup(self, A, b):
        Dinv = torch.diag(1.0 / A.diagonal())
        return Dinv @ A, Dinv @ b


class SymmetricScalingPreconditioner(Preconditioner):
    def setup(self, A, b):
        d = A.diagonal().abs().clamp_min(1e-12)
        self.Dinv_sqrt = torch.diag(1.0 / torch.sqrt(d))
        A_scaled = self.Dinv_sqrt @ A @ self.Dinv_sqrt
        b_scaled = self.Dinv_sqrt @ b
        return A_scaled, b_scaled

    def apply(self, A, r):
        return A, self.Dinv_sqrt @ r


class ApproximateInversePreconditioner(Preconditioner):
    def setup(self, A, b, iters=2):
        diag = A.diagonal().clamp_min(1e-12)
        M_inv = torch.diag(1.0 / diag)
        # Perform iterative refinement of the inverse
        for _ in range(iters):
            M_inv = M_inv @ (2 * torch.eye(A.size(0), device=A.device) - A @ M_inv)
        self.M_inv = M_inv
        return A, b

    def apply(self, A, r):
        return A, self.M_inv @ r


class LUPreconditioner(Preconditioner):
    def setup(self, A, b):
        self.L, self.U = torch.linalg.lu(A)
        return A, b

    def apply(self, A, r):
        y = torch.linalg.solve_triangular(self.L, r, upper=False)
        z = torch.linalg.solve_triangular(self.U, y, upper=True)
        return A, z


class IterativePreconditioner(Preconditioner):
    def __init__(self, inner_iters=3):
        super().__init__()
        self.inner_iters = inner_iters

    def setup(self, A, b):
        self.diag = A.diagonal().clamp_min(1e-12)
        return A, b

    def apply(self, A, r):
        x = torch.zeros_like(r)
        for _ in range(self.inner_iters):
            x = x + (r - (A @ x)) / self.diag
        return A, x
