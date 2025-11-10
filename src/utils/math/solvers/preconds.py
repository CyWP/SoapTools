import torch

from typing import Tuple

from ..sparse_ops import SparseTensor


class Preconditioner:

    def setup(
        self, A: SparseTensor, b: torch.Tensor
    ) -> Tuple[SparseTensor, torch.Tensor]:
        return A, b

    def apply(
        self, A: SparseTensor, r: torch.Tensor
    ) -> Tuple[SparseTensor, torch.Tensor]:
        return A, r


class JacobiPreconditioner(Preconditioner):
    def setup(
        self, A: torch.Tensor, b: torch.Tensor
    ) -> Tuple[SparseTensor, torch.Tensor]:
        # Precompute the inverse diagonal (custom function)
        self.inv_diag = -A.inv_diagonal().to_sparse_csr()
        return A, b

    def apply(
        self, A: torch.Tensor, r: torch.Tensor
    ) -> Tuple[SparseTensor, torch.Tensor]:
        A.csr = self.inv_diag @ A.csr
        r = (self.inv_diag @ r.unsqueeze(1)).squeeze(1)
        return A, r


class LeftScalingPreconditioner(Preconditioner):
    def setup(
        self, A: SparseTensor, b: torch.tensor
    ) -> Tuple[SparseTensor, torch.Tensor]:
        Dinv = A.inv_diagonal().to_sparse_csr()
        A.csr = Dinv @ A.csr
        A.update_coo()
        return A, (Dinv @ b.unsqueeze(1)).squeeze(1)
