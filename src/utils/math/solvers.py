import torch
import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix

from typing import Union, Optional

from .sparse_ops import sparse_cotan_laplacian, sparse_eye, sparse_kron, sparse_mask


def conjugate_gradient(
    A: torch.Tensor, b: torch.Tensor, tol: float = 1e-8, max_iter: int = 500
) -> torch.Tensor:
    """
    Solve Ax = b using Conjugate Gradient (CG) method.
    Works for symmetric positive definite (SPD) sparse matrices.
    Supports CUDA and CPU.
    """
    x = torch.zeros_like(b)
    # r = b - torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    r = b - (A @ x.unsqueeze(1)).squeeze(1)
    p = r.clone()
    rs_old = torch.dot(r, r)

    for _ in range(max_iter):
        # Ap = torch.sparse.mm(A, p.unsqueeze(1)).squeeze(1)
        Ap = (A @ p.unsqueeze(1)).squeeze(1)
        alpha = rs_old / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x


def sparse_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve Ax = b for sparse A.
    """
    device = b.device

    if device.type == "cuda":
        x = conjugate_gradient(A, b)
        return x
    else:
        if A.layout == torch.sparse_coo:
            row, col = A.indices()
            data = A.values()
            A_csr = csr_matrix(
                (data.cpu().numpy(), (row.cpu().numpy(), col.cpu().numpy())),
                shape=A.shape,
            )
        elif A.layout == torch.sparse_csr:
            data = A.values().cpu().numpy()
            indices = A.col_indices().cpu().numpy()
            indptr = A.crow_indices().cpu().numpy()
            A_csr = csr_matrix((data, indices, indptr), shape=A.shape)
        else:
            raise ValueError("Unsupported sparse layout")

        b_cpu = b.cpu().numpy()
        x_cpu = spla.spsolve(A_csr, b_cpu)
        return torch.tensor(x_cpu, device=device, dtype=b.dtype)


def solve_minimal_surface(
    V: torch.Tensor,
    F: torch.Tensor,
    fixed_idx: torch.Tensor,
    fixed_pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    n = V.shape[0]
    device = V.device
    L = sparse_cotan_laplacian(V, F)

    if fixed_pos is None:
        fixed_pos = V[fixed_idx]

    is_fixed = torch.zeros((n,), dtype=torch.bool, device=device)
    is_fixed[fixed_idx] = True
    not_fixed = ~is_fixed
    free_idx = torch.where(not_fixed)[0]
    if len(free_idx) == 0:
        return V.clone()

    L_II = sparse_mask(L, not_fixed, not_fixed)
    L_IB = sparse_mask(L, not_fixed, is_fixed)

    X_B = fixed_pos
    rhs = torch.sparse.mm(-L_IB, X_B)

    V_new = V
    for dim in range(3):
        b = rhs[:, dim]
        x_free = sparse_solve(L_II.to_sparse_csr(), b)
        V_new[free_idx, dim] = x_free
    V_new[fixed_idx] = fixed_pos

    return V_new


def solve_flation(
    V: torch.Tensor,
    F: torch.Tensor,
    N: torch.Tensor,
    target_offset: Union[float, torch.Tensor],
    fixed_idx: Optional[torch.Tensor] = None,
    lambda_lap: Union[float, torch.Tensor] = 1.0,
    beta_normal: Union[float, torch.Tensor] = 1.0,
    alpha_tangent: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Sparse solve of vertex displacement along normals with Laplacian smoothing constraints.
    """
    n = V.shape[0]
    device = V.device

    if fixed_idx is None:
        fixed_idx = torch.tensor([], dtype=torch.long, device=device)

    is_fixed = torch.zeros(n, dtype=torch.bool, device=device)
    is_fixed[fixed_idx] = True
    not_fixed = ~is_fixed
    free_idx = torch.where(not_fixed)[0]
    if len(free_idx) == 0:
        return V

    if isinstance(target_offset, float) or target_offset.ndim == 0:
        target_offset = torch.full((n,), target_offset, device=device, dtype=V.dtype)
    if isinstance(lambda_lap, float) or lambda_lap.ndim == 0:
        lambda_lap = torch.full((n,), lambda_lap, device=device, dtype=V.dtype)
    if isinstance(alpha_tangent, float) or alpha_tangent.ndim == 0:
        alpha_tangent = torch.full((n,), alpha_tangent, device=device, dtype=V.dtype)
    if isinstance(beta_normal, float) or beta_normal.ndim == 0:
        beta_normal = torch.full((n,), beta_normal, device=device, dtype=V.dtype)

    L = sparse_cotan_laplacian(V, F)  # sparse CSR (n x n)
    L2 = torch.sparse.mm(L.transpose(0, 1), L)
    # 3D block-diagonal Laplacian: kron(I3, L)
    A3_lap = sparse_kron(L2, sparse_eye(3, device=device))
    lambda_lap = lambda_lap.repeat_interleave(3)
    idx = A3_lap.indices()
    val = A3_lap.values()
    i, j = idx[0], idx[1]
    new_vals = (lambda_lap[i] + lambda_lap[j]) * val
    # Normal projection
    P = N[:, :, None] @ N[:, None, :]  # (n,3,3)
    I3 = torch.eye(3, device=device).expand(n, 3, 3)
    # Weighted blocks
    alpha = alpha_tangent[:, None, None]  # (n,1,1)
    beta = beta_normal[:, None, None]  # (n,1,1)
    Q_blocks = alpha * (I3 - P) + beta * P  # (n,3,3)
    # Flatten block values
    values = Q_blocks.reshape(-1)  # (9*n,)
    # Compute row/col indices for sparse COO
    block_size = 3
    offsets = torch.arange(block_size, device=device)  # [0,1,2]
    rows_block, cols_block = torch.meshgrid(offsets, offsets, indexing="ij")
    rows_block = rows_block.reshape(-1)  # (9,)
    cols_block = cols_block.reshape(-1)  # (9,)
    # Block offsets for n blocks
    block_offsets = torch.arange(n, device=device) * block_size  # (n,)
    # Broadcast to all blocks
    rows = rows_block[None, :] + block_offsets[:, None]  # (n, 9)
    cols = cols_block[None, :] + block_offsets[:, None]  # (n, 9)
    # Flatten to (9*n,)
    rows = rows.reshape(-1)
    cols = cols.reshape(-1)
    indices = torch.stack([rows, cols], dim=0)  # shape [2, 9*n]
    Q = torch.sparse_coo_tensor(indices, values, (3 * n, 3 * n), device=device)
    A3 = torch.sparse_coo_tensor(idx, new_vals, A3_lap.shape) + Q
    # RHS
    b = torch.zeros(3 * n, device=device, dtype=V.dtype)
    b[0::3] = beta_normal * target_offset * N[:, 0]
    b[1::3] = beta_normal * target_offset * N[:, 1]
    b[2::3] = beta_normal * target_offset * N[:, 2]
    # Mask fixed vertices
    mask = not_fixed.repeat_interleave(3)
    A3_free = sparse_mask(A3.coalesce(), mask, mask)
    b_free = b[mask]
    # Solve sparse system
    d_free = sparse_solve(A3_free.to_sparse_csr(), b_free)
    # Reassemble displacement
    D = torch.zeros_like(V)
    D[free_idx] = d_free.view(-1, 3)
    return V + D
