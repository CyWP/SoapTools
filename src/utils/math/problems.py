import torch

from typing import Optional

from .solvers import SolverConfig, Solver
from .sparse_ops import sparse_cotan_laplacian, sparse_eye, sparse_kron, sparse_mask


def solve_minimal_surface(
    config: SolverConfig,
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
    solver = Solver()
    for dim in range(3):
        b = rhs[:, dim]
        result = solver.solve(L_II, b, config)
        if isinstance(result.err, Exception):
            raise result.err
        x_free = result.result
        V_new[free_idx, dim] = x_free
    V_new[fixed_idx] = fixed_pos

    return V_new


def solve_flation(
    config: SolverConfig,
    V: torch.Tensor,
    F: torch.Tensor,
    N: torch.Tensor,
    target_offset: torch.Tensor,
    fixed_idx: torch.Tensor,
    lambda_lap: torch.Tensor,
    beta_normal: torch.Tensor,
    alpha_tangent: torch.Tensor,
) -> torch.Tensor:
    """
    Sparse solve of vertex displacement along normals with Laplacian smoothing constraints.
    """
    n = V.shape[0]
    device = V.device

    assert fixed_idx.ndim == 1 and fixed_idx.dtype == torch.long

    is_fixed = torch.zeros(n, dtype=torch.bool, device=device)
    is_fixed[fixed_idx] = True
    not_fixed = ~is_fixed
    free_idx = torch.where(not_fixed)[0]
    if len(free_idx) == 0:
        return V

    assert target_offset.shape == (n,), target_offset.shape
    assert lambda_lap.shape == (n,), lambda_lap.shape
    assert alpha_tangent.shape == (n,), alpha_tangent.shape
    assert beta_normal.shape == (n,), beta_normal.shape

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
    d_free = Solver().solve(A3_free, b_free, config).result
    # Reassemble displacement
    D = torch.zeros_like(V)
    D[free_idx] = d_free.view(-1, 3)
    return V + D
