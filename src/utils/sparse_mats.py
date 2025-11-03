import torch

# RETURN EVERYTHING AS COO


def sparse_cotan_laplacian(
    V: torch.Tensor, F: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    assert len(F.shape) == 2 and F.shape[1] == 3
    assert F.dtype == torch.long
    assert V.device == F.device

    n = V.shape[0]
    device = V.device

    i, j, k = F[:, 0], F[:, 1], F[:, 2]
    vi, vj, vk = V[i], V[j], V[k]

    v_ji = vj - vi
    v_ki = vk - vi
    v_ij = vi - vj
    v_kj = vk - vj
    v_ik = vi - vk
    v_jk = vj - vk

    cot_i = (v_ji * v_ki).sum(axis=1) / (
        torch.norm(torch.cross(v_ji, v_ki), dim=1) + eps
    )
    cot_j = (v_ij * v_kj).sum(axis=1) / (
        torch.norm(torch.cross(v_ij, v_kj), dim=1) + eps
    )
    cot_k = (v_ik * v_jk).sum(axis=1) / (
        torch.norm(torch.cross(v_ik, v_jk), dim=1) + eps
    )

    W = torch.cat([cot_i, cot_i, cot_j, cot_j, cot_k, cot_k]) * 0.5
    I = torch.cat([j, k, k, i, i, j])
    J = torch.cat([k, j, i, k, j, i])

    L = torch.sparse_coo_tensor(
        torch.stack([I, J]), W, (n, n), device=device
    ).coalesce()

    # Compute diagonal: sum of weights per row
    diag = torch.zeros(n, device=device)
    diag = diag.scatter_add(0, I, W)

    M = torch.sparse_coo_tensor(
        torch.stack([torch.arange(n), torch.arange(n)]), diag, (n, n), device=device
    ).coalesce()

    return (L - M).coalesce()


def sparse_eye(
    size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    indices = torch.arange(size, device=device, dtype=torch.long)
    indices = torch.stack([indices, indices], dim=0)  # shape: [2, size]
    values = torch.ones((size,), device=device, dtype=dtype)
    return torch.sparse_coo_tensor(
        indices, values, (size, size), device=device, dtype=dtype
    ).coalesce()


def sparse_kron(A: torch.Tensor, B: torch.Tensor):
    assert A.layout == torch.sparse_coo and B.layout == torch.sparse_coo

    iA, jA = A.indices()
    vA = A.values()
    nA, mA = A.shape

    iB, jB = B.indices()
    vB = B.values()
    nB, mB = B.shape

    new_i = (iA[:, None] * nB + iB[None, :]).reshape(-1)
    new_j = (jA[:, None] * mB + jB[None, :]).reshape(-1)
    new_values = (vA[:, None] * vB[None, :]).reshape(-1)

    new_indices = torch.stack([new_i, new_j], dim=0)
    new_shape = (nA * nB, mA * mB)

    return torch.sparse_coo_tensor(
        new_indices, new_values, new_shape, dtype=A.dtype, device=A.device
    ).coalesce()


def sparse_mask(
    x: torch.Tensor, row_mask: torch.Tensor, col_mask: torch.Tensor
) -> torch.Tensor:
    """
    Mask rows and columns of a 2D sparse COO matrix `x` using separate boolean masks.

    Keeps only entries (i,j) where row_mask[i] and col_mask[j] are True.
    """
    assert x.layout == torch.sparse_coo, "x must be a sparse COO tensor"
    assert row_mask.dim() == 1 and col_mask.dim() == 1, "Masks must be 1D"

    i = x.indices()  # shape (2, nnz)
    v = x.values()
    rows, cols = i[0], i[1]

    keep = row_mask[rows] & col_mask[cols]
    new_rows = torch.nonzero(row_mask, as_tuple=False).flatten()
    new_cols = torch.nonzero(col_mask, as_tuple=False).flatten()

    # remap row and column indices to new compacted indices
    remap_rows = -torch.ones_like(row_mask, dtype=torch.long)
    remap_cols = -torch.ones_like(col_mask, dtype=torch.long)
    remap_rows[row_mask] = torch.arange(new_rows.numel(), device=x.device)
    remap_cols[col_mask] = torch.arange(new_cols.numel(), device=x.device)

    new_i = torch.stack(
        [
            remap_rows[rows[keep]],
            remap_cols[cols[keep]],
        ]
    )
    new_v = v[keep]
    new_shape = (row_mask.sum().item(), col_mask.sum().item())

    return torch.sparse_coo_tensor(
        new_i, new_v, new_shape, device=x.device, dtype=x.dtype
    ).coalesce()
