import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def cotangent_laplacian(
    V: np.ndarray, F: np.ndarray, eps: float = 1e-12
) -> sp.csr_matrix:
    assert len(F.shape) == 2 and F.shape[1] == 3
    n = V.shape[0]

    i, j, k = F[:, 0], F[:, 1], F[:, 2]
    vi, vj, vk = V[i], V[j], V[k]

    v_ji = vj - vi
    v_ki = vk - vi
    v_ij = vi - vj
    v_kj = vk - vj
    v_ik = vi - vk
    v_jk = vj - vk

    cot_i = (v_ji * v_ki).sum(axis=1) / (
        np.linalg.norm(np.cross(v_ji, v_ki), axis=1) + eps
    )
    cot_j = (v_ij * v_kj).sum(axis=1) / (
        np.linalg.norm(np.cross(v_ij, v_kj), axis=1) + eps
    )
    cot_k = (v_ik * v_jk).sum(axis=1) / (
        np.linalg.norm(np.cross(v_ik, v_jk), axis=1) + eps
    )

    W = np.concatenate([cot_i, cot_i, cot_j, cot_j, cot_k, cot_k]) * 0.5
    I = np.concatenate([j, k, k, i, i, j])
    J = np.concatenate([k, j, i, k, j, i])

    L = sp.coo_matrix((W, (I, J)), shape=(n, n))
    L = (L + L.T) * 0.5  # force symmetry
    # set diagonal L_ii = -sum_{j != i} L_ij
    L = -sp.csgraph.laplacian(L, normed=False)
    return L.tocsr()


def solve_minimal_surface_with_fixed(V, F, fixed_idx, fixed_pos=None, solver="direct"):
    """
    Solve for minimal surface (harmonic coordinates) with given fixed vertices.
    V: (n,3) initial vertex positions (used for fixed positions if fixed_pos is None)
    F: (m,3) faces
    fixed_idx: array-like indices of fixed vertices
    fixed_pos: (len(fixed_idx),3) positions for fixed vertices; if None, use V[fixed_idx]
    solver: 'direct' (spsolve) or 'cg' (conjugate gradient)
    Returns: V_new (n,3)
    """
    n = V.shape[0]
    L = cotangent_laplacian(V, F)

    fixed_idx = np.asarray(fixed_idx, dtype=np.int64)
    if fixed_pos is None:
        fixed_pos = V[fixed_idx]

    is_fixed = np.zeros(n, dtype=bool)
    is_fixed[fixed_idx] = True
    free_idx = np.where(~is_fixed)[0]
    if free_idx.size == 0:
        return V.copy()  # nothing to solve

    # Partition matrices
    L_II = L[free_idx[:, None], free_idx]  # sparse
    L_IB = L[free_idx[:, None], fixed_idx]  # sparse

    # RHS = -L_IB * X_B
    X_B = np.asarray(fixed_pos)
    rhs = -L_IB.dot(X_B)  # (n_free, 3) dense

    V_new = V.copy()
    # Solve separately for x,y,z
    for dim in range(3):
        b = rhs[:, dim]
        if solver == "direct":
            # convert to CSR if not already
            x_free = spla.spsolve(L_II, b)
        elif solver == "cg":
            x_free, info = spla.cg(L_II, b, atol=1e-10)
            if info != 0:
                raise RuntimeError("CG did not converge, info=" + str(info))
        else:
            raise ValueError("Unknown solver: " + str(solver))
        V_new[free_idx, dim] = x_free
    # fixed vertices get fixed_pos
    V_new[fixed_idx] = fixed_pos
    return V_new


def solve_inflation(
    V,
    F,
    N,
    fixed_idx,
    target_offset=None,
    lambda_lap=1.0,
    alpha_tangent=0.0,
    beta_normal=1.0,
    solver="direct",
):
    """
    Solve for vertex displacements that approximately move vertices along their normals
    toward a target normal offset distance, while regularizing with the Laplacian.

    Parameters
    ----------
    V : (n,3) array
        Original vertex positions.
    F : (m,3) array
        Triangle indices.
    N : (n,3) array
        Unit vertex normals.
    fixed_idx : list[int]
        Indices of fixed vertices (will remain at their original position).
    target_offset : (n,) or scalar or None
        Desired signed offset distance along normal direction (n_i).
        If None, defaults to zero (minimal surface behavior).
    lambda_lap : float
        Weight for Laplacian smoothness term.
    alpha_tangent : float
        Weight for penalizing tangential motion (discourages deviation from normal direction).
    beta_normal : float
        Weight for enforcing target normal offset distance.
    solver : {'direct','cg'}
        Linear solver type.

    Returns
    -------
    V_new : (n,3) array
        Updated vertex positions.
    """

    n = V.shape[0]
    fixed_idx = np.asarray(fixed_idx, dtype=np.int64)
    is_fixed = np.zeros(n, dtype=bool)
    is_fixed[fixed_idx] = True
    free_idx = np.where(~is_fixed)[0]
    if len(free_idx) == 0:
        return V.copy()

    if np.isscalar(target_offset) or target_offset is None:
        target_offset = np.full(n, target_offset if target_offset is not None else 0.0)

    # Laplacian
    L = cotangent_laplacian(V, F)

    # Construct system for displacements D (n x 3)
    I = sp.eye(n, format="csr")

    # Quadratic operator A = λ LᵀL + α (I - P) + β P
    # where P = nnᵀ per vertex
    A = lambda_lap * (L.T @ L)
    if alpha_tangent != 0.0 or beta_normal != 0.0:
        blocks = []
        for i in range(n):
            n_i = N[i]
            P_i = np.outer(n_i, n_i)
            Q_i = alpha_tangent * (np.eye(3) - P_i) + beta_normal * P_i
            blocks.append(Q_i)
        Q = sp.block_diag(blocks, format="csr")
    else:
        Q = sp.csr_matrix((3 * n, 3 * n))
    # Expand Laplacian to 3D (block-diagonal)
    L3 = sp.kron(I, sp.eye(3))
    A3 = sp.kron(A, np.eye(3)) + Q

    # Right-hand side
    b = np.zeros(3 * n)
    for i in range(n):
        n_i = N[i]
        b_i = beta_normal * target_offset[i] * n_i
        b[3 * i : 3 * i + 3] = b_i

    # Apply fixed constraints
    mask = np.ones(3 * n, dtype=bool)
    for fi in fixed_idx:
        mask[3 * fi : 3 * fi + 3] = False

    A3_free = A3[mask][:, mask]
    b_free = b[mask]

    # Solve
    if solver == "direct":
        d_free = spla.spsolve(A3_free, b_free)
    else:
        d_free, info = spla.cg(A3_free, b_free, atol=1e-10)
        if info != 0:
            raise RuntimeError(f"CG did not converge: info={info}")

    # Reassemble displacement vector
    D = np.zeros((n, 3))
    D[free_idx] = d_free.reshape(-1, 3)
    V_new = V + D
    return V_new
