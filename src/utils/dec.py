import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def cotangent_laplacian(V, F):
    """
    Build the symmetric cotangent Laplacian L and (optionally) the mass diagonal.
    V: (n,3) vertices
    F: (m,3) faces (indices)
    Returns: L (n x n sparse CSR)
    """
    n = V.shape[0]
    I = []
    J = []
    W = []

    def cot(a, b):
        # cotangent of angle between vectors a and b
        # cot = a·b / ||a x b||
        cross = np.cross(a, b)
        denom = np.linalg.norm(cross, axis=1)
        # avoid divide-by-zero for degenerate triangles
        denom = np.where(denom == 0, 1e-12, denom)
        return np.einsum("ij,ij->i", a, b) / denom

    # For each face add cot weights for its three edges
    # We'll accumulate entries for symmetric matrix
    for face in F:
        i, j, k = face
        vi, vj, vk = V[i], V[j], V[k]
        # angles at vertices i, j, k:
        # for angle at i, use vectors (vj-vi) and (vk-vi)
        v_ji = vj - vi
        v_ki = vk - vi
        v_ij = vi - vj
        v_kj = vk - vj
        v_ik = vi - vk
        v_jk = vj - vk

        # compute cotangents (as scalars)
        cot_i = np.dot(v_ji, v_ki) / (np.linalg.norm(np.cross(v_ji, v_ki)) + 1e-12)
        cot_j = np.dot(v_ij, v_kj) / (np.linalg.norm(np.cross(v_ij, v_kj)) + 1e-12)
        cot_k = np.dot(v_ik, v_jk) / (np.linalg.norm(np.cross(v_ik, v_jk)) + 1e-12)

        # edge (j,k) opposite i gets weight cot_i
        for a, b, w in [
            (j, k, cot_i),
            (k, j, cot_i),
            (k, i, cot_j),
            (i, k, cot_j),
            (i, j, cot_k),
            (j, i, cot_k),
        ]:
            I.append(a)
            J.append(b)
            W.append(w * 0.5)  # commonly 1/2 factor

    # assemble sparse matrix and symmetrize (should be symmetric)
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
