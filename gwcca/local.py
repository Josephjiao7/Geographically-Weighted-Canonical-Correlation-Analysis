import numpy as np
from scipy.spatial import cKDTree

from .utils import check_inputs, stable_sign_flip
from .kernels import kernel_weights
from .covariance import weighted_cov_blocks
from .robust import robust_cov_blocks_huber

def fit_local(
    X, Y, coords, k_neighbors,
    q=2,
    include_self=True,
    kernel="gaussian",
    ridge=None,
    return_diagnostics=False
):
    """
    Classical GWCCA local estimation (non-robust).
    Uses weighted covariance + generalized eigen approach.

    Returns
    -------
    rho : (n,q)
    A : (n,p,q)
    B : (n,qY,q)
    diag : dict (optional)
    """
    X, Y, coords, k_neighbors = check_inputs(X, Y, coords, k_neighbors)
    n, p = X.shape
    qY = Y.shape[1]
    rmax = min(p, qY)
    q = int(q)
    if q < 1 or q > rmax:
        raise ValueError(f"q must be in [1, {rmax}], got {q}.")

    rho = np.full((n, q), np.nan)
    A = np.full((n, p, q), np.nan)
    B = np.full((n, qY, q), np.nan)
    bandwidths = np.full((n,), np.nan)

    tree = cKDTree(coords)

    for i in range(n):
        d, idx = tree.query(coords[i], k=k_neighbors + 1)
        if not include_self:
            m = (idx != i)
            d, idx = d[m], idx[m]
        if idx.size < 2:
            continue

        bw = float(d[-1]) if d.size > 0 else 1.0
        w = kernel_weights(d, bw, kernel=kernel).reshape(-1, 1)
        w = w / (w.sum() + 1e-15)

        Sxx, Syy, Sxy = weighted_cov_blocks(X[idx], Y[idx], w)
        if ridge is not None and ridge > 0:
            Sxx = Sxx + float(ridge) * np.eye(p)
            Syy = Syy + float(ridge) * np.eye(qY)

        # Solve M = inv(Sxx) Sxy inv(Syy) Syx
        try:
            invSxx = np.linalg.pinv(Sxx)
            invSyy = np.linalg.pinv(Syy)
            M = invSxx @ Sxy @ invSyy @ Sxy.T
            evals, evecs = np.linalg.eig(M)

            order = np.argsort(evals)[::-1]
            sel = order[:q]
            rhos = np.sqrt(np.clip(evals[sel].real, 0.0, None))

            a = evecs[:, sel].real
            b = (invSyy @ Sxy.T @ a).real
            a, b = stable_sign_flip(a, b)

            rho[i, :] = rhos
            A[i, :, :] = a
            B[i, :, :] = b
            bandwidths[i] = bw
        except np.linalg.LinAlgError:
            continue

    diag = {"bandwidths": bandwidths, "kernel": kernel, "include_self": include_self}
    if return_diagnostics:
        return rho, A, B, diag
    return rho, A, B, None


def fit_local_robust(
    X, Y, coords, k_neighbors,
    q=2,
    include_self=True,
    kernel="gaussian",
    eps_eig=1e-12,
    ridge=None,
    delta=1.345,
    max_iter=10,
    tol=1e-4,
    return_diagnostics=False,
    return_all=False
):
    """
    Robust GWCCA:
    1) Robust local covariance via Huber IRLS on Z=[X,Y]
    2) Whitening + SVD solution for local CCA
    3) Orthonormalization to ensure A^T Sxx A = I, B^T Syy B = I

    Parameters
    ----------
    return_all : bool
        If True, returns all possible canonical variates up to min(p,qY),
        and 'q' acts as a cap.

    Returns
    -------
    rho : (n,q)
    A : (n,p,q)
    B : (n,qY,q)
    diag : dict (optional)
    """
    X, Y, coords, k_neighbors = check_inputs(X, Y, coords, k_neighbors)
    n, p = X.shape
    qY = Y.shape[1]
    rmax = min(p, qY)

    q = int(q)
    if q < 1:
        raise ValueError("q must be >= 1.")
    if return_all:
        q = min(q, rmax)
    else:
        if q > rmax:
            raise ValueError(f"q must be <= {rmax}, got {q}.")

    rho = np.full((n, q), np.nan)
    A = np.full((n, p, q), np.nan)
    B = np.full((n, qY, q), np.nan)

    bandwidths = np.full((n,), np.nan)
    eff_weight_sums = np.full((n,), np.nan)

    tree = cKDTree(coords)

    for i in range(n):
        d, idx = tree.query(coords[i], k=k_neighbors + 1)
        if not include_self:
            m = (idx != i)
            d, idx = d[m], idx[m]
        if idx.size < 2:
            continue

        bw = float(d[-1]) if d.size > 0 else 1.0
        w0 = kernel_weights(d, bw, kernel=kernel).reshape(-1, 1)
        w0 = w0 / (w0.sum() + 1e-15)

        Xi, Yi = X[idx], Y[idx]
        Sxx, Syy, Sxy, W_eff, _, _ = robust_cov_blocks_huber(
            Xi, Yi, w0, delta=delta, max_iter=max_iter, tol=tol
        )

        if ridge is not None and ridge > 0:
            Sxx = Sxx + float(ridge) * np.eye(p)
            Syy = Syy + float(ridge) * np.eye(qY)

        # Whitening
        try:
            eval_x, evec_x = np.linalg.eigh(Sxx)
            eval_y, evec_y = np.linalg.eigh(Syy)
        except np.linalg.LinAlgError:
            continue

        eval_x = np.clip(eval_x, eps_eig, None)
        eval_y = np.clip(eval_y, eps_eig, None)

        Sxx_mh = evec_x @ np.diag(1.0 / np.sqrt(eval_x)) @ evec_x.T
        Syy_mh = evec_y @ np.diag(1.0 / np.sqrt(eval_y)) @ evec_y.T

        K = Sxx_mh @ Sxy @ Syy_mh

        try:
            U, svals, Vt = np.linalg.svd(K, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        svals = svals[:q]
        U = U[:, :q]
        Vt = Vt[:q, :]

        a = Sxx_mh @ U
        b = Syy_mh @ Vt.T

        # Re-normalize so that a^T Sxx a = 1, b^T Syy b = 1
        for j in range(q):
            aj = a[:, j]
            bj = b[:, j]
            sxa = float(np.sqrt(aj.T @ Sxx @ aj) + 1e-15)
            syb = float(np.sqrt(bj.T @ Syy @ bj) + 1e-15)
            a[:, j] = aj / sxa
            b[:, j] = bj / syb

        a, b = stable_sign_flip(a, b)

        rho[i, :] = svals
        A[i, :, :] = a
        B[i, :, :] = b
        bandwidths[i] = bw
        eff_weight_sums[i] = float(W_eff.sum())

    diag = {
        "bandwidths": bandwidths,
        "effective_weight_sum": eff_weight_sums,
        "kernel": kernel,
        "include_self": include_self,
        "robust": True,
        "delta": delta,
        "ridge": ridge
    }
    if return_diagnostics:
        return rho, A, B, diag
    return rho, A, B, None
