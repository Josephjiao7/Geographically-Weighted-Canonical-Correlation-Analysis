import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm

# ============================================================
# Kernels
# ============================================================

def gaussian_kernel(d, r):
    """
    Gaussian kernel weights.
    Parameters
    ----------
    d : array-like
        Distances.
    r : float
        Bandwidth (usually distance to k-th neighbor).
    Returns
    -------
    w : ndarray
        Weights.
    """
    d = np.asarray(d, dtype=float)
    r = float(r) if r is not None else 1.0
    r = max(r, 1e-15)
    return np.exp(-0.5*(d / r) ** 2)


def bisquare_kernel(d, r):
    """
    Bisquare kernel weights.
    Parameters
    ----------
    d : array-like
        Distances.
    r : float
        Bandwidth.
    Returns
    -------
    w : ndarray
        Weights.
    """
    d = np.asarray(d, dtype=float)
    r = float(r) if r is not None else 1.0
    r = max(r, 1e-15)
    w = np.zeros_like(d)
    mask = d <= r
    w[mask] = (1 - (d[mask] / r) ** 2) ** 2
    return w


def _kernel_weights(d, bw, kernel="gaussian"):
    if kernel == "bisquare":
        return bisquare_kernel(d, bw)
    return gaussian_kernel(d, bw)


# ============================================================
# Utilities
# ============================================================

def _stable_sign_flip(A, B):
    """
    Make the first element of each canonical vector positive to stabilize signs.
    """
    if A is None or B is None:
        return A, B
    s = np.sign(A[0, :])
    s[s == 0] = 1.0
    A = A * s
    B = B * s
    return A, B


def _check_inputs(X, Y, coords, k_neighbors):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    coords = np.asarray(coords, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays.")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be (n, 2).")
    if X.shape[0] != Y.shape[0] or X.shape[0] != coords.shape[0]:
        raise ValueError("X, Y, coords must have the same number of rows.")
    k_neighbors = int(k_neighbors)
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be >= 1.")
    return X, Y, coords, k_neighbors


# ============================================================
# Weighted covariance (local mean removed)
# ============================================================

def gw_covariance(X, Y, W):
    """
    Weighted covariance blocks after removing the weighted mean.
    Parameters
    ----------
    X : (m,p)
    Y : (m,q)
    W : (m,) or (m,1)
    Returns
    -------
    Sigma_XX, Sigma_YY, Sigma_XY
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W, dtype=float).reshape(-1, 1)
    sumW = float(W.sum()) + 1e-15

    mu_X = (W * X).sum(axis=0) / sumW
    mu_Y = (W * Y).sum(axis=0) / sumW

    Xc = X - mu_X
    Yc = Y - mu_Y

    Sigma_XX = (Xc * W).T @ Xc / sumW
    Sigma_YY = (Yc * W).T @ Yc / sumW
    Sigma_XY = (Xc * W).T @ Yc / sumW
    return Sigma_XX, Sigma_YY, Sigma_XY


# ============================================================
# Normal GWCCA local
# ============================================================

def gwcca_local(X, Y, coords, k_neighbors, q=2, include_self=True, kernel="gaussian"):
    """
    Classical GWCCA local estimation using generalized eigen approach.

    Note
    ----
    This keeps your original "Sigma += I" regularization (as-is).
    If you dislike it, set it to a smaller value yourself in your own code,
    but I will NOT change your core logic here.

    Returns
    -------
    rho : (n,q)
    a   : (n,p,q)
    b   : (n,qY,q)
    """
    X, Y, coords, k_neighbors = _check_inputs(X, Y, coords, k_neighbors)
    n, p = X.shape
    qY = Y.shape[1]
    q = int(q)
    rmax = min(p, qY)
    if q < 1 or q > rmax:
        raise ValueError(f"q must be in [1, {rmax}], got {q}.")

    rho = np.full((n, q), np.nan, dtype=float)
    a = np.full((n, p, q), np.nan, dtype=float)
    b = np.full((n, qY, q), np.nan, dtype=float)

    tree = cKDTree(coords)

    for i in range(n):
        d, idx = tree.query(coords[i], k=k_neighbors + 1)
        if not include_self:
            m = (idx != i)
            d, idx = d[m], idx[m]
        if idx.size < 2:
            continue

        bw = float(d[-1]) if d.size > 0 else 1.0
        W = _kernel_weights(d, bw, kernel=kernel)

        Sxx, Syy, Sxy = gw_covariance(X[idx], Y[idx], W)

        # Your original default regularization (kept exactly)
        Sxx = Sxx + np.eye(p)
        Syy = Syy + np.eye(qY)

        try:
            invSxx = np.linalg.pinv(Sxx)
            invSyy = np.linalg.pinv(Syy)
            M = invSxx @ Sxy @ invSyy @ Sxy.T

            # Use eigh if symmetric-ish; but keep your eig style
            evals, evecs = np.linalg.eig(M)

            evals = np.real(evals)
            evecs = np.real(evecs)

            order = np.argsort(evals)[::-1]
            sel = order[:q]

            rho[i, :] = np.sqrt(np.clip(evals[sel], 0.0, None))
            A = evecs[:, sel]
            B = invSyy @ Sxy.T @ A

            A, B = _stable_sign_flip(A, B)

            a[i, :, :] = A
            b[i, :, :] = B

        except np.linalg.LinAlgError:
            continue

    return rho, a, b


# ============================================================
# Robust covariance via Huber IRLS
# ============================================================

def huber_weight(u, delta=1.345):
    """
    Huber IRLS weight: psi(u)/u with psi(u)=min(u, delta).
    u must be nonnegative.
    """
    u = np.asarray(u, dtype=float)
    w = np.ones_like(u)
    m = u > 0
    w[m] = np.minimum(u[m], delta) / (u[m] + 1e-15)
    return w


def robust_cov_blocks_huber(X, Y, W_base, delta=1.345, max_iter=10, tol=1e-4):
    """
    Robustify local covariance using Huber M-estimation (IRLS) on Z=[X,Y].

    Returns
    -------
    Sxx, Syy, Sxy, W_eff, muX, muY
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    W0 = np.asarray(W_base, dtype=float).reshape(-1, 1)
    W0 = W0 / (W0.sum() + 1e-15)

    m, p = X.shape
    q = Y.shape[1]
    Z = np.hstack([X, Y])

    mu = (W0 * Z).sum(axis=0, keepdims=True)

    r = np.linalg.norm(Z - mu, axis=1)
    med = np.median(r)
    mad = np.median(np.abs(r - med)) + 1e-12
    s = 1.4826 * mad if mad > 0 else (r.mean() + 1e-12)

    W = W0.copy()
    for _ in range(max_iter):
        r = np.linalg.norm(Z - mu, axis=1)
        u = r / (s + 1e-15)
        wh = huber_weight(u, delta).reshape(-1, 1)
        W = W0 * wh
        W = W / (W.sum() + 1e-15)

        mu_new = (W * Z).sum(axis=0, keepdims=True)
        if np.linalg.norm(mu_new - mu) < tol:
            mu = mu_new
            break
        mu = mu_new

        r = np.linalg.norm(Z - mu, axis=1)
        s = np.sqrt(np.sum(W.ravel() * np.minimum(r, delta * s) ** 2) / (W.sum() + 1e-15)) + 1e-12

    muX = mu[:, :p].ravel()
    muY = mu[:, p:].ravel()
    Xc = X - muX
    Yc = Y - muY

    Sxx = (Xc * W).T @ Xc
    Syy = (Yc * W).T @ Yc
    Sxy = (Xc * W).T @ Yc

    return Sxx, Syy, Sxy, W, muX, muY


# ============================================================
# Robust GWCCA local
# ============================================================

def gwcca_local_robust(
    X, Y, coords, k_neighbors,
    q=None,
    include_self=True,
    return_all=True,
    delta=1.345,
    max_iter=10,
    tol=1e-4,
    kernel="gaussian",
    eps_eig=1e-12,
    ridge=None
):
    """
    Robust local GWCCA:
    - Robust covariance by Huber-IRLS on Z=[X,Y]
    - Whitening + SVD
    - Re-normalization so A^T Sxx A = I and B^T Syy B = I

    Returns
    -------
    rho : (n,r)
    a   : (n,p,r)
    b   : (n,qY,r)
    """
    X, Y, coords, k_neighbors = _check_inputs(X, Y, coords, k_neighbors)
    n, p = X.shape
    qY = Y.shape[1]
    rmax = min(p, qY)

    r = (q if q is not None else (rmax if return_all else 1))
    r = int(min(r, rmax))
    if r < 1:
        raise ValueError("q (or r) must be >= 1.")

    rho = np.full((n, r), np.nan, dtype=float)
    a = np.full((n, p, r), np.nan, dtype=float)
    b = np.full((n, qY, r), np.nan, dtype=float)

    tree = cKDTree(coords)

    for i in range(n):
        d, idx = tree.query(coords[i], k=k_neighbors + 1)
        if not include_self:
            m = (idx != i)
            d, idx = d[m], idx[m]
        if idx.size < 2:
            continue

        bw = float(d[-1]) if d.size > 0 else 1.0
        w0 = _kernel_weights(d, bw, kernel=kernel).reshape(-1, 1)
        w0 = w0 / (w0.sum() + 1e-15)

        Xi, Yi = X[idx], Y[idx]
        Sxx, Syy, Sxy, _, _, _ = robust_cov_blocks_huber(
            Xi, Yi, w0, delta=delta, max_iter=max_iter, tol=tol
        )

        if ridge is not None and ridge > 0:
            Sxx = Sxx + float(ridge) * np.eye(p)
            Syy = Syy + float(ridge) * np.eye(qY)

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

        svals = svals[:r]
        U = U[:, :r]
        Vt = Vt[:r, :]

        A = Sxx_mh @ U
        B = Syy_mh @ Vt.T

        for j in range(r):
            aj = A[:, j]
            bj = B[:, j]
            A[:, j] = aj / (np.sqrt(aj.T @ Sxx @ aj) + 1e-15)
            B[:, j] = bj / (np.sqrt(bj.T @ Syy @ bj) + 1e-15)

        A, B = _stable_sign_flip(A, B)

        rho[i, :] = svals
        a[i, :, :] = A
        b[i, :, :] = B

    return rho, a, b


# ============================================================
# Spatial smoothing of loadings
# ============================================================

def spatial_smooth_loadings(loadings, coords, k_neighbors, kernel="gaussian"):
    """
    Spatial smoothing of local loadings using KNN Gaussian weights.
    Parameters
    ----------
    loadings : (n,p,q)
    coords   : (n,2)
    k_neighbors : int
    kernel : {"gaussian","bisquare"}
    Returns
    -------
    smoothed : (n,p,q)
    """
    loadings = np.asarray(loadings, dtype=float)
    coords = np.asarray(coords, dtype=float)
    n, p, q = loadings.shape

    smoothed = np.zeros_like(loadings)
    tree = cKDTree(coords)

    for i in range(n):
        d, idx = tree.query(coords[i], k=k_neighbors + 1)
        bw = float(d[-1]) if d.size > 0 else 1.0
        w = _kernel_weights(d, bw, kernel=kernel)
        w = w / (w.sum() + 1e-15)
        smoothed[i] = np.average(loadings[idx], axis=0, weights=w)

    return smoothed


# ============================================================
# One-shot GWCCA (local + smoothing)
# ============================================================

def gwcca(X, Y, coords, k_neighbors, q=2, robust=False, kernel="gaussian", **robust_kwargs):
    """
    Run GWCCA and then spatially smooth the loadings.
    Parameters
    ----------
    robust : bool
        If True use gwcca_local_robust, otherwise gwcca_local.
    robust_kwargs :
        Passed to gwcca_local_robust (e.g., delta, max_iter, tol, ridge).
    Returns
    -------
    rho, a_smoothed, b_smoothed
    """
    if robust:
        rho, a, b = gwcca_local_robust(X, Y, coords, k_neighbors, q=q, kernel=kernel, **robust_kwargs)
    else:
        rho, a, b = gwcca_local(X, Y, coords, k_neighbors, q=q, kernel=kernel)
    a_sm = spatial_smooth_loadings(a, coords, k_neighbors, kernel=kernel)
    b_sm = spatial_smooth_loadings(b, coords, k_neighbors, kernel=kernel)
    return rho, a_sm, b_sm


# ============================================================
# Tuning utilities (your joint_optimize_k_q_early)
# ============================================================

def _concat_last_coeffs(A_full, B_full, idx):
    parts = []
    if A_full is not None and A_full.ndim == 3:
        parts.append(A_full[:, :, idx])
    if B_full is not None and B_full.ndim == 3:
        parts.append(B_full[:, :, idx])
    if not parts:
        return None
    return np.concatenate(parts, axis=1)


def _support_ratio(coefs_last, thr, frac=0.3):
    """
    Your original support criterion: location counts whose proportion of
    (|coef| > thr) across features exceeds frac.
    """
    if coefs_last is None or thr is None:
        return 0.0
    hit = (np.abs(coefs_last) > thr).astype(float)
    prop = np.nanmean(hit, axis=1)
    return float(np.nansum(prop >= float(frac))) / float(len(prop))


def _cosine_sim(U, V, eps=1e-12):
    num = np.sum(U * V, axis=1)
    den = (np.linalg.norm(U, axis=1) * np.linalg.norm(V, axis=1) + eps)
    sim = num / den
    sim = np.where(np.isfinite(sim), sim, 0.0)
    return sim


def joint_optimize_k_q_early(
    X, Y, coords,
    K_grid, q_grid, include_self=True,
    thr=None,
    min_support=None,
    support_mode="loc_any",   # kept for API compatibility; not used in your _support_ratio
    thr_quantile=95.0,
    thr_scale=0.5,
    support_rel=0.80,
    eps=1e-12, gof_floor=1e-8, enforce_q_lt_r=False,
    rel_tol=0.01, patience=2,
    slack=0.02,
    use_stability=True, dK=5, stab_tau=0.90,
    kernel="gaussian",
    robust=True,
    robust_kwargs=None
):
    """
    Joint tuning of (k_neighbors, q) with early stopping.

    Notes
    -----
    - By default robust=True because your original joint optimizer calls gwcca_local_robust.
      If you want the classical version, set robust=False.
    """
    X, Y, coords, _ = _check_inputs(X, Y, coords, k_neighbors=1)
    robust_kwargs = robust_kwargs or {}

    K_list = sorted(list(K_grid))
    q_list = list(q_grid)

    gof_arr = np.full((len(K_list),), np.inf, dtype=float)
    q_star_list = [None] * len(K_list)
    support_list = [np.nan] * len(K_list)
    stab_list = [np.nan] * len(K_list)

    cache = {}
    def fit_at_K(K):
        if K not in cache:
            if robust:
                rho, A, B = gwcca_local_robust(
                    X, Y, coords, K,
                    include_self=include_self,
                    return_all=True,
                    kernel=kernel,
                    **robust_kwargs
                )
            else:
                # For normal, we compute up to rmax by calling with q=rmax
                rmax = min(X.shape[1], Y.shape[1])
                rho, A, B = gwcca_local(
                    X, Y, coords, K,
                    q=rmax,
                    include_self=include_self,
                    kernel=kernel
                )
            cache[K] = (rho, A, B)
        return cache[K]

    # threshold
    if thr is None:
        if len(K_list) == 0:
            raise ValueError("K_grid is empty.")
        pilot_K = K_list[len(K_list)//2]
        _, A_pilot, B_pilot = fit_at_K(pilot_K)
        parts = []
        if A_pilot is not None: parts.append(np.abs(A_pilot).ravel())
        if B_pilot is not None: parts.append(np.abs(B_pilot).ravel())
        all_abs = np.concatenate(parts) if parts else np.array([np.nan])
        thr_used = np.nanpercentile(all_abs, thr_quantile) * float(thr_scale)
    else:
        thr_used = float(thr)

    last_gof = None
    no_improve = 0
    stop_idx = len(K_list)

    for i, K in enumerate(K_list):
        rho_full, A_full, B_full = fit_at_K(K)
        r = rho_full.shape[1]

        if min_support is None:
            S_vals = []
            for qq in q_list:
                if 1 <= qq <= r:
                    coefs_last_tmp = _concat_last_coeffs(A_full, B_full, qq-1)
                    S_vals.append(_support_ratio(coefs_last_tmp, thr_used))
            mean_support_K = float(np.nanmean(S_vals)) if len(S_vals) else 0.0
            min_support_used = support_rel * mean_support_K
        else:
            min_support_used = float(min_support)

        chosen_q, chosen_support = None, 0.0
        for qq in q_list:
            if qq < 1 or qq > r:
                continue
            if enforce_q_lt_r and qq >= r:
                continue
            coefs_last = _concat_last_coeffs(A_full, B_full, qq - 1)
            sup_ratio = _support_ratio(coefs_last, thr_used)
            if sup_ratio >= min_support_used:
                chosen_q, chosen_support = qq, sup_ratio

        q_star_list[i] = chosen_q
        support_list[i] = chosen_support

        if chosen_q is None:
            continue

        num = float(np.nansum(rho_full[:, :chosen_q] ** 2))
        den = float(np.nansum(rho_full ** 2)) + eps
        if not np.isfinite(num) or not np.isfinite(den) or den < eps:
            gof = np.nan
        else:
            ratio = num / den
            gof = 1.0 - ratio
            if gof < gof_floor:
                gof = float(gof_floor)

        gof_arr[i] = gof

        if use_stability:
            Kp = K + dK if (K + dK) in K_list else (K - dK if (K - dK) in K_list else None)
            if Kp is not None:
                rho_p, A_p, B_p = fit_at_K(Kp)
                if chosen_q <= rho_p.shape[1]:
                    U = _concat_last_coeffs(A_full, B_full, chosen_q - 1)
                    V = _concat_last_coeffs(A_p,    B_p,    chosen_q - 1)
                    if U is not None and V is not None and U.shape == V.shape:
                        sims = _cosine_sim(U, V, eps=eps)
                        stab_list[i] = float(np.nanmedian(sims))

        if np.isfinite(gof):
            if last_gof is None:
                last_gof, no_improve = gof, 0
            else:
                rel_impr = (last_gof - gof) / max(last_gof, eps)
                if rel_impr > rel_tol:
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        stop_idx = i + 1
                        break
                last_gof = gof

    observed = slice(0, stop_idx)
    have = np.isfinite(gof_arr[observed]) & (np.array(q_star_list[observed]) != None)
    if not np.any(have):
        return None, None, np.inf, {
            "Ks": np.array(K_list),
            "qs": np.array(q_star_list),
            "gof": gof_arr,
            "support": np.array(support_list),
            "stability": np.array(stab_list),
            "stop_at_index": stop_idx,
            "params": dict(
                thr=("auto" if thr is None else float(thr)),
                min_support=("auto" if min_support is None else float(min_support)),
                thr_quantile=thr_quantile, thr_scale=thr_scale, support_rel=support_rel,
                support_mode=support_mode,
                eps=eps, gof_floor=gof_floor, enforce_q_lt_r=enforce_q_lt_r,
                rel_tol=rel_tol, patience=patience, slack=slack,
                use_stability=use_stability, dK=dK, stab_tau=stab_tau,
                kernel=kernel, robust=robust, robust_kwargs=robust_kwargs
            ),
            "note": "no valid (K,q*) before early-stop (auto thresholds active)"
        }

    gof_obs = gof_arr[observed][have]
    K_obs   = np.array(K_list)[observed][have]
    q_obs   = np.array([q if q is not None else -1 for q in q_star_list[observed]])[have]
    stab_obs= np.array(stab_list)[observed][have]

    gmin = np.min(gof_obs)
    near = gof_obs <= (1.0 + slack) * gmin
    if use_stability:
        near = near & (np.isnan(stab_obs) | (stab_obs >= stab_tau))

    idxs = np.where(near)[0]
    if len(idxs) == 0:
        idxs = np.where(gof_obs <= (1.0 + slack) * gmin)[0]
    if len(idxs) == 0:
        cand_idx = int(np.argmin(gof_obs))
    else:
        order = np.lexsort((K_obs[idxs], gof_obs[idxs], -q_obs[idxs]))
        cand_idx = idxs[order[0]]

    best_K = int(K_obs[cand_idx])
    best_q = int(q_obs[cand_idx])
    best_g = float(max(gof_obs[cand_idx], gof_floor))

    summary = {
        "Ks": np.array(K_list),
        "qs": np.array(q_star_list),
        "gof": gof_arr,
        "support": np.array(support_list),
        "stability": np.array(stab_list),
        "stop_at_index": stop_idx,
        "params": dict(
            thr=("auto" if thr is None else float(thr)),
            min_support=("auto" if min_support is None else float(min_support)),
            thr_quantile=thr_quantile, thr_scale=thr_scale, support_rel=support_rel,
            support_mode=support_mode,
            eps=eps, gof_floor=gof_floor, enforce_q_lt_r=enforce_q_lt_r,
            rel_tol=rel_tol, patience=patience, slack=slack,
            use_stability=use_stability, dK=dK, stab_tau=stab_tau,
            kernel=kernel, robust=robust, robust_kwargs=robust_kwargs
        ),
        "thr_used": thr_used,
        "note": "early-stopping + near-opt selection with CCA-style GOF"
    }
    return best_K, best_q, best_g, summary

def plot_gwcca_result(gdf, coefficient, title, component_idx=1, ax=None):
    """
    Visualize a given GWCCA result (rho or loadings), optionally on a provided matplotlib axis.

    Parameters:
        gdf (GeoDataFrame): spatial data
        coefficient (ndarray): matrix of shape (n, q)
        title (str): title of the map
        component_idx (int): which canonical component to plot (1-based)
        ax (matplotlib axis): optional axis to draw on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    gdf['coefficient'] = coefficient[:, component_idx - 1]
    gdf.plot(column='coefficient', ax=ax, cmap='viridis', legend=True, legend_kwds={'shrink': 0.8})
    gdf.boundary.plot(ax=ax, linewidth=0.02, color='black')

    ax.set_title(f"{title} (Variate {component_idx})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return ax


def plot_loading_maps(gdf,
                      loading,
                      feature_names,
                      component_idx=1,
                      nrows=2,
                      ncols=2,
                      figsize=(10,7),
                      cmap="RdBu",
                      diverging=True):
    """
    Plot local loadings; if diverging=True the colour map is centred at 0.
    """
    p = loading.shape[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes = axes.flatten()

    # decide global min/max so every subplot shares the same scale
    if diverging:
        vmax = np.nanmax(np.abs(loading[:,:,component_idx-1]))
        vmin = -vmax
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        vmin, vmax, norm = None, None, None     # fallback to default

    for i, ax in enumerate(axes):
        if i < p:
            gdf["loading"] = loading[:, i, component_idx-1]
            gdf.plot(column="loading",
                     ax=ax,
                     cmap=cmap,
                     norm=norm,
                     vmin=vmin,
                     vmax=vmax,
                     legend=True)
            gdf.boundary.plot(ax=ax, linewidth=0.1, color="black")
            ax.set_title(f"{feature_names[i]}  (variate {component_idx})", fontsize=9)
            # ax.set_axis_off()
        else:
            ax.set_visible(False)

    plt.tight_layout()

def gwcca_local_permutation_test(
    X, Y, coords, k_neighbors, q,
    rho_obs,
    n_perm=999,
    random_state=0,
    two_sided=True
):
    """
    Local permutation significance test for GWCCA canonical correlations.

    Parameters
    ----------
    rho_obs : (n, q)
        Observed local canonical correlations from gwcca()
    n_perm : int
        Number of permutations

    Returns
    -------
    pvals : (n, q)
        Empirical p-values for each location and component
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    rho_perm = np.zeros((n_perm, n, q), dtype=float)

    for b in range(n_perm):
        # permutation: break X–Y correspondence
        perm_idx = rng.permutation(n)
        Y_perm = Y[perm_idx]

        rho_b, _, _ = gwcca(
            X, Y_perm, coords,
            k_neighbors=k_neighbors,
            q=q
        )
        rho_perm[b] = rho_b

    pvals = np.zeros_like(rho_obs)

    for k in range(q):
        obs = rho_obs[:, k]
        null = rho_perm[:, :, k]

        if two_sided:
            pvals[:, k] = (
                np.sum(np.abs(null) >= np.abs(obs), axis=0) + 1
            ) / (n_perm + 1)
        else:
            pvals[:, k] = (
                np.sum(null >= obs, axis=0) + 1
            ) / (n_perm + 1)

    return pvals