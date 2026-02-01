import numpy as np
from .local import fit_local_robust
from .utils import cosine_similarity_rows

def _concat_last(A_full, B_full, idx):
    parts = []
    if A_full is not None and A_full.ndim == 3:
        parts.append(A_full[:, :, idx])
    if B_full is not None and B_full.ndim == 3:
        parts.append(B_full[:, :, idx])
    if not parts:
        return None
    return np.concatenate(parts, axis=1)

def _support_ratio(coefs_last, thr, mode="loc_any"):
    if coefs_last is None or thr is None:
        return 0.0
    if mode == "coef_sum":
        cnt = int(np.nansum(np.abs(coefs_last) > thr))
        return cnt / float(coefs_last.shape[0])
    max_abs = np.nanmax(np.abs(coefs_last), axis=1)
    return float(np.nansum(max_abs > thr)) / float(len(max_abs))

def tune_k_q(
    X, Y, coords,
    K_grid,
    q_grid,
    include_self=True,
    kernel="gaussian",
    ridge=None,
    thr=None,
    thr_quantile=95.0,
    thr_scale=0.5,
    support_mode="loc_any",
    min_support=None,
    support_rel=0.80,
    eps=1e-12,
    gof_floor=1e-8,
    rel_tol=0.01,
    patience=2,
    slack=0.02,
    use_stability=True,
    dK=5,
    stab_tau=0.90,
    delta=1.345,
    max_iter=10,
    tol=1e-4,
):
    """
    Joint tuning of (K, q) with early stopping and optional stability screening.

    Returns
    -------
    best_K, best_q, best_gof, summary
    """
    K_list = sorted(list(K_grid))
    q_list = list(q_grid)

    gof_arr = np.full((len(K_list),), np.inf, dtype=float)
    q_star_list = [None] * len(K_list)
    support_list = [np.nan] * len(K_list)
    stab_list = [np.nan] * len(K_list)

    cache = {}
    def fit_at_K(K):
        if K not in cache:
            rho, A, B, _ = fit_local_robust(
                X, Y, coords, K,
                q=min(min(X.shape[1], Y.shape[1]), max(q_list)),
                include_self=include_self,
                kernel=kernel,
                ridge=ridge,
                delta=delta,
                max_iter=max_iter,
                tol=tol,
                return_diagnostics=False,
                return_all=True
            )
            cache[K] = (rho, A, B)
        return cache[K]

    # auto threshold
    if thr is None:
        pilot_K = K_list[len(K_list)//2]
        _, A_p, B_p = fit_at_K(pilot_K)
        parts = []
        if A_p is not None: parts.append(np.abs(A_p).ravel())
        if B_p is not None: parts.append(np.abs(B_p).ravel())
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

        # auto min_support: relative to mean support over q_grid at this K
        if min_support is None:
            S_vals = []
            for qq in q_list:
                if 1 <= qq <= r:
                    U = _concat_last(A_full, B_full, qq-1)
                    S_vals.append(_support_ratio(U, thr_used, mode=support_mode))
            mean_support_K = float(np.nanmean(S_vals)) if len(S_vals) else 0.0
            min_support_used = support_rel * mean_support_K
        else:
            min_support_used = float(min_support)

        # choose the largest significant q*
        chosen_q, chosen_support = None, 0.0
        for qq in q_list:
            if qq < 1 or qq > r:
                continue
            U = _concat_last(A_full, B_full, qq-1)
            sup = _support_ratio(U, thr_used, mode=support_mode)
            if sup >= min_support_used:
                chosen_q, chosen_support = qq, sup

        q_star_list[i] = chosen_q
        support_list[i] = chosen_support

        if chosen_q is None:
            continue

        # GOF = 1 - (sum_{j<=q*} rho^2) / (sum_{all} rho^2)
        num = float(np.nansum(rho_full[:, :chosen_q] ** 2))
        den = float(np.nansum(rho_full ** 2)) + eps
        ratio = num / den if (np.isfinite(num) and np.isfinite(den) and den > eps) else np.nan
        gof = 1.0 - ratio if np.isfinite(ratio) else np.nan
        gof = float(max(gof, gof_floor)) if np.isfinite(gof) else np.nan
        gof_arr[i] = gof

        # stability check (adjacent K)
        if use_stability:
            Kp = K + dK if (K + dK) in K_list else (K - dK if (K - dK) in K_list else None)
            if Kp is not None:
                _, A_p, B_p = fit_at_K(Kp)
                if chosen_q <= r:
                    U = _concat_last(A_full, B_full, chosen_q-1)
                    V = _concat_last(A_p, B_p, chosen_q-1)
                    if U is not None and V is not None and U.shape == V.shape:
                        sims = cosine_similarity_rows(U, V, eps=eps)
                        stab_list[i] = float(np.nanmedian(sims))

        # early stopping on GOF improvement
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
            "note": "no valid (K,q*) before early stop"
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
    cand_idx = int(idxs[np.argmin(K_obs[idxs])] ) if len(idxs) else int(np.argmin(gof_obs))

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
        "thr_used": thr_used,
        "params": dict(
            kernel=kernel, include_self=include_self, ridge=ridge,
            thr=("auto" if thr is None else float(thr)),
            thr_quantile=thr_quantile, thr_scale=thr_scale,
            support_mode=support_mode,
            min_support=("auto" if min_support is None else float(min_support)),
            support_rel=support_rel,
            rel_tol=rel_tol, patience=patience, slack=slack,
            use_stability=use_stability, dK=dK, stab_tau=stab_tau,
            delta=delta, max_iter=max_iter, tol=tol
        )
    }
    return best_K, best_q, best_g, summary
