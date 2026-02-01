from .local import fit_local, fit_local_robust
from .smoothing import smooth_loadings
from .tuning import tune_k_q
from .plotting import plot_component_map, plot_loading_maps
from .io import save_results_csv, save_results_geojson

def fit(
    X, Y, coords, k_neighbors,
    q=2,
    robust=True,
    smooth=False,
    smooth_k=None,
    kernel="gaussian",
    include_self=True,
    ridge=None,
    eps_eig=1e-12,
    delta=1.345,
    max_iter=10,
    tol=1e-4,
    return_diagnostics=False
):
    """
    High-level user API.

    Parameters
    ----------
    X : (n,p) array
    Y : (n,qY) array
    coords : (n,2) array
    k_neighbors : int
        KNN size used for local estimation.
    q : int
        Number of canonical variates to return.
    robust : bool
        If True, use robust local covariance (Huber IRLS) + whitening+SVD.
        If False, use classical local covariance + eigen approach.
    smooth : bool
        If True, spatially smooth A and B using KNN Gaussian smoother.
    smooth_k : int or None
        K used for smoothing. If None, uses k_neighbors.
    kernel : {"gaussian","bisquare"}
    include_self : bool
        Whether local neighborhood includes the center point itself.
    ridge : float or None
        Optional ridge added to Sxx and Syy (robust mode recommended when covariance may be near-singular).
    return_diagnostics : bool
        If True, return (rho, A, B, diagnostics dict).

    Returns
    -------
    rho : (n,q)
    A   : (n,p,q)
    B   : (n,qY,q)
    """
    if robust:
        rho, A, B, diag = fit_local_robust(
            X, Y, coords, k_neighbors,
            q=q,
            include_self=include_self,
            kernel=kernel,
            eps_eig=eps_eig,
            ridge=ridge,
            delta=delta,
            max_iter=max_iter,
            tol=tol,
            return_diagnostics=return_diagnostics
        )
    else:
        rho, A, B, diag = fit_local(
            X, Y, coords, k_neighbors,
            q=q,
            include_self=include_self,
            kernel=kernel,
            ridge=ridge,
            return_diagnostics=return_diagnostics
        )

    if smooth:
        kk = k_neighbors if smooth_k is None else int(smooth_k)
        A = smooth_loadings(A, coords, kk)
        B = smooth_loadings(B, coords, kk)

    if return_diagnostics:
        return rho, A, B, diag
    return rho, A, B