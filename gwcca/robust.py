import numpy as np

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

    Parameters
    ----------
    X : (m,p)
    Y : (m,q)
    W_base : (m,1)
        Spatial kernel weights (>=0).
    delta : float
        Huber tuning parameter.
    max_iter : int
    tol : float
        Convergence tolerance on location shift.

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

    # init location: spatially weighted mean
    mu = (W0 * Z).sum(axis=0, keepdims=True)

    # robust scale of residual norms (MAD)
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

        # proposal-2 style scale update
        r = np.linalg.norm(Z - mu, axis=1)
        s = np.sqrt(np.sum(W.ravel() * np.minimum(r, delta * s) ** 2) / (W.sum() + 1e-15)) + 1e-12

    # split and form covariances
    muX = mu[:, :p].ravel()
    muY = mu[:, p:].ravel()
    Xc = X - muX
    Yc = Y - muY

    Sxx = (Xc * W).T @ Xc
    Syy = (Yc * W).T @ Yc
    Sxy = (Xc * W).T @ Yc

    return Sxx, Syy, Sxy, W, muX, muY
