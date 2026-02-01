import numpy as np

def weighted_center(X, W):
    """
    Compute weighted mean and centered matrix.

    Parameters
    ----------
    X : (m,p)
    W : (m,1) nonnegative weights

    Returns
    -------
    mu : (p,)
    Xc : (m,p)
    """
    W = W.reshape(-1, 1).astype(float)
    sw = W.sum() + 1e-15
    mu = (W * X).sum(axis=0) / sw
    Xc = X - mu
    return mu, Xc

def weighted_cov_blocks(X, Y, W):
    """
    Weighted covariance blocks after removing weighted means.

    Returns
    -------
    Sxx, Syy, Sxy
    """
    W = W.reshape(-1, 1).astype(float)
    sw = W.sum() + 1e-15

    muX, Xc = weighted_center(X, W)
    muY, Yc = weighted_center(Y, W)

    Sxx = (Xc * W).T @ Xc / sw
    Syy = (Yc * W).T @ Yc / sw
    Sxy = (Xc * W).T @ Yc / sw
    return Sxx, Syy, Sxy
