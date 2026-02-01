import numpy as np
from scipy.spatial import cKDTree
from .kernels import gaussian_kernel

def smooth_loadings(loadings, coords, k_neighbors):
    """
    Spatial smoothing for loadings using KNN Gaussian kernel smoother.

    Parameters
    ----------
    loadings : (n,p,q) array
    coords : (n,2) array
    k_neighbors : int

    Returns
    -------
    smoothed : (n,p,q) array
    """
    loadings = np.asarray(loadings, dtype=float)
    coords = np.asarray(coords, dtype=float)
    n = loadings.shape[0]
    k_neighbors = int(k_neighbors)
    if k_neighbors < 2 or k_neighbors >= n:
        raise ValueError("k_neighbors for smoothing must be in [2, n-1].")

    smoothed = np.zeros_like(loadings)
    tree = cKDTree(coords)

    for i in range(n):
        d, idx = tree.query(coords[i], k=k_neighbors + 1)
        bw = float(d[-1]) if d.size > 0 else 1.0
        w = gaussian_kernel(d, bw)
        w = w / (w.sum() + 1e-15)
        smoothed[i] = np.average(loadings[idx], axis=0, weights=w)

    return smoothed
