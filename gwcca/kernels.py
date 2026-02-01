import numpy as np

def gaussian_kernel(d, bw):
    """
    Gaussian kernel weights.
    w(d) = exp(-0.5 * (d/bw)^2)
    """
    bw = float(bw) if bw > 0 else 1.0
    d = np.asarray(d, dtype=float)
    return np.exp(-0.5 * (d / bw) ** 2)

def bisquare_kernel(d, bw):
    """
    Bisquare kernel weights (compact support).
    w(d) = (1 - (d/bw)^2)^2 for d <= bw, else 0
    """
    bw = float(bw) if bw > 0 else 1.0
    d = np.asarray(d, dtype=float)
    w = np.zeros_like(d, dtype=float)
    m = d <= bw
    w[m] = (1.0 - (d[m] / bw) ** 2) ** 2
    return w

def kernel_weights(d, bw, kernel="gaussian"):
    kernel = str(kernel).lower()
    if kernel == "bisquare":
        return bisquare_kernel(d, bw)
    return gaussian_kernel(d, bw)
