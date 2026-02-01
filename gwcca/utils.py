import numpy as np

def as_2d_float(arr, name):
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
    return arr.astype(float, copy=False)

def check_inputs(X, Y, coords, k_neighbors):
    X = as_2d_float(X, "X")
    Y = as_2d_float(Y, "Y")
    coords = as_2d_float(coords, "coords")

    n = X.shape[0]
    if Y.shape[0] != n or coords.shape[0] != n:
        raise ValueError("X, Y, coords must have the same number of rows (n).")

    if coords.shape[1] != 2:
        raise ValueError("coords must have shape (n,2).")

    k_neighbors = int(k_neighbors)
    if k_neighbors < 2:
        raise ValueError("k_neighbors must be >= 2.")
    if k_neighbors >= n:
        raise ValueError("k_neighbors must be < n.")

    # Basic NaN check (do not auto-impute here to avoid surprises)
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN/Inf. Please clean/impute before calling GWCCA.")
    if not np.isfinite(Y).all():
        raise ValueError("Y contains NaN/Inf. Please clean/impute before calling GWCCA.")
    if not np.isfinite(coords).all():
        raise ValueError("coords contains NaN/Inf.")

    return X, Y, coords, k_neighbors

def stable_sign_flip(A, B):
    """
    Flip signs to make the first element of each canonical vector positive,
    improving comparability across locations.
    """
    if A is None or B is None:
        return A, B
    signs = np.sign(A[0, :])
    signs[signs == 0] = 1.0
    A = A * signs
    B = B * signs
    return A, B

def cosine_similarity_rows(U, V, eps=1e-12):
    num = np.sum(U * V, axis=1)
    den = (np.linalg.norm(U, axis=1) * np.linalg.norm(V, axis=1) + eps)
    sim = num / den
    sim = np.where(np.isfinite(sim), sim, 0.0)
    return sim
