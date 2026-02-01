import numpy as np

def save_results_csv(gdf, rho, A, B, X_columns, Y_columns, filename="gwcca_results.csv"):
    """
    Save GWCCA results to CSV.
    Requires that gdf supports .to_csv (GeoDataFrame / DataFrame).

    rho : (n,q)
    A   : (n,p,q)
    B   : (n,qY,q)
    """
    rho = np.asarray(rho)
    A = np.asarray(A)
    B = np.asarray(B)

    q = rho.shape[1]
    out = gdf.copy()

    for j in range(q):
        out[f"rho{j+1}"] = rho[:, j]

    for i, name in enumerate(X_columns):
        for j in range(q):
            out[f"a{j+1}_{name}"] = A[:, i, j]

    for i, name in enumerate(Y_columns):
        for j in range(q):
            out[f"b{j+1}_{name}"] = B[:, i, j]

    out.to_csv(filename, index=False)
    return filename

def save_results_geojson(gdf, rho, A, B, X_columns, Y_columns, filename="gwcca_results.geojson"):
    """
    Save GWCCA results to GeoJSON.
    Requires GeoDataFrame and geopandas installed.

    Notes
    -----
    Uses `driver="GeoJSON"`.
    """
    rho = np.asarray(rho)
    A = np.asarray(A)
    B = np.asarray(B)

    q = rho.shape[1]
    out = gdf.copy()

    for j in range(q):
        out[f"rho{j+1}"] = rho[:, j]

    for i, name in enumerate(X_columns):
        for j in range(q):
            out[f"a{j+1}_{name}"] = A[:, i, j]

    for i, name in enumerate(Y_columns):
        for j in range(q):
            out[f"b{j+1}_{name}"] = B[:, i, j]

    out.to_file(filename, driver="GeoJSON")
    return filename
