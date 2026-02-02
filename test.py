import numpy as np
import pandas as pd
from gwcca import joint_optimize_k_q_early, gwcca, plot_gwcca_result, plot_loading_maps
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore", message=".*The 'type' attribute is deprecated.*")



data = gpd.read_file("dataset/us1.geojson")
data = data.to_crs("+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

X_columns = ['arth', 'asthma','cancer', 'deprss', 'stroke', 'diabet']
X_names = ['Arthritis', 'Asthma', 'Cancer', 'Depression', 'Stroke', 'Diabetes']

Y_columns = ['age_65', 'white', 'binge', 'impoverishment', 'black', 'hispanic']
Y_names = ['Age 65', 'White', 'Binge', 'Poverty', 'Black', 'Hispanic']

# Standardize the variables to have mean = 0 and standard deviation = 1
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X = scaler_X.fit_transform(data[X_columns].to_numpy())
Y = scaler_Y.fit_transform(data[Y_columns].to_numpy())

coords = np.vstack([data.geometry.centroid.x, data.geometry.centroid.y]).T


p, qY = X.shape[1], Y.shape[1]

r = min(p, qY)

K_grid = range(30, 301, 2)
q_grid = range(1, r)

best_K, best_q, best_gof, summary = joint_optimize_k_q_early(
    X, Y, coords,
    K_grid=K_grid,
    q_grid=q_grid,
    rel_tol=0.01, patience=2,
    slack=0.02,
    thr=None,
    min_support=None,
    use_stability=False
)


print("Optimal K (k_neighbors):", best_K)
print("Optimal q:", best_q)
print("Best GOF:", best_gof)


rho, a, b = gwcca(X, Y, coords, k_neighbors=best_K, q=best_q)

fig, ax = plt.subplots(figsize=(10, 6))
plot_gwcca_result(data, rho, title="GWCCA coefficient", component_idx=1, ax=ax)
plt.tight_layout()
# plt.show()
plt.savefig("picture/rho1.png", dpi=600)


cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_diverging", ["#0042ca", "white", "#e9141e"], N=256
)

plot_loading_maps(data, a, X_names, component_idx=1, nrows=3, ncols=2, figsize=(9,8),cmap=cmap, diverging=True)
plt.savefig("picture/a1.png", dpi=600)


plot_loading_maps(data, b, Y_names, component_idx=1, nrows=3, ncols=2, figsize=(9,8),cmap=cmap, diverging=True)
plt.savefig("picture/b1.png", dpi=600)