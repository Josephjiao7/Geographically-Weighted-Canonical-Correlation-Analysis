import numpy as np
import matplotlib.pyplot as plt

def plot_component_map(gdf, values, title, component_idx=1, ax=None, cmap="viridis"):
    """
    Plot a map of a component (rho or loading of one feature).
    Requires GeoDataFrame `gdf`.

    Parameters
    ----------
    gdf : GeoDataFrame
    values : (n,q) array
    component_idx : int (1-based)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))

    comp = np.asarray(values)[:, component_idx - 1]
    gdf = gdf.copy()
    gdf["__value__"] = comp

    gdf.plot(column="__value__", ax=ax, cmap=cmap, legend=True, legend_kwds={"shrink": 0.8})
    gdf.boundary.plot(ax=ax, linewidth=0.2, color="black")

    ax.set_title(f"{title} (Variate {component_idx})")
    ax.set_axis_off()
    return ax

def plot_loading_maps(
    gdf,
    loadings,
    feature_names,
    component_idx=1,
    nrows=2,
    ncols=2,
    figsize=(10, 7),
    cmap="RdBu"
):
    """
    Plot local loadings for each feature (one component).
    Requires GeoDataFrame `gdf`.

    Notes
    -----
    Uses a shared symmetric range centered at 0 across all subplots.
    """
    loadings = np.asarray(loadings)
    n, p, q = loadings.shape
    if component_idx < 1 or component_idx > q:
        raise ValueError("component_idx out of range.")

    vmax = np.nanmax(np.abs(loadings[:, :, component_idx - 1]))
    vmin = -vmax if np.isfinite(vmax) else None

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i >= p:
            ax.set_visible(False)
            continue
        g = gdf.copy()
        g["__loading__"] = loadings[:, i, component_idx - 1]
        g.plot(column="__loading__", ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, legend=True)
        g.boundary.plot(ax=ax, linewidth=0.2, color="black")
        ax.set_title(f"{feature_names[i]} (variate {component_idx})", fontsize=9)
        ax.set_axis_off()

    plt.tight_layout()
    return fig
