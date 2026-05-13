"""
src/visualization.py
Static map exports (PNG) — used by standalone scripts, not by the Streamlit app.
"""

import os
import matplotlib.pyplot as plt


def _ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_vulnerability_map(data_df, output_path="outputs/maps/vulnerability_map.png"):
    """
    Save a choropleth map of vulnerability_score to disk.

    Parameters
    ----------
    data_df : GeoDataFrame with 'vulnerability_score' column.
    output_path : str
    """
    _ensure_dir(output_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    data_df.plot(
        column="vulnerability_score",
        cmap="Reds",
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgrey"},
    )
    ax.set_title("Territorial Vulnerability", fontsize=14)
    ax.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualization] Saved → {output_path}")


def save_cluster_map(data_df, output_path="outputs/maps/clusters_map.png"):
    """
    Save a choropleth map of cluster labels to disk.

    Parameters
    ----------
    data_df : GeoDataFrame with 'cluster' column.
    output_path : str
    """
    _ensure_dir(output_path)
    fig, ax = plt.subplots(figsize=(10, 10))
    data_df.plot(
        column="cluster",
        cmap="Set2",
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgrey"},
    )
    ax.set_title("Territorial Clusters", fontsize=14)
    ax.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualization] Saved → {output_path}")
