"""
Smart Territorial Dashboard — app/streamlit_app.py
Geospatial analysis for territorial decision support.
"""

import sys
from pathlib import Path

# ── Resolve project root so `src` is importable regardless of cwd ──
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import folium
from streamlit_folium import st_folium

from src.data_processing import load_data, merge_data, clean_data, SERVICE_COLUMNS
from src.scoring import compute_scores
from src.clustering import perform_clustering

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Territorial Dashboard",
    page_icon="🧭",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────
# SIDEBAR — weights + filters
# ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Parameters")

st.sidebar.markdown("**Indicator weights**")
w1 = st.sidebar.slider("DEC_D118", 0.0, 1.0, 0.30, 0.05)
w2 = st.sidebar.slider("DEC_D218", 0.0, 1.0, 0.20, 0.05)
w3 = st.sidebar.slider("DEC_D418", 0.0, 1.0, 0.25, 0.05)
w4 = st.sidebar.slider("DEC_D618", 0.0, 1.0, 0.25, 0.05)
weights = {"DEC_D118": w1, "DEC_D218": w2, "DEC_D418": w3, "DEC_D618": w4}

total_w = round(sum(weights.values()), 2)
if total_w != 1.0:
    st.sidebar.warning(f"⚠️ Weights sum = {total_w} (should be 1.0)")

n_clusters = st.sidebar.slider("Number of clusters (K-Means)", 2, 8, 4)

st.sidebar.markdown("---")
st.sidebar.header("🔎 Dashboard Filters")

# ──────────────────────────────────────────────────────────────────
# DATA LOADING — cached, with error handling
# ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and processing data…")
def prepare_data(w1, w2, w3, w4, n_clusters):
    """Full pipeline: load → merge → clean → score → cluster."""
    weights = {"DEC_D118": w1, "DEC_D218": w2, "DEC_D418": w3, "DEC_D618": w4}

    try:
        gdf, socio_df, _ = load_data()
    except FileNotFoundError as e:
        return None, str(e)

    data_df = merge_data(gdf, socio_df)

    # Guard: keep only SERVICE_COLUMNS that actually exist after merge
    available_cols = [c for c in SERVICE_COLUMNS if c in data_df.columns]
    if not available_cols:
        return None, (
            f"None of the expected columns {SERVICE_COLUMNS} were found after merge. "
            "Check that your FILO CSV uses ';' as separator and contains IRIS/DEC_D* columns."
        )

    data_df = clean_data(data_df, available_cols)
    data_df = compute_scores(data_df, available_cols, weights)
    data_df = perform_clustering(data_df, available_cols, n_clusters=n_clusters)
    data_df = data_df.dropna(subset=["vulnerability_score"])

    # Cast cluster to plain int to avoid numpy.int32 issues in Streamlit widgets
    data_df["cluster"] = data_df["cluster"].astype(int)

    return data_df, None


data_df, error = prepare_data(w1, w2, w3, w4, n_clusters)

# ──────────────────────────────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────────────────────────────
st.title("🧭 Smart Territorial Dashboard")
st.markdown("Interactive geospatial analysis for territorial decision support.")

if error:
    st.error(f"❌ Data loading failed: {error}")
    st.info(
        "Make sure the following files exist:\n"
        "- `data/raw/commune.shp` (+ .dbf, .shx)\n"
        "- `data/raw/BASE_TD_FILO_DEC_IRIS_2018.csv` (sep=`;`)\n"
        "- `data/raw/BPE_24.csv` (sep=`;`)"
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS (depend on loaded data)
# ──────────────────────────────────────────────────────────────────
all_clusters = sorted(data_df["cluster"].unique().tolist())

selected_clusters = st.sidebar.multiselect(
    "Select Cluster(s)",
    options=all_clusters,
    default=all_clusters,
)

vulnerability_threshold = st.sidebar.slider(
    "Minimum Vulnerability Score",
    min_value=float(data_df["vulnerability_score"].min()),
    max_value=float(data_df["vulnerability_score"].max()),
    value=float(data_df["vulnerability_score"].min()),  # default: show all
    step=0.01,
    format="%.2f",
)

# ──────────────────────────────────────────────────────────────────
# FILTER
# ──────────────────────────────────────────────────────────────────
if not selected_clusters:
    st.warning("⚠️ No cluster selected. Please select at least one cluster in the sidebar.")
    st.stop()

filtered_df = data_df[
    (data_df["cluster"].isin(selected_clusters)) &
    (data_df["vulnerability_score"] >= vulnerability_threshold)
]

if filtered_df.empty:
    st.warning(
        "⚠️ No territories match the current filters. "
        "Try lowering the vulnerability threshold or selecting more clusters."
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────
# KPIs
# ──────────────────────────────────────────────────────────────────
st.subheader("📊 Key Indicators")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Territories", len(filtered_df))
col2.metric("Avg Vulnerability", f"{filtered_df['vulnerability_score'].mean():.3f}")
col3.metric("Max Vulnerability", f"{filtered_df['vulnerability_score'].max():.3f}")
col4.metric("Active Clusters", len(selected_clusters))

st.markdown("---")

# ──────────────────────────────────────────────────────────────────
# MAP
# ──────────────────────────────────────────────────────────────────
st.subheader("🗺️ Vulnerability Map")

map_center = [46.5, 2.5]
m = folium.Map(location=map_center, zoom_start=6, tiles="cartodbpositron")

# Safe GeoJson: only pass relevant columns to avoid serialisation issues
map_cols = ["vulnerability_score", "cluster", "geometry"]
map_gdf = filtered_df[[c for c in map_cols if c in filtered_df.columns]].copy()

# Ensure valid geometries
map_gdf = map_gdf[map_gdf.geometry.notna() & map_gdf.geometry.is_valid]

if not map_gdf.empty:
    tooltip_fields = [c for c in ["vulnerability_score", "cluster"] if c in map_gdf.columns]
    tooltip_aliases = ["Vulnerability", "Cluster"][: len(tooltip_fields)]

    folium.GeoJson(
        map_gdf.__geo_interface__,
        style_function=lambda feat: {
            "fillColor": "#d73027",
            "color": "#555",
            "weight": 0.4,
            "fillOpacity": min(
                0.9,
                max(0.1, feat["properties"].get("vulnerability_score", 0)),
            ),
        },
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
        ),
    ).add_to(m)
else:
    st.info("No valid geometries to display on the map.")

st_folium(m, use_container_width=True, height=550)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────
# CLUSTER DISTRIBUTION
# ──────────────────────────────────────────────────────────────────
st.subheader("📈 Cluster Distribution")

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    cluster_count = (
        filtered_df.groupby("cluster")
        .size()
        .reset_index(name="count")
    )
    fig_bar = px.bar(
        cluster_count,
        x="cluster",
        y="count",
        color="cluster",
        labels={"cluster": "Cluster", "count": "Number of territories"},
        title="Territories per Cluster",
        color_continuous_scale="Viridis",
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_chart2:
    fig_box = px.box(
        filtered_df,
        x="cluster",
        y="vulnerability_score",
        color="cluster",
        labels={"cluster": "Cluster", "vulnerability_score": "Vulnerability Score"},
        title="Vulnerability Distribution per Cluster",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────
# TOP PRIORITY AREAS
# ──────────────────────────────────────────────────────────────────
st.subheader("🚨 Top Priority Areas")

# Drop geometry to avoid Streamlit serialisation crash
display_cols = [
    c for c in
    ["vulnerability_score", "service_access_score", "cluster"] + SERVICE_COLUMNS
    if c in filtered_df.columns
]
priority_df = (
    filtered_df[display_cols]
    .sort_values("vulnerability_score", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

st.dataframe(
    priority_df.style.format({c: "{:.4f}" for c in display_cols if c != "cluster"}),
    use_container_width=True,
)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────
# CLUSTER SUMMARY
# ──────────────────────────────────────────────────────────────────
st.subheader("🧠 Cluster Summary")

available_service_cols = [c for c in SERVICE_COLUMNS if c in filtered_df.columns]
cluster_summary = (
    filtered_df.groupby("cluster")[available_service_cols]
    .mean()
    .round(4)
)
st.dataframe(cluster_summary.style.background_gradient(cmap="RdYlGn_r"), use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────
# INSIGHTS
# ──────────────────────────────────────────────────────────────────
st.subheader("💡 Decision Insights")

vuln_by_cluster = filtered_df.groupby("cluster")["vulnerability_score"].mean()

if not vuln_by_cluster.empty:
    highest_cluster = int(vuln_by_cluster.idxmax())
    highest_score = vuln_by_cluster.max()
    lowest_cluster = int(vuln_by_cluster.idxmin())
    lowest_score = vuln_by_cluster.min()

    st.info(
        f"🔴 **Cluster {highest_cluster}** has the highest average vulnerability "
        f"({highest_score:.3f}) → priority intervention zone."
    )
    st.success(
        f"🟢 **Cluster {lowest_cluster}** has the lowest average vulnerability "
        f"({lowest_score:.3f}) → best served territories."
    )

    # Vulnerability bar chart by cluster
    fig_vuln = px.bar(
        vuln_by_cluster.reset_index(),
        x="cluster",
        y="vulnerability_score",
        color="vulnerability_score",
        color_continuous_scale="Reds",
        labels={"cluster": "Cluster", "vulnerability_score": "Avg Vulnerability"},
        title="Average Vulnerability Score by Cluster",
    )
    st.plotly_chart(fig_vuln, use_container_width=True)
