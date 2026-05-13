"""
Smart Territorial Dashboard — app/streamlit_app.py
Geospatial analysis for territorial decision support.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium

from src.data_processing import (
    load_data, merge_data, clean_data, SERVICE_COLUMNS, detect_shp_code_column
)
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
# SIDEBAR
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
# DATA PIPELINE — cached
# ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and processing data…")
def prepare_data(w1, w2, w3, w4, n_clusters):
    weights = {"DEC_D118": w1, "DEC_D218": w2, "DEC_D418": w3, "DEC_D618": w4}

    # ── Load ──────────────────────────────────────────────────────
    try:
        gdf, socio_df, _ = load_data()
    except (FileNotFoundError, RuntimeError) as e:
        return None, str(e), {}

    # ── Shapefile diagnostic info (passed back for display) ───────
    shp_info = {
        "columns": list(gdf.columns),
        "rows": len(gdf),
        "crs": str(gdf.crs),
        "detected_key": detect_shp_code_column(gdf),
    }

    # ── Merge ─────────────────────────────────────────────────────
    try:
        data_df, shp_key = merge_data(gdf, socio_df)
        shp_info["used_key"] = shp_key
    except KeyError as e:
        return None, str(e), shp_info

    # ── Guard: keep only SERVICE_COLUMNS that exist ───────────────
    available_cols = [c for c in SERVICE_COLUMNS if c in data_df.columns]
    if not available_cols:
        return None, (
            f"None of the expected columns {SERVICE_COLUMNS} were found after merge.\n"
            f"Columns available: {[c for c in data_df.columns if 'DEC' in c or 'dec' in c]}\n"
            "Check that your FILO CSV uses ';' as separator and contains IRIS/DEC_D* columns."
        ), shp_info

    # ── Clean → Score → Cluster ───────────────────────────────────
    try:
        data_df = clean_data(data_df, available_cols)
    except KeyError as e:
        return None, str(e), shp_info

    data_df = compute_scores(data_df, available_cols, weights)
    data_df = perform_clustering(data_df, available_cols, n_clusters=n_clusters)
    data_df = data_df.dropna(subset=["vulnerability_score"])
    data_df["cluster"] = data_df["cluster"].astype(int)

    return data_df, None, shp_info


data_df, error, shp_info = prepare_data(w1, w2, w3, w4, n_clusters)

# ──────────────────────────────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────────────────────────────
st.title("🧭 Smart Territorial Dashboard")
st.markdown("Interactive geospatial analysis for territorial decision support.")

# ── Shapefile diagnostic expander (always visible for debugging) ──
if shp_info:
    with st.expander("🔬 Shapefile diagnostic", expanded=bool(error)):
        col_a, col_b = st.columns(2)
        col_a.markdown(f"**Rows:** {shp_info.get('rows', '—')}")
        col_a.markdown(f"**CRS:** `{shp_info.get('crs', '—')}`")
        col_b.markdown(f"**Detected join key:** `{shp_info.get('detected_key', 'NOT FOUND ❌')}`")
        col_b.markdown(f"**Used join key:** `{shp_info.get('used_key', '—')}`")
        st.markdown("**All shapefile columns:**")
        st.code(str(shp_info.get("columns", [])))

if error:
    st.error(f"❌ Pipeline error:\n\n{error}")
    st.info(
        "**Checklist:**\n"
        "- `data/raw/commune.shp` + `commune.dbf` + `commune.shx` all present?\n"
        "- `data/raw/BASE_TD_FILO_DEC_IRIS_2018.csv` present, separator = `;`?\n"
        "- Shapefile has a commune/postal code column? (see diagnostic above)"
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ──────────────────────────────────────────────────────────────────
all_clusters = sorted(data_df["cluster"].unique().tolist())

selected_clusters = st.sidebar.multiselect(
    "Select Cluster(s)", options=all_clusters, default=all_clusters
)
vulnerability_threshold = st.sidebar.slider(
    "Minimum Vulnerability Score",
    min_value=float(data_df["vulnerability_score"].min()),
    max_value=float(data_df["vulnerability_score"].max()),
    value=float(data_df["vulnerability_score"].min()),
    step=0.01, format="%.2f",
)

# ──────────────────────────────────────────────────────────────────
# FILTER
# ──────────────────────────────────────────────────────────────────
if not selected_clusters:
    st.warning("⚠️ No cluster selected. Please select at least one in the sidebar.")
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

m = folium.Map(location=[46.5, 2.5], zoom_start=6, tiles="cartodbpositron")

map_gdf = filtered_df[
    [c for c in ["vulnerability_score", "cluster", "geometry"] if c in filtered_df.columns]
].copy()
map_gdf = map_gdf[map_gdf.geometry.notna() & map_gdf.geometry.is_valid]

if not map_gdf.empty:
    tooltip_fields = [c for c in ["vulnerability_score", "cluster"] if c in map_gdf.columns]
    folium.GeoJson(
        map_gdf.__geo_interface__,
        style_function=lambda feat: {
            "fillColor": "#d73027",
            "color": "#555",
            "weight": 0.4,
            "fillOpacity": min(0.9, max(0.1, feat["properties"].get("vulnerability_score", 0))),
        },
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=["Vulnerability", "Cluster"][: len(tooltip_fields)],
        ),
    ).add_to(m)

st_folium(m, use_container_width=True, height=520)
st.markdown("---")

# ──────────────────────────────────────────────────────────────────
# CLUSTER DISTRIBUTION
# ──────────────────────────────────────────────────────────────────
st.subheader("📈 Cluster Distribution")
col_c1, col_c2 = st.columns(2)

with col_c1:
    cluster_count = filtered_df.groupby("cluster").size().reset_index(name="count")
    fig_bar = px.bar(
        cluster_count, x="cluster", y="count", color="cluster",
        labels={"cluster": "Cluster", "count": "Territories"},
        title="Territories per Cluster", color_continuous_scale="Viridis",
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_c2:
    fig_box = px.box(
        filtered_df, x="cluster", y="vulnerability_score", color="cluster",
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

display_cols = [
    c for c in ["vulnerability_score", "service_access_score", "cluster"] + SERVICE_COLUMNS
    if c in filtered_df.columns
]
priority_df = (
    filtered_df[display_cols]
    .sort_values("vulnerability_score", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
fmt = {c: "{:.4f}" for c in display_cols if c != "cluster"}
st.dataframe(priority_df.style.format(fmt), use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────
# CLUSTER SUMMARY
# ──────────────────────────────────────────────────────────────────
st.subheader("🧠 Cluster Summary")
available_service_cols = [c for c in SERVICE_COLUMNS if c in filtered_df.columns]
cluster_summary = filtered_df.groupby("cluster")[available_service_cols].mean().round(4)
st.dataframe(
    cluster_summary.style.background_gradient(cmap="RdYlGn_r"),
    use_container_width=True
)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────
# INSIGHTS
# ──────────────────────────────────────────────────────────────────
st.subheader("💡 Decision Insights")
vuln_by_cluster = filtered_df.groupby("cluster")["vulnerability_score"].mean()

if not vuln_by_cluster.empty:
    highest_cluster = int(vuln_by_cluster.idxmax())
    lowest_cluster  = int(vuln_by_cluster.idxmin())

    st.info(
        f"🔴 **Cluster {highest_cluster}** — highest avg vulnerability "
        f"({vuln_by_cluster.max():.3f}) → priority intervention zone."
    )
    st.success(
        f"🟢 **Cluster {lowest_cluster}** — lowest avg vulnerability "
        f"({vuln_by_cluster.min():.3f}) → best served territories."
    )

    fig_vuln = px.bar(
        vuln_by_cluster.reset_index(),
        x="cluster", y="vulnerability_score", color="vulnerability_score",
        color_continuous_scale="Reds",
        labels={"cluster": "Cluster", "vulnerability_score": "Avg Vulnerability"},
        title="Average Vulnerability Score by Cluster",
    )
    st.plotly_chart(fig_vuln, use_container_width=True)
