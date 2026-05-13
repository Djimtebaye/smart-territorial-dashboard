import streamlit as st
import pandas as pd
import plotly.express as px
import folium

from streamlit_folium import st_folium

from src.data_processing import (
    load_data,
    merge_data,
    clean_data,
    SERVICE_COLUMNS
)

from src.scoring import compute_scores
from src.clustering import perform_clustering


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Territorial Dashboard",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🧭 Smart Territorial Dashboard")
st.markdown(
    "Interactive geospatial analysis for territorial decision support"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data

def prepare_data():

    gdf, socio_df, bpe_df = load_data()

    data_df = merge_data(gdf, socio_df)

    data_df = clean_data(data_df)

    data_df = compute_scores(
        data_df,
        SERVICE_COLUMNS
    )

    data_df = perform_clustering(
        data_df,
        SERVICE_COLUMNS,
        n_clusters=4
    )

    data_df = data_df.dropna(
        subset=["vulnerability_score"]
    )

    return data_df



data_df = prepare_data()



# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

selected_cluster = st.sidebar.multiselect(
    "Select Cluster(s)",
    options=sorted(data_df["cluster"].unique()),
    default=sorted(data_df["cluster"].unique())
)

vulnerability_threshold = st.sidebar.slider(
    "Minimum Vulnerability Score",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# --------------------------------------------------
# FILTER DATA
# --------------------------------------------------

filtered_df = data_df[
    (data_df["cluster"].isin(selected_cluster)) &
    (data_df["vulnerability_score"] >= vulnerability_threshold)
]


# --------------------------------------------------
# KPIs
# --------------------------------------------------

st.subheader("📊 Key Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Territories",
        len(filtered_df)
    )

with col2:
    st.metric(
        "Average Vulnerability",
        round(
            filtered_df["vulnerability_score"].mean(),
            2
        )
    )

with col3:
    st.metric(
        "Clusters",
        len(selected_cluster)
    )


# --------------------------------------------------
# MAP
# --------------------------------------------------

st.subheader("🗺️ Vulnerability Map")

map_center = [46.5, 2.5]

m = folium.Map(
    location=map_center,
    zoom_start=6,
    tiles="cartodbpositron"
)

folium.GeoJson(
    filtered_df.to_json(),
    tooltip=folium.GeoJsonTooltip(
        fields=[
            "vulnerability_score",
            "cluster"
        ],
        aliases=[
            "Vulnerability",
            "Cluster"
        ]
    )
).add_to(m)

st_folium(m, width=1200, height=600)


# --------------------------------------------------
# CLUSTER DISTRIBUTION
# --------------------------------------------------

st.subheader("📈 Cluster Distribution")

cluster_chart = px.histogram(
    filtered_df,
    x="cluster"
)

st.plotly_chart(
    cluster_chart,
    use_container_width=True
)


# --------------------------------------------------
# TOP PRIORITY AREAS
# --------------------------------------------------

st.subheader("🚨 Top Priority Areas")

priority_df = (
    filtered_df
    .sort_values(
        "vulnerability_score",
        ascending=False
    )
    .head(10)
)

st.dataframe(priority_df)


# --------------------------------------------------
# CLUSTER SUMMARY
# --------------------------------------------------

st.subheader("🧠 Cluster Summary")

cluster_summary = (
    filtered_df
    .groupby("cluster")[SERVICE_COLUMNS]
    .mean()
)

st.dataframe(cluster_summary)


# --------------------------------------------------
# INSIGHTS
# --------------------------------------------------

st.subheader("💡 Decision Insights")

highest_cluster = (
    filtered_df
    .groupby("cluster")["vulnerability_score"]
    .mean()
    .idxmax()
)

st.info(
    f"Cluster {highest_cluster} presents the highest average vulnerability level and should be prioritized for intervention."
)
