import streamlit as st
st.subheader("📈 Cluster Distribution")

cluster_chart = st.histogram(
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
