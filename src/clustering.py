"""
src/clustering.py
K-Means clustering on standardised service indicators.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def perform_clustering(data_df, service_columns, n_clusters=4):
    """
    Fit K-Means on StandardScaler-normalised service columns and assign cluster labels.

    Parameters
    ----------
    data_df : GeoDataFrame
    service_columns : list of str
        Must be numeric (already cleaned/normalised upstream).
    n_clusters : int
        Number of clusters (K).  Must be >= 2 and <= len(data_df).

    Returns
    -------
    data_df : GeoDataFrame with a new 'cluster' column (int).
    """
    data_df = data_df.copy()   # avoid mutating the cached dataframe

    # Guard: k must not exceed number of rows
    n_clusters = min(n_clusters, len(data_df))
    if n_clusters < 2:
        data_df["cluster"] = 0
        return data_df

    scaler = StandardScaler()
    X = scaler.fit_transform(data_df[service_columns].fillna(0))

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )
    data_df["cluster"] = kmeans.fit_predict(X).astype(int)

    return data_df
