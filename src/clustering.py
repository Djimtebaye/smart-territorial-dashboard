from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



def perform_clustering(data_df, service_columns, n_clusters=4):
    """Perform KMeans clustering."""

    scaler = StandardScaler()

    X = scaler.fit_transform(
        data_df[service_columns]
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    data_df["cluster"] = kmeans.fit_predict(X)

    return data_df
