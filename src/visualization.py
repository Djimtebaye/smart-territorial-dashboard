import matplotlib.pyplot as plt



def save_vulnerability_map(data_df):
    fig, ax = plt.subplots(figsize=(10, 10))

    data_df.plot(
        column="vulnerability_score",
        cmap="Reds",
        legend=True,
        ax=ax
    )

    ax.set_title("Territorial Vulnerability")
    ax.axis("off")

    plt.savefig(
        "outputs/maps/vulnerability_map.png",
        dpi=300,
        bbox_inches="tight"
    )



def save_cluster_map(data_df):
    fig, ax = plt.subplots(figsize=(10, 10))

    data_df.plot(
        column="cluster",
        cmap="Set2",
        legend=True,
        ax=ax
    )

    ax.set_title("Territorial Clusters")
    ax.axis("off")

    plt.savefig(
        "outputs/maps/clusters_map.png",
        dpi=300,
        bbox_inches="tight"
    )
