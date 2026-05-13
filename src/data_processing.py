import geopandas as gpd
import pandas as pd


SERVICE_COLUMNS = [
    "DEC_D118",
    "DEC_D218",
    "DEC_D418",
    "DEC_D618"
]


def load_data():
    """Load all datasets."""

    gdf = gpd.read_file("data/raw/commune.shp")

    socio_df = pd.read_csv(
        "data/raw/BASE_TD_FILO_DEC_IRIS_2018.csv",
        sep=";"
    )

    socio_df["codcom"] = socio_df["IRIS"].str[:5]

    bpe_df = pd.read_csv(
        "data/raw/BPE_24.csv",
        sep=";", encoding="ANSI"
    )

    bpe_df["CODPOS"] = bpe_df["CODPOS"].astype(str)

    return gdf, socio_df, bpe_df



def merge_data(gdf, socio_df):
    """Merge spatial and socio-economic datasets."""

    data_df = gdf.merge(
        socio_df,
        left_on="code_posta",
        right_on="codcom",
        how="left"
    )

    return data_df



def clean_data(data_df):
    """Clean and prepare service columns."""

    data_df[SERVICE_COLUMNS] = (
        data_df[SERVICE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    return data_df
