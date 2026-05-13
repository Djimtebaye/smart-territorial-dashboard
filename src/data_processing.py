"""
src/data_processing.py
Load, merge, and clean territorial datasets.
"""

import geopandas as gpd
import pandas as pd

SERVICE_COLUMNS = [
    "DEC_D118",
    "DEC_D218",
    "DEC_D418",
    "DEC_D618",
]


def load_data():
    """
    Load all datasets from data/raw/.

    Returns
    -------
    gdf : GeoDataFrame   — commune shapefile
    socio_df : DataFrame — FILO socio-economic data
    bpe_df : DataFrame   — BPE equipment data

    Raises
    ------
    FileNotFoundError if any required file is missing.
    """
    gdf = gpd.read_file("data/raw/commune.shp")

    socio_df = pd.read_csv(
        "data/raw/BASE_TD_FILO_DEC_IRIS_2018.csv",
        sep=";",
        dtype=str,          # read as str first to avoid mixed-type issues
        low_memory=False,
    )
    # Build commune code from IRIS (first 5 chars)
    if "IRIS" not in socio_df.columns:
        raise KeyError("FILO CSV must contain an 'IRIS' column.")
    socio_df["codcom"] = socio_df["IRIS"].str[:5]

    # BPE is optional — load it but don't crash if absent
    try:
        bpe_df = pd.read_csv(
            "data/raw/BPE_24.csv",
            sep=";",
            encoding="cp1252",
            low_memory=False,
        )
        bpe_df["CODPOS"] = bpe_df["CODPOS"].astype(str)
    except FileNotFoundError:
        bpe_df = pd.DataFrame()   # empty placeholder

    return gdf, socio_df, bpe_df


def merge_data(gdf, socio_df):
    """
    Merge spatial and socio-economic datasets on commune code.

    The shapefile join key is 'code_posta'; the FILO key is 'codcom'.
    Adjust these if your shapefile uses a different column name.
    """
    if "code_posta" not in gdf.columns:
        # Fallback: try common alternative names
        for alt in ["CODE_POST", "code_post", "cp", "CODE_POSTA"]:
            if alt in gdf.columns:
                gdf = gdf.rename(columns={alt: "code_posta"})
                break
        else:
            raise KeyError(
                f"Shapefile columns: {list(gdf.columns)}. "
                "Expected 'code_posta' (or a known alias). "
                "Please rename the postal-code column to 'code_posta'."
            )

    data_df = gdf.merge(
        socio_df,
        left_on="code_posta",
        right_on="codcom",
        how="left",
    )
    return data_df


def clean_data(data_df, service_columns=None):
    """
    Coerce SERVICE_COLUMNS to numeric and fill NaN with 0.

    Parameters
    ----------
    data_df : GeoDataFrame
    service_columns : list of str, optional
        Subset of SERVICE_COLUMNS actually present in data_df.
        Defaults to SERVICE_COLUMNS (all four).
    """
    if service_columns is None:
        service_columns = SERVICE_COLUMNS

    missing = [c for c in service_columns if c not in data_df.columns]
    if missing:
        raise KeyError(
            f"Columns {missing} not found after merge. "
            "Check that the FILO CSV contains DEC_D118, DEC_D218, DEC_D418, DEC_D618."
        )

    data_df = data_df.copy()
    data_df[service_columns] = (
        data_df[service_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    return data_df
