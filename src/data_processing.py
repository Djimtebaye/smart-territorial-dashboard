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


def _read_csv_safe(path, sep=";", dtype=None):
    """
    Read a CSV trying utf-8, then utf-8-sig, then latin-1.
    Final fallback: latin-1 + errors='replace' which CANNOT raise UnicodeDecodeError.
    Raises FileNotFoundError if the file does not exist.
    """
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(
                path, sep=sep, dtype=dtype, encoding=enc, low_memory=False
            )
        except UnicodeDecodeError:
            continue

    # Absolute last resort — latin-1 maps every byte 0x00-0xFF, plus replace
    return pd.read_csv(
        path,
        sep=sep,
        dtype=dtype,
        encoding="latin-1",
        encoding_errors="replace",
        low_memory=False,
    )


def load_data():
    """
    Load all datasets from data/raw/.

    Returns
    -------
    gdf      : GeoDataFrame  — commune shapefile
    socio_df : DataFrame     — FILO socio-economic data (IRIS 2018)
    bpe_df   : DataFrame     — BPE equipment data (empty DataFrame if absent)

    Raises
    ------
    FileNotFoundError  if commune.shp or FILO CSV are missing.
    KeyError           if FILO CSV has no 'IRIS' column.
    """
    # 1. Shapefile
    gdf = gpd.read_file("data/raw/commune.shp")

    # 2. FILO socio-economic CSV
    socio_df = _read_csv_safe(
        "data/raw/BASE_TD_FILO_DEC_IRIS_2018.csv", sep=";", dtype=str
    )
    if "IRIS" not in socio_df.columns:
        raise KeyError(
            f"FILO CSV must contain an 'IRIS' column. "
            f"Found: {list(socio_df.columns[:10])}"
        )
    socio_df["codcom"] = socio_df["IRIS"].str[:5]

    # 3. BPE equipment CSV — optional
    bpe_df = pd.DataFrame()
    try:
        bpe_df = _read_csv_safe("data/raw/BPE_24.csv", sep=";")
        if "CODPOS" in bpe_df.columns:
            bpe_df["CODPOS"] = bpe_df["CODPOS"].astype(str)
    except FileNotFoundError:
        pass  # BPE is optional — silently skip

    return gdf, socio_df, bpe_df


def merge_data(gdf, socio_df):
    """
    Merge spatial and socio-economic datasets on commune code.

    Shapefile join key : 'code_posta' (or a known alias).
    FILO join key      : 'codcom' (built from IRIS column).
    """
    if "code_posta" not in gdf.columns:
        for alt in ["CODE_POST", "code_post", "cp", "CODE_POSTA", "codepostal"]:
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
    Coerce service columns to numeric and fill NaN with 0.

    Parameters
    ----------
    data_df         : GeoDataFrame
    service_columns : list of str — columns to clean (default: SERVICE_COLUMNS)
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
