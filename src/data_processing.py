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

# Candidate column names for the postal/commune code in the shapefile
# Add any other name your shapefile might use.
_SHP_CODE_CANDIDATES = [
    "code_posta", "CODE_POSTA",
    "code_post",  "CODE_POST",
    "codepostal", "CODEPOSTAL",
    "code_com",   "CODE_COM",
    "codecom",    "CODECOM",
    "insee",      "INSEE",
    "code",       "CODE",
    "cp",         "CP",
    "id",         "ID",
]


def _read_csv_safe(path, sep=";", dtype=None):
    """
    Read a CSV trying utf-8, utf-8-sig, latin-1 in sequence.
    Final fallback: latin-1 + errors='replace' — physically cannot raise
    UnicodeDecodeError regardless of file content.
    Raises FileNotFoundError if the file does not exist.
    """
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(
                path, sep=sep, dtype=dtype, encoding=enc, low_memory=False
            )
        except UnicodeDecodeError:
            continue
    return pd.read_csv(
        path, sep=sep, dtype=dtype,
        encoding="latin-1", encoding_errors="replace", low_memory=False,
    )


def load_data():
    """
    Load all datasets from data/raw/.

    Returns
    -------
    gdf      : GeoDataFrame  — commune shapefile
    socio_df : DataFrame     — FILO socio-economic data (IRIS 2018)
    bpe_df   : DataFrame     — BPE equipment data (empty if absent)

    Raises
    ------
    FileNotFoundError  if commune.shp or FILO CSV are missing.
    KeyError           if FILO CSV has no 'IRIS' column.
    """
    # 1. Shapefile
    gdf = gpd.read_file("data/raw/commune.shp")

    if len(gdf.columns) <= 1:
        raise RuntimeError(
            "The shapefile loaded with only geometry and no attribute columns. "
            "Make sure commune.dbf and commune.shx are present in data/raw/ "
            f"alongside commune.shp. Columns found: {list(gdf.columns)}"
        )

    # 2. FILO CSV
    socio_df = _read_csv_safe(
        "data/raw/BASE_TD_FILO_DEC_IRIS_2018.csv", sep=";", dtype=str
    )
    if "IRIS" not in socio_df.columns:
        raise KeyError(
            f"FILO CSV must contain an 'IRIS' column. "
            f"Found: {list(socio_df.columns[:10])}"
        )
    socio_df["codcom"] = socio_df["IRIS"].str[:5]

    # 3. BPE CSV — optional
    bpe_df = pd.DataFrame()
    try:
        bpe_df = _read_csv_safe("data/raw/BPE_24.csv", sep=";")
        if "CODPOS" in bpe_df.columns:
            bpe_df["CODPOS"] = bpe_df["CODPOS"].astype(str)
    except FileNotFoundError:
        pass

    return gdf, socio_df, bpe_df


def detect_shp_code_column(gdf):
    """
    Auto-detect which column in the shapefile holds the commune/postal code.

    Returns the column name, or None if nothing matches.
    """
    for candidate in _SHP_CODE_CANDIDATES:
        if candidate in gdf.columns:
            return candidate
    return None


def merge_data(gdf, socio_df):
    """
    Merge spatial and socio-economic datasets on commune/postal code.

    Auto-detects the join key in the shapefile from a list of known aliases.
    If none is found, raises a descriptive KeyError listing all columns.
    """
    shp_key = detect_shp_code_column(gdf)

    if shp_key is None:
        raise KeyError(
            f"Cannot find a commune-code column in the shapefile.\n"
            f"Shapefile columns: {list(gdf.columns)}\n"
            f"Expected one of: {_SHP_CODE_CANDIDATES}\n"
            "Please rename the relevant column in your shapefile to 'code_posta'."
        )

    # Normalise join keys to stripped strings for a clean match
    gdf = gdf.copy()
    gdf[shp_key] = gdf[shp_key].astype(str).str.strip()
    socio_df = socio_df.copy()
    socio_df["codcom"] = socio_df["codcom"].astype(str).str.strip()

    data_df = gdf.merge(
        socio_df,
        left_on=shp_key,
        right_on="codcom",
        how="left",
    )
    return data_df, shp_key   # return key so app can report it


def clean_data(data_df, service_columns=None):
    """
    Coerce service columns to numeric and fill NaN with 0.
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
