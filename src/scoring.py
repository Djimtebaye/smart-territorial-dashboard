"""
src/scoring.py
Compute MinMax-normalised service access and vulnerability scores.
"""

from sklearn.preprocessing import MinMaxScaler

# Default weights — can be overridden by passing a custom dict
DEFAULT_WEIGHTS = {
    "DEC_D118": 0.30,
    "DEC_D218": 0.20,
    "DEC_D418": 0.25,
    "DEC_D618": 0.25,
}


def compute_scores(data_df, service_columns, weights=None):
    """
    Normalise service columns and compute weighted vulnerability score.

    Parameters
    ----------
    data_df : GeoDataFrame
    service_columns : list of str
        Columns to normalise (must all be numeric after clean_data).
    weights : dict {col: float}, optional
        Custom weights per column.  Missing columns default to 0.
        Defaults to DEFAULT_WEIGHTS.

    Returns
    -------
    data_df : GeoDataFrame with new columns:
        - service_access_score
        - vulnerability_score
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    data_df = data_df.copy()

    # MinMax normalisation in-place on service columns
    scaler = MinMaxScaler()
    data_df[service_columns] = scaler.fit_transform(data_df[service_columns])

    # Weighted sum — only use columns that are both in service_columns and weights
    data_df["service_access_score"] = sum(
        weights.get(col, 0.0) * data_df[col]
        for col in service_columns
    )

    data_df["vulnerability_score"] = 1.0 - data_df["service_access_score"]

    return data_df
