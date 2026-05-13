from sklearn.preprocessing import MinMaxScaler


WEIGHTS = {
    "DEC_D118": 0.30,
    "DEC_D218": 0.20,
    "DEC_D418": 0.25,
    "DEC_D618": 0.25
}



def compute_scores(data_df, service_columns):
    """Compute accessibility and vulnerability scores."""

    scaler = MinMaxScaler()

    data_df[service_columns] = scaler.fit_transform(
        data_df[service_columns]
    )

    data_df["service_access_score"] = (
        WEIGHTS["DEC_D118"] * data_df["DEC_D118"] +
        WEIGHTS["DEC_D218"] * data_df["DEC_D218"] +
        WEIGHTS["DEC_D418"] * data_df["DEC_D418"] +
        WEIGHTS["DEC_D618"] * data_df["DEC_D618"]
    )

    data_df["vulnerability_score"] = (
        1 - data_df["service_access_score"]
    )

    return data_df
