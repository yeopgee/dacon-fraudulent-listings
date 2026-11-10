"""Train-fitted missing-value handling for the real-estate listing data."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def fit_fill_values(train: pd.DataFrame) -> dict:
    """Fit imputation quantities on training rows only."""
    parking = train.copy()
    parking.loc[parking["주차가능여부"] == "불가능", "총주차대수"] = 0
    valid = parking.dropna(subset=["총주차대수", "총층"])
    parking_model = LinearRegression().fit(valid[["총층"]], valid["총주차대수"])
    return {
        "전용면적": train["전용면적"].median(),
        "총층": train["총층"].median(),
        "욕실수": train["욕실수"].mode().iloc[0],
        "방수": train["방수"].mode().iloc[0],
        "월세": train.loc[train["월세"] != 0, "월세"].mean(),
        "parking_model": parking_model,
    }


def transform_missing(data: pd.DataFrame, fill_values: dict, seed: int = 42) -> pd.DataFrame:
    """Create a missingness indicator and apply train-derived fill values."""
    data = data.copy()
    data["NA여부"] = np.select(
        [
            data["월세"].eq(0),
            data[["방수", "해당층", "욕실수", "총층"]].isna().any(axis=1),
            data[["전용면적", "총주차대수"]].isna().any(axis=1),
        ],
        [3, 2, 1],
        default=0,
    )
    for col in ["전용면적", "총층", "욕실수", "방수"]:
        data[col] = data[col].fillna(fill_values[col])
    rng = np.random.default_rng(seed)
    missing_floor = data["해당층"].isna()
    upper = data.loc[missing_floor, "총층"].clip(lower=1).astype(int).to_numpy()
    data.loc[missing_floor, "해당층"] = [rng.integers(1, top + 1) for top in upper]
    data.loc[data["주차가능여부"] == "불가능", "총주차대수"] = 0
    missing_parking = data["총주차대수"].isna()
    if missing_parking.any():
        prediction = fill_values["parking_model"].predict(data.loc[missing_parking, ["총층"]])
        data.loc[missing_parking, "총주차대수"] = np.maximum(0, np.round(prediction))
    data.loc[data["월세"] == 0, "월세"] = fill_values["월세"]
    return data


def filter_training_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """Apply the exploratory outlier rules from the original notebook."""
    keep = ~((train["총주차대수"] == 590) | ((train["총주차대수"] > 120) & (train["총층"] < 10)))
    return train.loc[keep & (train["관리비"] < 40)].copy()
