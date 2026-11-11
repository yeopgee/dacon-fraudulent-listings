"""Feature engineering, including fold-specific agency risk grades."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

TARGET = "허위매물여부"
AGENCY = "중개사무소"
ONE_HOT_COLUMNS = ["방향", "매물확인방식", "제공플랫폼"]


def encode_agency_risk(train_agency, y_train, evaluation_agency, alpha=1.5, n_grades=8):
    """Use leave-one-out rates for training rows and training-only rates for evaluation."""
    train_agency = train_agency.astype("string").fillna("__MISSING_AGENCY__")
    evaluation_agency = evaluation_agency.astype("string").fillna("__MISSING_AGENCY__")
    y_train = pd.Series(np.asarray(y_train, dtype=float), index=train_agency.index)
    stats = pd.DataFrame({AGENCY: train_agency, TARGET: y_train}).groupby(AGENCY)[TARGET].agg(["sum", "count"])
    train_sum = train_agency.map(stats["sum"]).astype(float) - y_train
    train_count = train_agency.map(stats["count"]).astype(float) - 1
    train_rate = (train_sum + alpha) / (train_count + 2 * alpha)
    eval_sum = evaluation_agency.map(stats["sum"]).fillna(0).astype(float)
    eval_count = evaluation_agency.map(stats["count"]).fillna(0).astype(float)
    eval_rate = (eval_sum + alpha) / (eval_count + 2 * alpha)
    inner_bins = np.unique(np.quantile(train_rate, np.linspace(0, 1, n_grades + 1)))[1:-1]
    return (
        pd.Series(np.digitize(train_rate, inner_bins), index=train_agency.index, name="신뢰등급"),
        pd.Series(np.digitize(eval_rate, inner_bins), index=evaluation_agency.index, name="신뢰등급"),
    )


def fit_feature_state(train: pd.DataFrame) -> dict:
    """Fit date reference, rare platform levels, and categorical encoder on train."""
    state = {"max_date": pd.to_datetime(train["게재일"]).max()}
    counts = train["제공플랫폼"].value_counts(normalize=True)
    state["rare_platforms"] = set(counts[counts < 0.02].index)
    platform = train[ONE_HOT_COLUMNS].copy()
    platform["제공플랫폼"] = platform["제공플랫폼"].replace(list(state["rare_platforms"]), "기타")
    state["encoder"] = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False).fit(platform)
    return state


def transform_features(data: pd.DataFrame, state: dict, is_train=False) -> pd.DataFrame:
    """Apply train-fitted feature choices without changing the input frame."""
    result = data.drop(columns=["ID", "주차가능여부", "해당층", "욕실수"], errors="ignore").copy()
    result["게재일수"] = (state["max_date"] - pd.to_datetime(result.pop("게재일"))).dt.days.clip(lower=0)
    result["총주차대수"] = np.log1p(result["총주차대수"])
    categorical = result[ONE_HOT_COLUMNS].copy()
    # As in the notebook, collapse rare platforms in training only. Unseen
    # test levels are handled by the fitted encoder's handle_unknown setting.
    if is_train:
        categorical["제공플랫폼"] = categorical["제공플랫폼"].replace(list(state["rare_platforms"]), "기타")
    encoded = state["encoder"].transform(categorical)
    encoded = pd.DataFrame(encoded, columns=state["encoder"].get_feature_names_out(ONE_HOT_COLUMNS), index=result.index)
    return pd.concat([result.drop(columns=ONE_HOT_COLUMNS), encoded], axis=1)
