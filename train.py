"""Tune XGBoost with repeated stratified CV and create a DACON submission.

This refactors the user's exploratory notebook into a command-line workflow.
The competition CSV files are intentionally not included in this repository.
"""

import argparse
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from features import AGENCY, TARGET, encode_agency_risk, fit_feature_state, transform_features
from preprocess import filter_training_outliers, fit_fill_values, transform_missing


def add_agency_grade(train_x, train_y, evaluation_x):
    """Fit target-derived agency grades on the current training fold only."""
    train_x = train_x.copy()
    evaluation_x = evaluation_x.copy()
    train_grade, evaluation_grade = encode_agency_risk(
        train_x[AGENCY], train_y, evaluation_x[AGENCY]
    )
    train_x["신뢰등급"] = train_grade
    evaluation_x["신뢰등급"] = evaluation_grade
    return train_x.drop(columns=AGENCY), evaluation_x.drop(columns=AGENCY)


def prepare_data(train_raw, test_raw):
    """Apply the notebook's preprocessing and feature engineering choices."""
    fill_values = fit_fill_values(train_raw)
    train = filter_training_outliers(transform_missing(train_raw, fill_values))
    # The notebook computes test-set fill values from its already transformed
    # training frame, while retaining the first transformation for training.
    test_fill_values = fit_fill_values(transform_missing(train_raw, fill_values))
    test = transform_missing(test_raw, test_fill_values)
    feature_state = fit_feature_state(train)
    return transform_features(train, feature_state, is_train=True), transform_features(test, feature_state)


def tune_model(x, y, trials, folds, repeats, seed, optuna_seed=None):
    class_weight = (y == 0).sum() / (y == 1).sum()
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": seed,
            "scale_pos_weight": class_weight,
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        rounds = trial.suggest_int("num_boost_round", 100, 1000)
        scores = []
        for training, validation in cv.split(x, y):
            x_train, x_valid = add_agency_grade(
                x.iloc[training], y.iloc[training], x.iloc[validation]
            )
            dtrain = xgb.DMatrix(x_train, label=y.iloc[training])
            dvalid = xgb.DMatrix(x_valid, label=y.iloc[validation])
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=rounds,
                evals=[(dvalid, "validation")],
                early_stopping_rounds=10,
                verbose_eval=False,
            )
            prediction = (model.predict(dvalid) > 0.5).astype(int)
            # Match the notebook's positive-class (binary) F1 tuning objective;
            # the team's final comparison chart reports macro F1 separately.
            scores.append(f1_score(y.iloc[validation], prediction))
        return float(np.mean(scores))

    # The notebook leaves Optuna's search unseeded; an optional seed supports
    # repeatable reruns without changing the default experiment.
    sampler = optuna.samplers.TPESampler(seed=optuna_seed) if optuna_seed is not None else None
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials)
    return study.best_params, study.best_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--model-output", type=Path, default=Path("outputs/xgboost.json"))
    parser.add_argument("--features-output", type=Path, default=Path("outputs/train_features.csv"))
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optuna-seed", type=int)
    args = parser.parse_args()

    train, test = prepare_data(pd.read_csv(args.train), pd.read_csv(args.test))
    y = train.pop(TARGET)
    test = test.drop(columns=TARGET, errors="ignore")
    params, score = tune_model(train, y, args.trials, args.folds, args.repeats,
                               args.seed, args.optuna_seed)
    rounds = params.pop("num_boost_round")
    class_weight = (y == 0).sum() / (y == 1).sum()
    params.update({"objective": "binary:logistic", "eval_metric": "logloss",
                   "random_state": args.seed, "scale_pos_weight": class_weight})
    train_final, test_final = add_agency_grade(train, y, test)
    model = xgb.train(params, xgb.DMatrix(train_final, label=y), num_boost_round=rounds)
    predicted = (model.predict(xgb.DMatrix(test_final)) > 0.5).astype(int)
    submission = pd.read_csv(args.sample_submission)
    if len(submission) != len(predicted):
        raise ValueError("Submission template and test set have different row counts")
    submission[TARGET] = predicted
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    model.save_model(args.model_output)
    train_final.to_csv(args.features_output, index=False)
    print(f"Mean repeated-CV F1: {score:.4f}; saved {args.output}, {args.model_output}, and {args.features_output}")


if __name__ == "__main__":
    main()
