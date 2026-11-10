# Fraudulent real estate listing detection with XGBoost

This repository presents my contribution to the [KUBIG Team 2 DACON real estate listing project](https://github.com/KU-BIG/KUBIG_2025_SPRING/tree/main/KUBIG%20CONTEST/DL/Team2). The competition asked us to classify potentially fraudulent listings from tabular listing data. This personal repository reorganizes my local XGBoost notebook by task; the linked team repository describes the broader group project and other models.

## My work

- Investigated missingness and listing features, including listing age and real estate agency information.
- Built smoothed agency risk grades. Within cross-validation, each validation grade uses only the corresponding training fold, and each training grade excludes its own outcome.
- Tuned XGBoost with repeated stratified cross-validation and Optuna, then examined gain-based feature importance and SHAP values.

The code here is adapted from my own exploratory notebook. The notebook includes alternative feature choices and experiments that are not part of the streamlined script.

## Files

| File | Purpose |
| --- | --- |
| `preprocess.py` | Train-fitted missing-value handling and outlier rules |
| `features.py` | Listing features, categorical encoding, and agency risk grades |
| `train.py` | Repeated-CV tuning, final XGBoost fit, and submission output |
| `explain.py` | Gain importance and SHAP plots from a saved model |
| `notebooks/preprocessed_xgboost.ipynb` | Cleaned copy of the original exploratory notebook |

## Use

Obtain `train.csv`, `test.csv`, and `sample_submission.csv` from the [DACON competition](https://dacon.io/competitions/official/236439/data). These datasets are not included here. Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python train.py --train data/train.csv --test data/test.csv \
  --sample-submission data/sample_submission.csv
python explain.py --model outputs/xgboost.json \
  --features outputs/train_features.csv
```

The original experiment used 100 Optuna trials and 5-fold cross-validation repeated 3 times, so a full run can take substantial time. For a short trial, pass `--trials 2 --repeats 1`.

## Interpretation

The agency grade is protected against direct label leakage within each cross-validation split. The other preprocessing steps are fitted once on the full training data before cross-validation, as in the exploratory notebook; therefore, the CV score is an internal tuning measure rather than an independent estimate of generalization. No competition data or trained model is included in this repository.
