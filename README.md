# Detecting fraudulent real estate listings from tabular data

This repository contains my exploratory analysis, feature engineering, and XGBoost experiments from the [KUBIG Team 2 DACON project](https://github.com/KU-BIG/KUBIG_2025_SPRING/tree/main/KUBIG%20CONTEST/DL/Team2). For more information about the project, see the team repository.

## Goal

The task was to identify fraudulent real estate listings from tabular data, without visiting properties or viewing photographs. The dataset was small and imbalanced, making feature construction particularly important.

## My contribution

- Conducted exploratory data analysis (EDA) to examine patterns in the variables.
- Engineered listing and agency features based on EDA, including an agency risk grade computed from additive smoothing.
- Tuned an XGBoost classifier with Optuna and examined gain importance and SHAP values.

## Results

The team compared models using macro F1. The chart below shows my XGBoost approach alongside other team members’ models and comes from the [team README](https://github.com/KU-BIG/KUBIG_2025_SPRING/tree/main/KUBIG%20CONTEST/DL/Team2).

![Team comparison of models using macro F1](https://github.com/user-attachments/assets/7140f5f3-aa5c-4cd9-8a63-952f83060f2b)

XGBoost achieved a macro F1 score of **0.8871**, close to the highest reported score of **0.8881** for TabPFN. The notebook and [`train.py`](train.py) tune XGBoost using binary F1; the chart reports the team's macro F1 comparison.

## Interpretation

This project showed me how EDA can guide feature engineering and how feature importance can help interpret predictions. My XGBoost model performed well, while the transformer-based TabPFN model scored slightly higher on this artificial tabular dataset.

## Code structure

| File | Role |
| --- | --- |
| [`preprocess.py`](preprocess.py) | Missing-value and outlier handling |
| [`features.py`](features.py) | Listing and agency features |
| [`train.py`](train.py) | Tuning, final fit, and submission |
| [`explain.py`](explain.py) | Gain and SHAP feature importance |
| [`notebooks/preprocessed_xgboost.ipynb`](notebooks/preprocessed_xgboost.ipynb) | Original exploratory experiments |

## Running the code

Run commands from the repository root. Use a separate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For interactive exploration, install `requirements-notebook.txt` instead. The
notebook now includes section headings linking each stage to its Python module.
The Python scripts expose the main workflow; the notebook also preserves optional
plots and alternative experiments. Datasets, checkpoints, and generated outputs
are not committed.

On macOS, XGBoost also needs the OpenMP runtime (`libomp`), which is not
installed by pip. Install it with `brew install libomp` if the XGBoost import
reports a missing `libomp.dylib`.

Place the competition's `train.csv`, `test.csv`, and `sample_submission.csv` under
`data/`, then run:

```bash
python train.py --train data/train.csv --test data/test.csv --sample-submission data/sample_submission.csv
python explain.py --model outputs/xgboost.json --features outputs/train_features.csv --output-dir outputs
```

Use `--trials`, `--folds`, and `--repeats` to control tuning cost. `--optuna-seed`
optionally seeds the search; the original experiment left it unseeded.

The CLI corresponds to the notebook's smoothed agency-risk-grade branch. The
continuous agency-rate/count features and random-forest imputation remain in the
notebook as alternatives. The notebook restores a saved copy of the imputed data
before switching branches. Its final fit now explicitly retains the same binary
classification objective as the tuning loop and CLI.

Agency target encoding is fitted within each CV training fold, with leave-one-out
rates for training rows. Other preprocessing and the class-weight calculation
remain outside CV to preserve the experiment; therefore this is not a completely
fold-isolated preprocessing pipeline. Binary F1 drives tuning, while the README's
team comparison reports macro F1.
