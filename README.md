# Detecting fraudulent real estate listings with XGBoost

This repository contains my exploratory analysis, feature engineering, and XGBoost experiments from the [KUBIG Team 2 DACON project](https://github.com/KU-BIG/KUBIG_2025_SPRING/tree/main/KUBIG%20CONTEST/DL/Team2). For more information about the project, see the team repository.

## Goal

The task was to identify fraudulent real estate listings from tabular data, without visiting properties or viewing photographs. The dataset was small and imbalanced, making feature construction particularly important.

## My contribution

- Conducted exploratory data analysis (EDA) to examine patterns in the listings.
- Engineered listing and agency features based on those findings.
- Tuned an XGBoost classifier with Optuna and examined gain importance and SHAP values.

## Results

The team compared models using macro F1. The chart below shows my XGBoost approach alongside other team members’ models and comes from the [team README](https://github.com/KU-BIG/KUBIG_2025_SPRING/tree/main/KUBIG%20CONTEST/DL/Team2).

![Team comparison of models using macro F1](https://github.com/user-attachments/assets/7140f5f3-aa5c-4cd9-8a63-952f83060f2b)

XGBoost achieved a macro F1 score of **0.8871**, close to the highest reported score of **0.8881** for TabPFN.

## Discussion

This project showed me how EDA can guide feature engineering and how feature importance can help interpret a model’s predictions. My XGBoost approach performed close to the team’s top-scoring model.

## Code structure

| File | Role |
| --- | --- |
| [`preprocess.py`](preprocess.py) | Missing-value and outlier handling |
| [`features.py`](features.py) | Listing and agency features |
| [`train.py`](train.py) | Tuning, final fit, and submission |
| [`explain.py`](explain.py) | Gain and SHAP feature importance |
| [`notebooks/preprocessed_xgboost.ipynb`](notebooks/preprocessed_xgboost.ipynb) | Original exploratory experiments |
