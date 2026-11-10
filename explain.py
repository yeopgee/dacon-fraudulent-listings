"""Plot feature importance and SHAP values for a saved XGBoost model."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True, help="Preprocessed features used to fit the model")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = xgb.Booster(model_file=str(args.model))
    features = pd.read_csv(args.features)
    xgb.plot_importance(model, importance_type="gain", max_num_features=10)
    plt.tight_layout()
    plt.savefig(args.output_dir / "feature_importance.png", dpi=180)
    plt.close()
    values = shap.Explainer(model)(features)
    shap.summary_plot(values, features, show=False)
    plt.tight_layout()
    plt.savefig(args.output_dir / "shap_summary.png", dpi=180)


if __name__ == "__main__":
    main()
