#!/usr/bin/env python3
"""
Isolation Forest hyperparameter sweep for user anomaly detection.
Focuses on comparing feature sampling (max_features) performance.

Reads:
  - data/processed/user_features.parquet
Writes:
  - outputs/user_anomaly_scores_<config>.parquet (one per config)
  - outputs/user_comparison_summary.parquet      (table of results)
"""

import os
import sys
from pathlib import Path
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "user_features.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
SUMMARY_OUTPUT = DEFAULT_OUTPUT_DIR / "user_comparison_summary.parquet"

TRACKING_URI_DEFAULT = "http://localhost:5001"
DEFAULT_EXPERIMENT = "user_anomaly_detection"

FEATURE_COLUMNS = [
    "avg_stars_given",
    "pct_5_star_reviews",
    "pct_1_star_reviews",
    "review_count",
    "reviews_per_day",
    "account_age_days",
    "num_friends",
]

# Configurations for comparison
# We keep 300/0.05/1.0 as the primary, and 0.7 as the comparison
CONFIGS = [
    {"name": "full_features", "contamination": 0.05, "n_estimators": 300, "max_features": 1.0},
    {"name": "subsampled_features", "contamination": 0.05, "n_estimators": 300, "max_features": 0.7},
]

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drops NaNs and prepares numeric matrix."""
    df = df.dropna(subset=["account_age_days", "num_friends"])
    model_input = df[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        model_input[col] = pd.to_numeric(model_input[col], errors="coerce")
        model_input[col] = model_input[col].replace([np.inf, -np.inf], np.nan)
        model_input[col] = model_input[col].fillna(model_input[col].median())
    return df, model_input

def main():
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading data: {DEFAULT_INPUT}")
    raw_df = pd.read_parquet(DEFAULT_INPUT)
    user_df, model_input = prepare_data(raw_df)
    
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI_DEFAULT))
    mlflow.set_experiment(DEFAULT_EXPERIMENT)
    
    run_history = []

    for cfg in CONFIGS:
        run_name = f"IF_Sweep_{cfg['name']}"
        print(f"--- Running Config: {cfg['name']} ---")
        
        with mlflow.start_run(run_name=run_name):
            # Log Parameters
            mlflow.log_params(cfg)
            mlflow.log_param("n_rows", len(model_input))

            # Train Model
            model = IsolationForest(
                n_estimators=cfg["n_estimators"],
                contamination=cfg["contamination"],
                max_features=cfg["max_features"],
                random_state=42,
                n_jobs=-1
            )
            model.fit(model_input)

            # Scores & Metrics
            # Negate decision_function: higher = more anomalous
            scores = -model.decision_function(model_input)
            preds = (model.predict(model_input) == -1).astype(int)

            mlflow.log_metric("n_anomalies", int(preds.sum()))
            mlflow.log_metric("score_mean", float(scores.mean()))
            mlflow.log_metric("score_std", float(scores.std()))
            mlflow.log_metric("score_max", float(scores.max()))

            # Save per-run output
            out_file = DEFAULT_OUTPUT_DIR / f"user_anomaly_scores_{cfg['name']}.parquet"
            result_df = user_df[["user_id"]].copy()
            result_df["anomaly_score"] = scores
            result_df["is_anomaly"] = preds
            result_df.to_parquet(out_file, index=False)
            
            mlflow.log_artifact(str(out_file))
            
            run_history.append({
                "config": cfg["name"],
                "max_features": cfg["max_features"],
                "score_std": scores.std(),
                "n_anomalies": preds.sum()
            })

    # Final summary to console
    summary_df = pd.DataFrame(run_history)
    summary_df.to_parquet(SUMMARY_OUTPUT, index=False)
    print("\nComparison Summary:")
    print(summary_df)

if __name__ == "__main__":
    main()