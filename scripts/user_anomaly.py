#!/usr/bin/env python3
"""
Train an Isolation Forest model on user-level features and score anomalies.

Reads:
  - data/processed/user_features.parquet
Writes:
  - outputs/user_anomaly_scores.parquet
Logs:
  - MLflow params, metrics, and output artifact
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import mlflow.sklearn

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "user_features.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "user_anomaly_scores.parquet"
DEFAULT_EXPERIMENT = "user_anomaly_detection"
MODEL_SELECTION_OUTPUT = PROJECT_ROOT / "outputs" / "user_anomaly_model_runs.parquet"

# User features to use for anomaly detection
FEATURE_COLUMNS = [
    "avg_stars_given",
    "pct_5_star_reviews",
    "pct_1_star_reviews",
    "review_count",
    "reviews_per_day",
    "account_age_days",
    "num_friends",
]

# Reproducible, checked-in run configuration.
# Update values here (and commit) to keep experiments fully traceable.
CONFIGS = [
    {"name": "v1_c001", "contamination": 0.01, "n_estimators": 300, "random_state": 42},
    {"name": "v1_c005", "contamination": 0.05, "n_estimators": 300, "random_state": 42},
    {"name": "v1_c010", "contamination": 0.10, "n_estimators": 300, "random_state": 42},
]

# Default configuration to use for downstream scripts
DEFAULT_CONFIG_NAME = "v1_c005"

# Validate the input data has the required columns
def validate_inputs(df: pd.DataFrame) -> None:
    missing_cols = [c for c in ["user_id", *FEATURE_COLUMNS] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input data: {missing_cols}")

# Main function to run the anomaly detection
def main() -> None:
    input_path = DEFAULT_INPUT
    output_path = DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if DEFAULT_CONFIG_NAME not in {cfg["name"] for cfg in CONFIGS}:
        raise ValueError(f"DEFAULT_CONFIG_NAME '{DEFAULT_CONFIG_NAME}' is not in CONFIGS")

    # Read the user features
    print(f"Reading user features: {input_path}")
    user_df = pd.read_parquet(input_path)
    validate_inputs(user_df)

    # Prepare the model input
    model_input = user_df[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        # Convert the column to numeric, handling errors by coercing to NaN
        model_input[col] = pd.to_numeric(model_input[col], errors="coerce")
        # Replace infinity values with NaN
        model_input[col] = model_input[col].replace([np.inf, -np.inf], np.nan)
        # Fill missing values with the median value or 0 if the median is NaN
        median_val = model_input[col].median()
        # If the median is NaN, fill with 0
        model_input[col] = model_input[col].fillna(0 if pd.isna(median_val) else median_val)

    # Set up MLflow
    mlflow.set_experiment(DEFAULT_EXPERIMENT)
    run_summaries = []

    # Run the anomaly detection for each configuration
    for cfg in CONFIGS:
        run_name = f"isolation_forest_{cfg['name']}"
        run_output_path = output_path.with_name(
            f"{output_path.stem}_{cfg['name']}{output_path.suffix}"
        )
        # Start an MLflow run for this configuration
        with mlflow.start_run(run_name=run_name):
            # Log the configuration parameters
            mlflow.log_param("config_name", cfg["name"])
            mlflow.log_param("input_path", str(input_path))
            mlflow.log_param("output_path", str(run_output_path))
            mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
            mlflow.log_param("n_estimators", cfg["n_estimators"])
            mlflow.log_param("contamination", cfg["contamination"])
            mlflow.log_param("random_state", cfg["random_state"])
            mlflow.log_param("n_rows", len(model_input))

            # Train the model
            model = IsolationForest(
                n_estimators=cfg["n_estimators"],
                contamination=cfg["contamination"],
                random_state=cfg["random_state"],
                n_jobs=-1,
            )
            # Fit the model to the data
            model.fit(model_input)

            # Calculate the anomaly scores
            # decision_function: higher => more normal; negate so higher means more anomalous.
            anomaly_score = -model.decision_function(model_input)
            pred = model.predict(model_input)  # -1 anomaly, 1 normal
            is_anomaly = (pred == -1).astype(int)

            # Create a result dataframe with the user_id and the anomaly score
            result_df = user_df[["user_id", *FEATURE_COLUMNS]].copy()
            result_df["anomaly_score"] = anomaly_score
            result_df["is_anomaly"] = is_anomaly
            result_df["anomaly_rank"] = result_df["anomaly_score"].rank(
                method="first", ascending=False
            ).astype(int)
            result_df = result_df.sort_values("anomaly_score", ascending=False)

            print(f"[{cfg['name']}] Writing anomaly scores: {run_output_path}")
            result_df.to_parquet(run_output_path, index=False)
            if cfg["name"] == DEFAULT_CONFIG_NAME:
                # Keep a canonical output path for downstream scripts.
                result_df.to_parquet(output_path, index=False)

            # Calculate the metrics for this configuration
            n_anomalies = int(result_df["is_anomaly"].sum())
            anomaly_rate = float(result_df["is_anomaly"].mean())
            score_mean = float(result_df["anomaly_score"].mean())
            score_std = float(result_df["anomaly_score"].std(ddof=0))
            score_max = float(result_df["anomaly_score"].max())
            score_min = float(result_df["anomaly_score"].min())

            # Log the metrics to MLflow
            mlflow.log_metric("n_users_scored", int(len(result_df)))
            mlflow.log_metric("n_anomalies", n_anomalies)
            mlflow.log_metric("anomaly_rate", anomaly_rate)
            mlflow.log_metric("anomaly_score_mean", score_mean)
            mlflow.log_metric("anomaly_score_std", score_std)
            mlflow.log_metric("anomaly_score_max", score_max)
            mlflow.log_metric("anomaly_score_min", score_min)
            # Log the output artifact
            mlflow.sklearn.log_model(model, artifact_path="model")
            mlflow.log_artifact(str(run_output_path))

            # Log the summary of the run
            run_summaries.append(
                {
                    "config_name": cfg["name"],
                    "n_estimators": cfg["n_estimators"],
                    "contamination": cfg["contamination"],
                    "random_state": cfg["random_state"],
                    "n_users_scored": int(len(result_df)),
                    "n_anomalies": n_anomalies,
                    "anomaly_rate": anomaly_rate,
                    "anomaly_score_mean": score_mean,
                    "anomaly_score_std": score_std,
                }
            )

            # Print the summary of the run
            print(
                f"[{cfg['name']}] Run complete | "
                f"users={len(result_df)} | anomalies={n_anomalies}"
            )

    # Save the summary of the runs
    summary_df = pd.DataFrame(run_summaries).sort_values("config_name")
    summary_df.to_parquet(MODEL_SELECTION_OUTPUT, index=False)
    print(f"Saved model comparison table: {MODEL_SELECTION_OUTPUT}")


if __name__ == "__main__":
    main()
