#!/usr/bin/env python3
"""
Model sweep script for user anomaly detection (Isolation Forest + LOF).

Reads:
  - data/processed/user_features.parquet
Writes:
  - outputs/user_anomaly_scores.parquet
Logs:
  - MLflow params, metrics, and output artifact
"""

import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import mlflow.sklearn
import sys
import sklearn


# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "user_features.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "user_anomaly_scores.parquet"
DEFAULT_EXPERIMENT = "user_anomaly_detection"
# Match local server: mlflow server --host 127.0.0.1 --port 5001
# Override: export MLFLOW_TRACKING_URI="file:/path/to/mlruns" (no server)
TRACKING_URI_DEFAULT = "http://localhost:5001"
MODEL_SELECTION_OUTPUT = PROJECT_ROOT / "outputs" / "user_anomaly_model_runs.parquet"
WINNER_OUTPUT = PROJECT_ROOT / "outputs" / "user_anomaly_scores_winner.parquet"

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
    {"name": "v1_c001_mf10", "contamination": 0.01, "n_estimators": 300, "max_features": 1.0, "random_state": 42},
    {"name": "v1_c005_mf10", "contamination": 0.05, "n_estimators": 300, "max_features": 1.0, "random_state": 42},
    {"name": "v1_c010_mf10", "contamination": 0.10, "n_estimators": 300, "max_features": 1.0, "random_state": 42},
    {"name": "v1_c001_mf07", "contamination": 0.01, "n_estimators": 300, "max_features": 0.7, "random_state": 42},
    {"name": "v1_c005_mf07", "contamination": 0.05, "n_estimators": 300, "max_features": 0.7, "random_state": 42},
    {"name": "v1_c010_mf07", "contamination": 0.10, "n_estimators": 300, "max_features": 0.7, "random_state": 42},
]

# LOF sweep config (same contamination values for apples-to-apples comparison)
LOF_CONFIGS = [
    {"name": "lof_c001", "contamination": 0.01, "n_neighbors": 35},
    {"name": "lof_c005", "contamination": 0.05, "n_neighbors": 35},
    {"name": "lof_c010", "contamination": 0.10, "n_neighbors": 35},
]

TARGET_ANOMALY_RATE = 0.05
TARGET_RATE_BAND = (0.03, 0.08)

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

    # Read the user features (large parquet = can sit here with no output for a while)
    print(f"Reading user features: {input_path}")
    user_df = pd.read_parquet(input_path)
    print(f"Loaded {len(user_df):,} rows — validating columns…")
    validate_inputs(user_df)

    # Prepare the model input
    print("Preparing feature matrix (numeric + impute)…")
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

    # Set up MLflow (URI must match where mlflow server listens, if you use HTTP)
    print("Connecting to MLflow…")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI_DEFAULT))
    mlflow.set_experiment(DEFAULT_EXPERIMENT)
    run_summaries = []
    result_paths = {}
    model_input_example = model_input.head(5)

    # Run the anomaly detection for each configuration
    print(f"Training + scoring {len(CONFIGS)} configs (each can take several minutes)…")
    for cfg in CONFIGS:
        run_name = f"isolation_forest_{cfg['name']}"
        run_output_path = output_path.with_name(
            f"{output_path.stem}_{cfg['name']}{output_path.suffix}"
        )
        # Start an MLflow run for this configuration
        print(f"[{cfg['name']}] Starting MLflow run (fit + score)…")
        with mlflow.start_run(run_name=run_name):
            # Log the configuration parameters
            mlflow.log_param("config_name", cfg["name"])
            mlflow.log_param("input_path", str(input_path))
            mlflow.log_param("output_path", str(run_output_path))
            mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
            mlflow.log_param("n_estimators", cfg["n_estimators"])
            mlflow.log_param("max_features", cfg["max_features"])
            mlflow.log_param("contamination", cfg["contamination"])
            mlflow.log_param("random_state", cfg["random_state"])
            mlflow.log_param("n_rows", len(model_input))
            mlflow.log_param("python_version", sys.version.split()[0])
            # Log the versions of the dependencies for reproducibility
            mlflow.log_param("numpy_version", np.__version__)
            mlflow.log_param("pandas_version", pd.__version__)
            mlflow.log_param("sklearn_version", sklearn.__version__)
            mlflow.log_param("mlflow_version", mlflow.__version__)
            
            # Train the model
            model = IsolationForest(
                n_estimators=cfg["n_estimators"],
                contamination=cfg["contamination"],
                max_features=cfg["max_features"],
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
            result_paths[cfg["name"]] = run_output_path

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
            mlflow.sklearn.log_model(model, name="model", input_example=model_input_example)
            mlflow.log_artifact(str(run_output_path))

            # Log the summary of the run
            run_summaries.append(
                {
                    "model_type": "isolation_forest",
                    "config_name": cfg["name"],
                    "n_estimators": cfg["n_estimators"],
                    "max_features": cfg["max_features"],
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

    print(f"Training + scoring {len(LOF_CONFIGS)} LOF configs…")
    for cfg in LOF_CONFIGS:
        run_name = f"local_outlier_factor_{cfg['name']}"
        run_output_path = output_path.with_name(
            f"{output_path.stem}_{cfg['name']}{output_path.suffix}"
        )

        print(f"[{cfg['name']}] Starting MLflow run (fit + score)…")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_type", "local_outlier_factor")
            mlflow.log_param("config_name", cfg["name"])
            mlflow.log_param("input_path", str(input_path))
            mlflow.log_param("output_path", str(run_output_path))
            mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
            mlflow.log_param("n_neighbors", cfg["n_neighbors"])
            mlflow.log_param("contamination", cfg["contamination"])
            mlflow.log_param("n_rows", len(model_input))
            mlflow.log_param("python_version", sys.version.split()[0])
            mlflow.log_param("numpy_version", np.__version__)
            mlflow.log_param("pandas_version", pd.__version__)
            mlflow.log_param("sklearn_version", sklearn.__version__)
            mlflow.log_param("mlflow_version", mlflow.__version__)

            lof = LocalOutlierFactor(
                n_neighbors=cfg["n_neighbors"],
                contamination=cfg["contamination"],
                n_jobs=-1,
            )
            pred = lof.fit_predict(model_input)  # -1 anomaly, 1 normal
            is_anomaly = (pred == -1).astype(int)
            # negative_outlier_factor_: lower => more abnormal; negate so higher => more anomalous
            anomaly_score = -lof.negative_outlier_factor_

            result_df = user_df[["user_id", *FEATURE_COLUMNS]].copy()
            result_df["anomaly_score"] = anomaly_score
            result_df["is_anomaly"] = is_anomaly
            result_df["anomaly_rank"] = result_df["anomaly_score"].rank(
                method="first", ascending=False
            ).astype(int)
            result_df = result_df.sort_values("anomaly_score", ascending=False)

            print(f"[{cfg['name']}] Writing anomaly scores: {run_output_path}")
            result_df.to_parquet(run_output_path, index=False)
            result_paths[cfg["name"]] = run_output_path

            n_anomalies = int(result_df["is_anomaly"].sum())
            anomaly_rate = float(result_df["is_anomaly"].mean())
            score_mean = float(result_df["anomaly_score"].mean())
            score_std = float(result_df["anomaly_score"].std(ddof=0))
            score_max = float(result_df["anomaly_score"].max())
            score_min = float(result_df["anomaly_score"].min())

            mlflow.log_metric("n_users_scored", int(len(result_df)))
            mlflow.log_metric("n_anomalies", n_anomalies)
            mlflow.log_metric("anomaly_rate", anomaly_rate)
            mlflow.log_metric("anomaly_score_mean", score_mean)
            mlflow.log_metric("anomaly_score_std", score_std)
            mlflow.log_metric("anomaly_score_max", score_max)
            mlflow.log_metric("anomaly_score_min", score_min)
            mlflow.log_artifact(str(run_output_path))

            run_summaries.append(
                {
                    "model_type": "local_outlier_factor",
                    "config_name": cfg["name"],
                    "n_neighbors": cfg["n_neighbors"],
                    "contamination": cfg["contamination"],
                    "n_users_scored": int(len(result_df)),
                    "n_anomalies": n_anomalies,
                    "anomaly_rate": anomaly_rate,
                    "anomaly_score_mean": score_mean,
                    "anomaly_score_std": score_std,
                }
            )

            print(
                f"[{cfg['name']}] Run complete | "
                f"users={len(result_df)} | anomalies={n_anomalies}"
            )

    # Save the summary of all runs
    summary_df = pd.DataFrame(run_summaries).sort_values("config_name")
    summary_df.to_parquet(MODEL_SELECTION_OUTPUT, index=False)
    print(f"Saved model comparison table: {MODEL_SELECTION_OUTPUT}")

    # Recommend a "winner" run based on practical anomaly-rate range.
    # If nothing lands in the target band, choose the one closest to target rate.
    low, high = TARGET_RATE_BAND
    in_band = summary_df[
        (summary_df["anomaly_rate"] >= low) & (summary_df["anomaly_rate"] <= high)
    ].copy()

    candidate_df = in_band if not in_band.empty else summary_df.copy()
    candidate_df["distance_to_target"] = (
        candidate_df["anomaly_rate"] - TARGET_ANOMALY_RATE
    ).abs()
    winner = candidate_df.sort_values("distance_to_target").iloc[0]

    print("\nRecommended default config:")
    print(
        f"model={winner['model_type']} | config={winner['config_name']} | "
        f"anomaly_rate={winner['anomaly_rate']:.4f} | "
        f"n_anomalies={int(winner['n_anomalies'])}"
    )
    if in_band.empty:
        print(
            f"(No run in target band {low:.2f}-{high:.2f}; picked closest to "
            f"target rate {TARGET_ANOMALY_RATE:.2f}.)"
        )

    # Save winner output for notebook use
    winner_config_name = winner["config_name"]
    winner_path = result_paths.get(winner_config_name)
    if winner_path is not None:
        winner_df = pd.read_parquet(winner_path)
        winner_df.to_parquet(WINNER_OUTPUT, index=False)
        # Keep canonical output untouched during sweep experiments.
        # winner_df.to_parquet(output_path, index=False)
        print(f"Saved winner output: {WINNER_OUTPUT}")
        # print(f"Updated canonical output: {output_path}")


if __name__ == "__main__":
    main()
