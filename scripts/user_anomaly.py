#!/usr/bin/env python3
"""
Default user anomaly pipeline: Isolation Forest (5% contamination, 300 trees, max_features=1.0).

Reads:
  data/processed/user_features.parquet
Writes:
  outputs/user_anomaly_scores.parquet

Logs:
  MLflow experiment user_anomaly_detection — params, metrics, model artifact.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "user_features.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "user_anomaly_scores.parquet"

# What to look for on MLflow
DEFAULT_EXPERIMENT = "user_anomaly_detection"
TRACKING_URI_DEFAULT = "http://localhost:5001"

# Isolation Forest parameters
CONTAMINATION = 0.05
N_ESTIMATORS = 300
MAX_FEATURES = 1.0
RANDOM_STATE = 42
CONFIG_NAME = "if_c005_est300_mf10"

FEATURE_COLUMNS = [
    "avg_stars_given",
    "pct_5_star_reviews",
    "pct_1_star_reviews",
    "review_count",
    "reviews_per_day",
    "account_age_days",
    "num_friends",
]

# Validate the inputs
def validate_inputs(df: pd.DataFrame) -> None:
    missing = [c for c in ["user_id", *FEATURE_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

# Prepare the features by imputing the median
def prepare_features(user_df: pd.DataFrame) -> pd.DataFrame:
    model_input = user_df[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        model_input[col] = pd.to_numeric(model_input[col], errors="coerce")
        model_input[col] = model_input[col].replace([np.inf, -np.inf], np.nan)
        med = model_input[col].median()
        model_input[col] = model_input[col].fillna(0 if pd.isna(med) else med)
    return model_input

# Make the Isolation Forest model
def make_isolation_forest() -> IsolationForest:
    return IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def main() -> None:
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Read the user features
    print(f"Reading user features: {DEFAULT_INPUT}")
    user_df = pd.read_parquet(DEFAULT_INPUT)
    print(f"Loaded {len(user_df):,} rows — validating…")
    validate_inputs(user_df)

    # Drop the 6 rows where critical metrics are missing
    user_df = user_df.dropna(subset=["account_age_days", "num_friends"])

    # Prepare the feature matrix
    print("Preparing feature matrix (median imputation -> Isolation Forest)…")
    model_input = prepare_features(user_df)
    model_input_example = model_input.head(5)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI_DEFAULT))
    mlflow.set_experiment(DEFAULT_EXPERIMENT)

    # Set the run name
    run_name = f"user_anomaly_{CONFIG_NAME}"
    print(
        f"MLflow run {run_name} | IF contamination={CONTAMINATION}, "
        f"n_estimators={N_ESTIMATORS}, max_features={MAX_FEATURES}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", "isolation_forest")
        mlflow.log_param("config_name", CONFIG_NAME)
        mlflow.log_param("contamination", CONTAMINATION)
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_features", MAX_FEATURES)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("input_path", str(DEFAULT_INPUT))
        mlflow.log_param("output_path", str(DEFAULT_OUTPUT))
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
        mlflow.log_param("n_rows", len(model_input))
        mlflow.log_param("python_version", sys.version.split()[0])
        mlflow.log_param("numpy_version", np.__version__)
        mlflow.log_param("pandas_version", pd.__version__)
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("mlflow_version", mlflow.__version__)

        model = make_isolation_forest()
        model.fit(model_input)
        pred = model.predict(model_input)
        is_anomaly = (pred == -1).astype(int)
        # Calculate the anomaly score - decision_function: higher => more normal
        # then negate so higher => more anomalous
        anomaly_score = -model.decision_function(model_input)

        # Create the result dataframe
        result_df = user_df[["user_id", *FEATURE_COLUMNS]].copy()
        # Add the anomaly score and is_anomaly columns
        result_df["anomaly_score"] = anomaly_score
        # Add the is_anomaly column
        result_df["is_anomaly"] = is_anomaly
        # Add the anomaly rank column
        result_df["anomaly_rank"] = (
            result_df["anomaly_score"].rank(method="first", ascending=False).astype(int)
        )
        result_df = result_df.sort_values("anomaly_score", ascending=False)

        result_df.to_parquet(DEFAULT_OUTPUT, index=False)

        # Log the metrics
        n_anomalies = int(result_df["is_anomaly"].sum())
        # Calculate the anomaly rate
        anomaly_rate = float(result_df["is_anomaly"].mean())
        # Log the number of users scored
        mlflow.log_metric("n_users_scored", int(len(result_df)))
        # Log the number of anomalies
        mlflow.log_metric("n_anomalies", n_anomalies)
        # Log the anomaly rate
        mlflow.log_metric("anomaly_rate", anomaly_rate)
        # Log the mean, std, max, and min of the anomaly scores
        mlflow.log_metric("anomaly_score_mean", float(result_df["anomaly_score"].mean()))
        mlflow.log_metric("anomaly_score_std", float(result_df["anomaly_score"].std(ddof=0)))
        mlflow.log_metric("anomaly_score_max", float(result_df["anomaly_score"].max()))
        mlflow.log_metric("anomaly_score_min", float(result_df["anomaly_score"].min()))

        # Log the model
        mlflow.sklearn.log_model(
            model, artifact_path="model", input_example=model_input_example
        )
        mlflow.log_artifact(str(DEFAULT_OUTPUT))

    # Print the results 
    print(f"Done | anomalies={n_anomalies} | rate={anomaly_rate:.4f}")
    # Print the output path
    print(f"Wrote {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
