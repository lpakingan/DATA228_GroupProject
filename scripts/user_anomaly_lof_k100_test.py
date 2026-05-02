#!/usr/bin/env python3
"""
Quick LOF test: n_neighbors=100 only (same jitter + RobustScaler as user_anomaly_lof_final.py).

Does not overwrite outputs/user_anomaly_scores.parquet — writes a k100-specific parquet.
Computes LOF stability (mean pairwise Jaccard of top-K users across seeded subsample fits).
"""

from __future__ import annotations

import os
import sys
from itertools import combinations
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "user_features.parquet"
# Dedicated paths so this test run does not clobber your main sweep outputs
TEST_OUTPUT = PROJECT_ROOT / "outputs" / "user_anomaly_scores_lof_c001_k100.parquet"
STABILITY_OUTPUT = PROJECT_ROOT / "outputs" / "user_anomaly_lof_k100_stability.parquet"

MLFLOW_EXPERIMENT = "user_anomaly_lof_k100_test"
TRACKING_URI_DEFAULT = "http://localhost:5001"

# Fixed contamination for all runs (1% expected outliers)
CONTAMINATION = 0.01
N_NEIGHBORS = 100
# Fixed metric for all runs (euclidean distance)
METRIC = "euclidean"
CONFIG_NAME = "lof_c001_k100"

# Same as user_anomaly_lof_final.py
JITTER_STD = 1e-6
JITTER_RANDOM_STATE = 42

LOF_STABILITY_SEEDS = [42, 43, 44, 45, 46]
LOF_STABILITY_SAMPLE_FRAC = 0.8
TOP_K_FOR_STABILITY = 100

FEATURE_COLUMNS = [
    "avg_stars_given",
    "pct_5_star_reviews",
    "pct_1_star_reviews",
    "review_count",
    "reviews_per_day",
    "account_age_days",
    "num_friends",
]

# Validate the input data
def validate_inputs(df: pd.DataFrame) -> None:
    missing = [c for c in ["user_id", *FEATURE_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

# Prepare the features
def prepare_features(user_df: pd.DataFrame) -> pd.DataFrame:
    # Add jitter
    rng = np.random.default_rng(JITTER_RANDOM_STATE)
    model_input = user_df[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        model_input[col] = pd.to_numeric(model_input[col], errors="coerce")
        # Replace infinity with NaN
        model_input[col] = model_input[col].replace([np.inf, -np.inf], np.nan)
        med = model_input[col].median()
        # Fill the missing values with the median
        model_input[col] = model_input[col].fillna(0 if pd.isna(med) else med)
        noise = rng.normal(0, JITTER_STD, size=len(model_input))
        model_input[col] = model_input[col] + noise
    return model_input

# Make the LOF pipeline
def make_lof_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "lof",
                LocalOutlierFactor(
                    n_neighbors=N_NEIGHBORS,
                    contamination=CONTAMINATION,
                    metric=METRIC,
                    novelty=True,
                    n_jobs=-1,
                ),
            ),
        ]
    )

# Calculate the Jaccard similarity between two sets
def jaccard(a: set, b: set) -> float:
    u = a | b
    return 0.0 if not u else len(a & b) / len(u)

# Calculate the mean pairwise Jaccard similarity between a list of sets
def mean_pairwise_jaccard(sets: list[set]) -> float:
    pairs = list(combinations(sets, 2))
    if not pairs:
        return 0.0
    return float(np.mean([jaccard(x, y) for x, y in pairs]))

# Main function
def main() -> None:
    TEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Create the output directory if it doesn't exist
    print(f"Reading user features: {DEFAULT_INPUT}")
    user_df = pd.read_parquet(DEFAULT_INPUT)
    print(f"Loaded {len(user_df):,} rows — validating…")
    validate_inputs(user_df)

    # Prepare the features
    print("Preparing feature matrix (jitter + later RobustScaler in pipeline)…")
    model_input = prepare_features(user_df)
    model_input_example = model_input.head(5)

    # Set the tracking URI and experiment
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI_DEFAULT))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = f"lof_test_{CONFIG_NAME}"
    print(f"MLflow run {run_name} | n_neighbors={N_NEIGHBORS}, contamination={CONTAMINATION}")

    # Start the MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log the parameters
        mlflow.log_param("model_type", "local_outlier_factor")
        mlflow.log_param("config_name", CONFIG_NAME)
        mlflow.log_param("contamination", CONTAMINATION)
        mlflow.log_param("n_neighbors", N_NEIGHBORS)
        mlflow.log_param("metric", METRIC)
        mlflow.log_param("input_path", str(DEFAULT_INPUT))
        mlflow.log_param("output_path", str(TEST_OUTPUT))
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))
        mlflow.log_param("n_rows", len(model_input))
        mlflow.log_param("python_version", sys.version.split()[0])
        mlflow.log_param("numpy_version", np.__version__)
        mlflow.log_param("pandas_version", pd.__version__)
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("mlflow_version", mlflow.__version__)
        mlflow.log_param("feature_scaling", "RobustScaler")
        mlflow.log_param("jitter", "gaussian_prepare_features")
        mlflow.log_param("jitter_std", JITTER_STD)
        mlflow.log_param("jitter_random_state", JITTER_RANDOM_STATE)

        # Make the LOF pipeline
        pipe = make_lof_pipeline()
        # Fit the LOF pipeline on the model input
        pipe.fit(model_input)
        # Predict the anomalies
        pred = pipe.predict(model_input)
        # Create the result dataframe
        is_anomaly = (pred == -1).astype(int)
        anomaly_score = -pipe.score_samples(model_input)

        # Create the result dataframe
        result_df = user_df[["user_id", *FEATURE_COLUMNS]].copy()
        # Add the anomaly score and is_anomaly columns
        result_df["anomaly_score"] = anomaly_score
        result_df["is_anomaly"] = is_anomaly
        result_df["anomaly_rank"] = (
            result_df["anomaly_score"].rank(method="first", ascending=False).astype(int)
        )
        result_df = result_df.sort_values("anomaly_score", ascending=False)

        result_df.to_parquet(TEST_OUTPUT, index=False)
        # Log the metrics
        n_anomalies = int(result_df["is_anomaly"].sum())
        anomaly_rate = float(result_df["is_anomaly"].mean())
        mlflow.log_metric("n_users_scored", int(len(result_df)))
        mlflow.log_metric("n_anomalies", n_anomalies)
        mlflow.log_metric("anomaly_rate", anomaly_rate)
        mlflow.log_metric("anomaly_score_mean", float(result_df["anomaly_score"].mean()))
        mlflow.log_metric("anomaly_score_std", float(result_df["anomaly_score"].std(ddof=0)))
        mlflow.log_metric("anomaly_score_max", float(result_df["anomaly_score"].max()))
        mlflow.log_metric("anomaly_score_min", float(result_df["anomaly_score"].min()))

        # Stability: fit on seeded row subsamples, score table
        # Get the top k users by anomaly score
        top_k = min(TOP_K_FOR_STABILITY, len(user_df))
        sample_n = max(1000, int(len(model_input) * LOF_STABILITY_SAMPLE_FRAC))
        sample_n = min(sample_n, len(model_input))
        mlflow.log_param("stability_top_k", top_k)
        mlflow.log_param("stability_subsample_n", sample_n)
        mlflow.log_param("stability_subsample_frac", LOF_STABILITY_SAMPLE_FRAC)
        mlflow.log_param("stability_seeds", ",".join(str(s) for s in LOF_STABILITY_SEEDS))

        top_user_sets: list[set] = []
        for seed in LOF_STABILITY_SEEDS:
            sample_idx = (
                pd.Series(np.arange(len(model_input)))
                .sample(n=sample_n, random_state=seed, replace=False)
                .to_numpy()
            )
            sample_input = model_input.iloc[sample_idx]
            st_pipe = make_lof_pipeline()
            st_pipe.fit(sample_input)
            st_scores = -st_pipe.score_samples(model_input)
            st_df = pd.DataFrame({"user_id": user_df["user_id"], "anomaly_score": st_scores})
            top_ids = set(st_df.nlargest(top_k, "anomaly_score")["user_id"].tolist())
            top_user_sets.append(top_ids)

        stability_value = mean_pairwise_jaccard(top_user_sets)
        # Log the stability metric
        mlflow.log_metric("stability_mean_pairwise_jaccard", stability_value)

        # Save the stability to a parquet file
        pd.DataFrame(
            [
                {
                    "config_name": CONFIG_NAME,
                    "n_neighbors": N_NEIGHBORS,
                    "contamination": CONTAMINATION,
                    "top_k": int(top_k),
                    "stability_mean_pairwise_jaccard": stability_value,
                    "stability_seeds": ",".join(str(s) for s in LOF_STABILITY_SEEDS),
                    "stability_subsample_frac": LOF_STABILITY_SAMPLE_FRAC,
                    "stability_subsample_n": sample_n,
                }
            ]
        ).to_parquet(STABILITY_OUTPUT, index=False)

        # Log the model
        mlflow.sklearn.log_model(
            pipe, artifact_path="model", input_example=model_input_example
        )
        mlflow.log_artifact(str(TEST_OUTPUT))
        mlflow.log_artifact(str(STABILITY_OUTPUT))
    # Print the results
    print(f"Done | anomalies={n_anomalies} | rate={anomaly_rate:.4f}")
    print(
        f"Stability (top-{top_k} mean pairwise Jaccard, seeds {LOF_STABILITY_SEEDS}): "
        f"{stability_value:.4f}"
    )
    print(f"Wrote {TEST_OUTPUT}")
    print(f"Wrote {STABILITY_OUTPUT}")


if __name__ == "__main__":
    main()
