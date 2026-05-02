#!/usr/bin/env python3
"""
Final LOF-only pipeline: contamination 0.01, several configs as separate MLflow runs.

Reads:
  data/processed/user_features.parquet
Writes:
  outputs/user_anomaly_scores_<config>.parquet    (scores + stability_* columns; does not overwrite user_anomaly_scores.parquet)

MLflow: one run per LOF_CONFIGS row — tags sweep + lof_config for Compare runs;
params/metrics (including stability_mean_pairwise_jaccard per config) + pipeline + artifacts.
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
OUTPUT_DIR = PROJECT_ROOT / "outputs"
# Sweep rows become outputs/user_anomaly_scores_<config>.parquet — no reference to user_anomaly_scores.parquet
SWEEP_SCORES_PREFIX = "user_anomaly_scores"

# Separate experiment name so MLflow UI stays easy to filter for your final work
LOF_FINAL_EXPERIMENT = "user_anomaly_lof_final"
# Filter runs in the UI: tag sweep == this value, then select runs → Compare
MLFLOW_SWEEP_TAG = "lof_final_config_sweep"
TRACKING_URI_DEFAULT = "http://localhost:5001"

# Fixed contamination for all runs (1% expected outliers)
CONTAMINATION = 0.01

# Match-scale noise after impute (reproducible via JITTER_RANDOM_STATE)
JITTER_STD = 1e-6
JITTER_RANDOM_STATE = 42

# Several hyperparameter sets → several MLflow runs and making the names unique
# CANONICAL_CONFIG_NAME must match one entry’s "name".
LOF_CONFIGS: list[dict] = [
    {"name": "lof_c001_k15", "n_neighbors": 15, "metric": "euclidean"},
    {"name": "lof_c001_k25", "n_neighbors": 25, "metric": "euclidean"},
    {"name": "lof_c001_k35", "n_neighbors": 35, "metric": "euclidean"},
    {"name": "lof_c001_k50", "n_neighbors": 50, "metric": "euclidean"},
    {"name": "lof_c001_k35_l1", "n_neighbors": 35, "metric": "manhattan"},
]

# Temporary safe canonical config name
CANONICAL_CONFIG_NAME = "lof_c001_k35"

# LOF stability: seeded subsamples (LOF has no random_state on full fit)
# 5 seeds for stability calculation
LOF_STABILITY_SEEDS = [42, 43, 44, 45, 46]
LOF_STABILITY_SAMPLE_FRAC = 0.8
TOP_K_FOR_STABILITY = 100

# Features to use for the LOF model
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

# Jitter the features
def prepare_features(user_df: pd.DataFrame) -> pd.DataFrame:
    """Coerce, impute, tiny Gaussian jitter per column, then LOF pipeline scales."""
    rng = np.random.default_rng(JITTER_RANDOM_STATE)
    model_input = user_df[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        model_input[col] = pd.to_numeric(model_input[col], errors="coerce")
        model_input[col] = model_input[col].replace([np.inf, -np.inf], np.nan)
        med = model_input[col].median()
        model_input[col] = model_input[col].fillna(0 if pd.isna(med) else med)
        noise = rng.normal(0, JITTER_STD, size=len(model_input))
        model_input[col] = model_input[col] + noise
    return model_input

# Make the LOF pipeline
def make_lof_pipeline(cfg: dict) -> Pipeline:
    """RobustScaler (median/IQR) → LOF; input matrix already jittered in prepare_features."""
    return Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "lof",
                LocalOutlierFactor(
                    n_neighbors=cfg["n_neighbors"],
                    contamination=CONTAMINATION,
                    metric=cfg["metric"],
                    novelty=True,
                    n_jobs=-1,
                ),
            ),
        ]
    )

# Helper functions for stability calculation
def jaccard(a: set, b: set) -> float:
    u = a | b
    return 0.0 if not u else len(a & b) / len(u)

def mean_pairwise_jaccard(sets: list[set]) -> float:
    pairs = list(combinations(sets, 2))
    if not pairs:
        return 0.0
    return float(np.mean([jaccard(x, y) for x, y in pairs]))


def main() -> None:
    input_path = DEFAULT_INPUT
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading user features: {input_path}")
    user_df = pd.read_parquet(input_path)
    print(f"Loaded {len(user_df):,} rows — validating…")
    validate_inputs(user_df)

    print("Preparing feature matrix…")
    model_input = prepare_features(user_df)
    model_input_example = model_input.head(5)
    # Set the tracking URI and experiment
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI_DEFAULT))
    mlflow.set_experiment(LOF_FINAL_EXPERIMENT)

    # Track the output paths for each config
    result_paths: dict[str, Path] = {}

    # Same for every config (shared feature matrix)
    top_k = min(TOP_K_FOR_STABILITY, len(user_df))
    sample_n = max(1000, int(len(model_input) * LOF_STABILITY_SAMPLE_FRAC))
    sample_n = min(sample_n, len(model_input))

    # Footer print (canonical run only) — same names as user_anomaly_lof_k100_test.py
    stability_value = None
    print_n_anomalies = None
    print_anomaly_rate = None

    print(
        f"Running {len(LOF_CONFIGS)} LOF configs "
        f"(contamination={CONTAMINATION}) → separate MLflow runs…"
    )
    # Run the MLflow runs for each configuration
    for cfg in LOF_CONFIGS:
        run_name = f"lof_final_{cfg['name']}"
        run_output_path = OUTPUT_DIR / f"{SWEEP_SCORES_PREFIX}_{cfg['name']}.parquet"

        print(f"[{cfg['name']}] MLflow run (fit + score)…")
        # Log the configuration parameters
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "sweep": MLFLOW_SWEEP_TAG,
                    "lof_config": cfg["name"],
                }
            )
            # Log the configuration parameters
            mlflow.log_param("model_type", "local_outlier_factor")
            mlflow.log_param("config_name", cfg["name"])
            mlflow.log_param("contamination", CONTAMINATION)
            mlflow.log_param("n_neighbors", cfg["n_neighbors"])
            mlflow.log_param("metric", cfg["metric"])
            mlflow.log_param("input_path", str(input_path))
            mlflow.log_param("output_path", str(run_output_path))
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

            pipe = make_lof_pipeline(cfg)
            pipe.fit(model_input)
            pred = pipe.predict(model_input)
            # Predict the anomalies (1 = anomaly, 0 = not anomaly)
            is_anomaly = (pred == -1).astype(int)
            # Score the anomalies - higher scores = more anomalous
            anomaly_score = -pipe.score_samples(model_input)
            # Create the result dataframe
            result_df = user_df[["user_id", *FEATURE_COLUMNS]].copy()
            # Add the anomaly score and is_anomaly columns
            result_df["anomaly_score"] = anomaly_score
            result_df["is_anomaly"] = is_anomaly
            result_df["anomaly_rank"] = (
                result_df["anomaly_score"].rank(method="first", ascending=False).astype(int)
            )
            # Sort the result dataframe by anomaly score descending
            result_df = result_df.sort_values("anomaly_score", ascending=False)
            # Calculate the number of anomalies
            n_anomalies = int(result_df["is_anomaly"].sum())
            # Calculate the anomaly rate
            anomaly_rate = float(result_df["is_anomaly"].mean())
            # Calculate the mean, std, max, and min of the anomaly scores
            score_mean = float(result_df["anomaly_score"].mean())
            # Calculate the standard deviation - ddof=0 for population standard deviation
            score_std = float(result_df["anomaly_score"].std(ddof=0))
            # Calculate the maximum anomaly score  
            score_max = float(result_df["anomaly_score"].max())
            # Calculate the minimum anomaly score
            score_min = float(result_df["anomaly_score"].min())

            # Log the metrics to MLflow
            mlflow.log_metric("n_users_scored", int(len(result_df)))
            mlflow.log_metric("n_anomalies", n_anomalies)
            mlflow.log_metric("anomaly_rate", anomaly_rate)
            mlflow.log_metric("anomaly_score_mean", score_mean)
            mlflow.log_metric("anomaly_score_std", score_std)
            mlflow.log_metric("anomaly_score_max", score_max)
            mlflow.log_metric("anomaly_score_min", score_min)

            # Stability before log_model — same order as user_anomaly_lof_k100_test.py (every config)
            # Stability: fit on seeded row subsamples, score table
            mlflow.log_param("stability_top_k", top_k)
            mlflow.log_param("stability_subsample_n", sample_n)
            mlflow.log_param("stability_subsample_frac", LOF_STABILITY_SAMPLE_FRAC)
            mlflow.log_param(
                "stability_seeds",
                ",".join(str(s) for s in LOF_STABILITY_SEEDS),
            )
            # Calculate the mean pairwise Jaccard similarity for each seed
            top_user_sets: list[set] = []
            for seed in LOF_STABILITY_SEEDS:
                sample_idx = (
                    pd.Series(np.arange(len(model_input)))
                    .sample(n=sample_n, random_state=seed, replace=False)
                    .to_numpy()
                )
                # Get the sample input
                sample_input = model_input.iloc[sample_idx]
                # Make the LOF pipeline
                st_pipe = make_lof_pipeline(cfg)
                # Fit the LOF pipeline on the sample input
                st_pipe.fit(sample_input)
                st_scores = -st_pipe.score_samples(model_input)
                st_df = pd.DataFrame(
                    {"user_id": user_df["user_id"], "anomaly_score": st_scores}
                )
                # Get the top k users by anomaly score
                top_ids = set(st_df.nlargest(top_k, "anomaly_score")["user_id"].tolist())
                top_user_sets.append(top_ids)

            stability_run_value = mean_pairwise_jaccard(top_user_sets)
            mlflow.log_metric(
                "stability_mean_pairwise_jaccard", stability_run_value
            )

            # Same scalar stability metadata on every row (for downstream notebooks / joins)
            result_df["stability_mean_pairwise_jaccard"] = stability_run_value
            result_df["stability_top_k"] = top_k
            result_df["stability_subsample_n"] = sample_n
            result_df["stability_subsample_frac"] = LOF_STABILITY_SAMPLE_FRAC
            result_df["stability_seeds"] = ",".join(
                str(s) for s in LOF_STABILITY_SEEDS
            )

            result_df.to_parquet(run_output_path, index=False)
            result_paths[cfg["name"]] = run_output_path

            if cfg["name"] == CANONICAL_CONFIG_NAME:
                print_n_anomalies = n_anomalies
                print_anomaly_rate = anomaly_rate
                stability_value = stability_run_value

            # Log the model
            mlflow.sklearn.log_model(
                pipe, artifact_path="model", input_example=model_input_example
            )
            mlflow.log_artifact(str(run_output_path))

            print(
                f"[{cfg['name']}] done | users={len(result_df)} | anomalies={n_anomalies} "
                f"| rate={anomaly_rate:.4f}"
            )
    # Check if the canonical config is in the result paths
    if CANONICAL_CONFIG_NAME not in result_paths:
        raise ValueError(
            f"CANONICAL_CONFIG_NAME={CANONICAL_CONFIG_NAME!r} not in sweep results. "
            f"Available: {list(result_paths.keys())}"
        )
    canonical_path = result_paths[CANONICAL_CONFIG_NAME]
    print(f"Canonical config: {CANONICAL_CONFIG_NAME}")
    print(f"Canonical scores file: {canonical_path}")

    # Print the results (same layout as user_anomaly_lof_k100_test.py after the MLflow run)
    if (
        print_n_anomalies is not None
        and print_anomaly_rate is not None
        and top_k is not None
        and stability_value is not None
    ):
        print(
            f"Done | anomalies={print_n_anomalies} | rate={print_anomaly_rate:.4f}"
        )
        print(
            f"Stability (top-{top_k} mean pairwise Jaccard, seeds {LOF_STABILITY_SEEDS}): "
            f"{stability_value:.4f}"
        )
        print(
            "Canonical run includes stability_* columns in: "
            f"{OUTPUT_DIR / f'{SWEEP_SCORES_PREFIX}_{CANONICAL_CONFIG_NAME}.parquet'}"
        )


if __name__ == "__main__":
    main()
