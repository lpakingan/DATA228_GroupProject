#!/usr/bin/env python3
"""
Person 4 — Suspicious Business Detection

Identify businesses that have review-volume spikes AND suspicious (anomalous) users active
around those spike days.

Inputs:
  - outputs/review_spikes.parquet
  - outputs/user_anomaly_scores.parquet
  - data/processed/restaurant_reviews.parquet   (for user<->business linking on dates)

Output:
  - outputs/suspicious_businesses.parquet

Method (simple + explainable):
  1) Filter to spike business-days (is_spike=1)
  2) Filter to suspicious users (is_anomaly=1)
  3) Join spike business-days to restaurant reviews in a configurable time window
     (e.g., same day, +/- 1 day, +/- 3 days), then keep only reviews by suspicious users
  4) Aggregate per business & spike_date:
       - total reviews in window
       - suspicious reviews in window
       - unique suspicious users in window
       - max/avg anomaly_score of suspicious users involved
"""

import argparse
import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    to_date,
    count as f_count,
    countDistinct,
    max as f_max,
    avg as f_avg,
    date_add,
    lit,
    broadcast,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect suspicious businesses using spikes + anomalous users.")
    p.add_argument("--review_spikes", default=str(OUTPUTS_DIR / "review_spikes.parquet"))
    p.add_argument(
        "--user_anomaly_scores",
        default="",
        help="Path to outputs/user_anomaly_scores.parquet (from anomaly detection step).",
    )
    p.add_argument("--restaurant_reviews", default=str(PROCESSED_DIR / "restaurant_reviews.parquet"))
    p.add_argument("--output", default=str(OUTPUTS_DIR / "suspicious_businesses.parquet"))
    p.add_argument(
        "--window_days",
        type=int,
        default=0,
        help="Join window around spike day: 0=same day, 1=±1 day, 3=±3 days, etc.",
    )
    p.add_argument("--driver_memory", default="6g")
    p.add_argument("--shuffle_partitions", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.user_anomaly_scores:
        raise SystemExit(
            "Missing --user_anomaly_scores. Example:\n"
            "  --user_anomaly_scores /path/to/outputs/user_anomaly_scores.parquet"
        )

    spark = (
        SparkSession.builder.appName("BusinessSuspiciousActivity")
        .master("local[*]")
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    review_spikes_path = Path(args.review_spikes)
    user_anomaly_path = Path(args.user_anomaly_scores)
    restaurant_reviews_path = Path(args.restaurant_reviews)
    output_path = Path(args.output)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading spikes: {review_spikes_path}")
    spikes = spark.read.parquet(str(review_spikes_path))
    spikes = spikes.filter(col("is_spike") == 1).select(
        "business_id",
        col("review_date").alias("spike_date"),
        "review_count",
        "baseline_mean",
        "baseline_std",
        "history_days",
        "abs_increase",
        "z_score",
    )

    print(f"Reading anomaly scores: {user_anomaly_path}")
    users = spark.read.parquet(str(user_anomaly_path))
    required = {"user_id", "is_anomaly", "anomaly_score"}
    missing = sorted(required - set(users.columns))
    if missing:
        raise RuntimeError(f"user_anomaly_scores missing columns {missing}. Found: {users.columns}")

    suspicious_users = (
        users.filter(col("is_anomaly") == 1)
        .select("user_id", col("anomaly_score").cast("double").alias("anomaly_score"))
    )

    print(f"Reading restaurant reviews: {restaurant_reviews_path}")
    rr = spark.read.parquet(str(restaurant_reviews_path)).select(
        "review_id",
        "business_id",
        "user_id",
        to_date(col("date")).alias("review_date"),
    ).filter(col("review_date").isNotNull())

    # Join spikes to reviews in a date window
    wd = int(args.window_days)
    if wd < 0:
        raise ValueError("--window_days must be >= 0")

    # Keep total review volume around spike day (all users)
    join_cond = (
        (rr["business_id"] == spikes["business_id"])
        & (rr["review_date"] >= date_add(spikes["spike_date"], -wd))
        & (rr["review_date"] <= date_add(spikes["spike_date"], wd))
    )

    spike_window_reviews = rr.join(spikes, on=join_cond, how="inner").select(
        spikes["business_id"],
        spikes["spike_date"],
        rr["review_id"],
        rr["user_id"],
        rr["review_date"],
        spikes["review_count"],
        spikes["baseline_mean"],
        spikes["baseline_std"],
        spikes["history_days"],
        spikes["abs_increase"],
        spikes["z_score"],
    )

    total_counts = spike_window_reviews.groupBy("business_id", "spike_date").agg(
        f_count(lit(1)).alias("total_reviews_in_window")
    )

    # Suspicious subset (broadcast: suspicious users ~5% of users)
    suspicious_reviews = spike_window_reviews.join(
        broadcast(suspicious_users),
        on="user_id",
        how="inner",
    )

    suspicious_aggs = suspicious_reviews.groupBy("business_id", "spike_date").agg(
        f_count(lit(1)).alias("suspicious_reviews_in_window"),
        countDistinct("user_id").alias("suspicious_user_count"),
        f_max("anomaly_score").alias("max_anomaly_score"),
        f_avg("anomaly_score").alias("avg_anomaly_score"),
    )

    # Combine with spike stats
    spike_stats = spikes.select(
        "business_id",
        "spike_date",
        "review_count",
        "baseline_mean",
        "baseline_std",
        "history_days",
        "abs_increase",
        "z_score",
    )

    out = (
        spike_stats
        .join(total_counts, on=["business_id", "spike_date"], how="left")
        .join(suspicious_aggs, on=["business_id", "spike_date"], how="left")
        .fillna(
            {
                "total_reviews_in_window": 0,
                "suspicious_reviews_in_window": 0,
                "suspicious_user_count": 0,
            }
        )
        .withColumn("window_days", lit(wd))
    )

    print(f"Writing suspicious businesses: {output_path}")
    out.write.mode("overwrite").parquet(str(output_path))
    print(f"Saved to {output_path}")

    spark.stop()


if __name__ == "__main__":
    main()

