#!/usr/bin/env python3
"""
Detect review spikes (sudden increases) per business over time.

Reads:  data/processed/restaurant_reviews.parquet
Writes: outputs/review_spikes.parquet

Method (simple + explainable):
  1) Aggregate to daily review counts per business
  2) For each business/day, compute a rolling baseline over the prior N days
  3) Flag a spike if today's count is much larger than the baseline

Default rule:
  - baseline = rolling mean of previous 30 days (excluding today)
  - spike if:
      count >= max(min_reviews, baseline_mean * ratio_threshold)
      AND count - baseline_mean >= abs_increase_threshold
      AND business has at least min_history_days in history
"""

import argparse
import os
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col,
    count as f_count,
    to_date,
    lit,
    avg as f_avg,
    stddev_samp,
    greatest,
    when,
    coalesce,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect spikes in review volume per business/day.")
    p.add_argument("--input", default=str(PROCESSED_DIR / "restaurant_reviews.parquet"))
    p.add_argument("--output", default=str(OUTPUTS_DIR / "review_spikes.parquet"))
    p.add_argument("--window_days", type=int, default=30, help="Rolling baseline window size (prior days).")
    p.add_argument("--min_history_days", type=int, default=30, help="Require at least this many prior days for baseline.")
    p.add_argument("--min_reviews", type=int, default=5, help="Minimum reviews/day to consider a spike.")
    p.add_argument("--ratio_threshold", type=float, default=3.0, help="Count must be >= baseline_mean * ratio.")
    p.add_argument(
        "--abs_increase_threshold",
        type=float,
        default=5.0,
        help="Count - baseline_mean must be >= this value.",
    )
    p.add_argument("--driver_memory", default="6g")
    p.add_argument("--shuffle_partitions", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    spark = (
        SparkSession.builder.appName("DetectReviewSpikes")
        .master("local[*]")
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    input_path = Path(args.input)
    output_path = Path(args.output)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading restaurant reviews from {input_path}")
    reviews = spark.read.parquet(str(input_path))

    required_cols = {"business_id", "date"}
    missing = sorted(required_cols - set(reviews.columns))
    if missing:
        raise RuntimeError(
            f"Input missing required columns {missing}. "
            f"Found columns: {sorted(reviews.columns)[:50]}"
        )

    # Daily counts per business
    daily = (
        reviews
        .select(col("business_id"), to_date(col("date")).alias("review_date"))
        .filter(col("review_date").isNotNull())
        .groupBy("business_id", "review_date")
        .agg(f_count(lit(1)).alias("review_count"))
    )

    # Rolling baseline over prior N days (exclude current day)
    w = (
        Window.partitionBy("business_id")
        .orderBy(col("review_date").cast("timestamp"))
        .rowsBetween(-args.window_days, -1)
    )

    with_baseline = (
        daily
        .withColumn("baseline_mean", f_avg(col("review_count")).over(w))
        .withColumn("baseline_std", stddev_samp(col("review_count")).over(w))
        .withColumn("history_days", col("baseline_mean").isNotNull().cast("int"))
    )

    # history_days above is just not-null; compute actual number of prior rows in window
    w_count = (
        Window.partitionBy("business_id")
        .orderBy(col("review_date").cast("timestamp"))
        .rowsBetween(-args.window_days, -1)
    )
    # Spark doesn't have count(*) over window with the same import name as agg count,
    # so just reuse f_count with a literal.
    with_baseline = with_baseline.withColumn("history_days", f_count(lit(1)).over(w_count))

    baseline_mean = coalesce(col("baseline_mean"), lit(0.0))
    spike_min = greatest(lit(float(args.min_reviews)), baseline_mean * lit(float(args.ratio_threshold)))

    spikes = (
        with_baseline
        .withColumn("abs_increase", col("review_count") - baseline_mean)
        .withColumn(
            "is_spike",
            when(
                (col("history_days") >= lit(int(args.min_history_days)))
                & (col("review_count") >= spike_min)
                & (col("abs_increase") >= lit(float(args.abs_increase_threshold))),
                lit(1),
            ).otherwise(lit(0)),
        )
        .withColumn(
            "z_score",
            when(col("baseline_std").isNull() | (col("baseline_std") == 0), lit(None)).otherwise(
                (col("review_count") - baseline_mean) / col("baseline_std")
            ),
        )
        .select(
            "business_id",
            "review_date",
            "review_count",
            "baseline_mean",
            "baseline_std",
            "history_days",
            "abs_increase",
            "z_score",
            "is_spike",
        )
    )

    print(f"Writing spikes table to {output_path}")
    spikes.write.mode("overwrite").parquet(str(output_path))
    print(f"Saved to {output_path}")

    spark.stop()


if __name__ == "__main__":
    main()

