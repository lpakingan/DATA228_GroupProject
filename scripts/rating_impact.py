#!/usr/bin/env python3
"""
Person 4 — Rating Impact Analysis (MAIN RESULT)

For each (business_id, spike_date) event, compute average star rating:
  - BEFORE window
  - DURING window
  - AFTER window

Inputs:
  - data/processed/restaurant_reviews.parquet  (business_id, date, stars)
  - outputs/suspicious_businesses.parquet      (business_id, spike_date, ...)
Output:
  - outputs/rating_impact.parquet

Default windows (configurable):
  - BEFORE: [spike_date - before_days, spike_date - 1]
  - DURING: [spike_date - during_days, spike_date + during_days]
           (during_days defaults to 0 = same day)
  - AFTER:  [spike_date + 1, spike_date + after_days]
"""

import argparse
import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    to_date,
    date_add,
    avg as f_avg,
    count as f_count,
    lit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute rating before/during/after suspicious spike events.")
    p.add_argument("--restaurant_reviews", default=str(PROCESSED_DIR / "restaurant_reviews.parquet"))
    p.add_argument("--suspicious_businesses", default=str(OUTPUTS_DIR / "suspicious_businesses.parquet"))
    p.add_argument("--output", default=str(OUTPUTS_DIR / "rating_impact.parquet"))
    p.add_argument("--before_days", type=int, default=30)
    p.add_argument("--during_days", type=int, default=0, help="0 = spike day only; 1 = ±1 day; etc.")
    p.add_argument("--after_days", type=int, default=30)
    p.add_argument("--driver_memory", default="6g")
    p.add_argument("--shuffle_partitions", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.before_days < 1:
        raise ValueError("--before_days must be >= 1")
    if args.after_days < 1:
        raise ValueError("--after_days must be >= 1")
    if args.during_days < 0:
        raise ValueError("--during_days must be >= 0")

    spark = (
        SparkSession.builder.appName("RatingImpactAnalysis")
        .master("local[*]")
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    rr_path = Path(args.restaurant_reviews)
    sb_path = Path(args.suspicious_businesses)
    out_path = Path(args.output)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading restaurant reviews: {rr_path}")
    rr = (
        spark.read.parquet(str(rr_path))
        .select(
            col("business_id"),
            to_date(col("date")).alias("review_date"),
            col("stars").cast("double").alias("stars"),
        )
        .filter(col("review_date").isNotNull() & col("stars").isNotNull())
    )

    print(f"Reading suspicious spike events: {sb_path}")
    events = spark.read.parquet(str(sb_path)).select(
        "business_id",
        col("spike_date").alias("spike_date"),
        "review_count",
        "baseline_mean",
        "baseline_std",
        "history_days",
        "abs_increase",
        "z_score",
        "total_reviews_in_window",
        "suspicious_reviews_in_window",
        "suspicious_user_count",
        "max_anomaly_score",
        "avg_anomaly_score",
        "window_days",
    )

    # BEFORE window aggregates
    before_join = (
        (rr["business_id"] == events["business_id"])
        & (rr["review_date"] >= date_add(events["spike_date"], -int(args.before_days)))
        & (rr["review_date"] <= date_add(events["spike_date"], -1))
    )
    before = rr.join(events, on=before_join, how="inner").groupBy(
        events["business_id"], events["spike_date"]
    ).agg(
        f_avg(rr["stars"]).alias("avg_stars_before"),
        f_count(lit(1)).alias("n_reviews_before"),
    )

    # DURING window aggregates
    during_join = (
        (rr["business_id"] == events["business_id"])
        & (rr["review_date"] >= date_add(events["spike_date"], -int(args.during_days)))
        & (rr["review_date"] <= date_add(events["spike_date"], int(args.during_days)))
    )
    during = rr.join(events, on=during_join, how="inner").groupBy(
        events["business_id"], events["spike_date"]
    ).agg(
        f_avg(rr["stars"]).alias("avg_stars_during"),
        f_count(lit(1)).alias("n_reviews_during"),
    )

    # AFTER window aggregates
    after_join = (
        (rr["business_id"] == events["business_id"])
        & (rr["review_date"] >= date_add(events["spike_date"], 1))
        & (rr["review_date"] <= date_add(events["spike_date"], int(args.after_days)))
    )
    after = rr.join(events, on=after_join, how="inner").groupBy(
        events["business_id"], events["spike_date"]
    ).agg(
        f_avg(rr["stars"]).alias("avg_stars_after"),
        f_count(lit(1)).alias("n_reviews_after"),
    )

    out = (
        events
        .join(before, on=["business_id", "spike_date"], how="left")
        .join(during, on=["business_id", "spike_date"], how="left")
        .join(after, on=["business_id", "spike_date"], how="left")
        .withColumn("before_days", lit(int(args.before_days)))
        .withColumn("during_days", lit(int(args.during_days)))
        .withColumn("after_days", lit(int(args.after_days)))
    )

    print(f"Writing rating impact table: {out_path}")
    out.write.mode("overwrite").parquet(str(out_path))
    print(f"Saved to {out_path}")

    spark.stop()


if __name__ == "__main__":
    main()

