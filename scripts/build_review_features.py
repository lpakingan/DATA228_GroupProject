#!/usr/bin/env python3
"""
Build per-review feature table for restaurant reviews.

Inputs:
  - data/processed/restaurant_reviews.parquet
  - data/processed/users_base.parquet
Output:
  - data/processed/review_features.parquet

Features:
  - review_length: character count of review text
  - review_hour: hour-of-day (0-23) from review timestamp
  - day_of_week: 1-7 (Spark: Monday=1 ... Sunday=7)
  - days_since_signup: days between yelping_since and review date
  - user_review_sequence: 1..N order of user's reviews (by date, then review_id)
"""

import argparse
import os
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col,
    length,
    hour,
    dayofweek,
    datediff,
    to_date,
    row_number,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-review features for restaurant reviews.")
    p.add_argument("--restaurant_reviews", default=str(PROCESSED_DIR / "restaurant_reviews.parquet"))
    p.add_argument("--users_base", default=str(PROCESSED_DIR / "users_base.parquet"))
    p.add_argument("--output", default=str(PROCESSED_DIR / "review_features.parquet"))
    p.add_argument("--driver_memory", default="6g")
    p.add_argument("--shuffle_partitions", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    spark = (
        SparkSession.builder.appName("BuildReviewFeatures")
        .master("local[*]")
        .config("spark.driver.memory", args.driver_memory)
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    restaurant_reviews_path = Path(args.restaurant_reviews)
    users_base_path = Path(args.users_base)
    output_path = Path(args.output)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading restaurant reviews from {restaurant_reviews_path}")
    rr = spark.read.parquet(str(restaurant_reviews_path))

    print(f"Reading users base from {users_base_path}")
    users = spark.read.parquet(str(users_base_path)).select("user_id", "yelping_since")

    # Join yelping_since onto each review (left join to keep all reviews)
    joined = rr.join(users, on="user_id", how="left")

    # Deterministic per-user ordering for sequence
    seq_w = Window.partitionBy("user_id").orderBy(col("date").asc_nulls_last(), col("review_id").asc_nulls_last())

    feats = (
        joined
        .withColumn("review_length", length(col("text")))
        .withColumn("review_hour", hour(col("date")))
        .withColumn("day_of_week", dayofweek(col("date")))
        .withColumn("days_since_signup", datediff(to_date(col("date")), to_date(col("yelping_since"))))
        .withColumn("user_review_sequence", row_number().over(seq_w))
        .select(
            "review_id",
            "business_id",
            "user_id",
            "date",
            "review_year",
            "stars",
            "review_length",
            "review_hour",
            "day_of_week",
            "days_since_signup",
            "user_review_sequence",
        )
    )

    print(f"Writing {output_path}")
    feats.write.mode("overwrite").partitionBy("review_year").parquet(str(output_path))
    print(f"Saved to {output_path}")

    spark.stop()


if __name__ == "__main__":
    main()

