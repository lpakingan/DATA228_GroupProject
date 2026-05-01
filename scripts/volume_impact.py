#!/usr/bin/env python3
"""
Compares review volume before, during, and after suspicious spike dates.

Reads:
  - data/processed/restaurant_reviews.parquet
  - outputs/suspicious_businesses.parquet

Writes:
  - outputs/volume_impact.parquet
"""

import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, date_add, lit, round as spark_round, to_date, when


# Setup project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEWS_PATH = PROJECT_ROOT / "data" / "processed" / "restaurant_reviews.parquet"
SUSPICIOUS_BUSINESSES_PATH = PROJECT_ROOT / "outputs" / "suspicious_businesses.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "volume_impact.parquet"

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")


# Review volume comparison window
# This compares 30 days before and 30 days after each suspicious spike date.
WINDOW_DAYS = 30


spark = (
    SparkSession.builder.appName("VolumeImpact")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Reading restaurant reviews from {REVIEWS_PATH}")
reviews = (
    spark.read.parquet(str(REVIEWS_PATH))
    .select("business_id", "date")
    .filter(col("business_id").isNotNull())
    .filter(col("date").isNotNull())
    .withColumn("review_date", to_date(col("date")))
)

print(f"Reading suspicious businesses from {SUSPICIOUS_BUSINESSES_PATH}")
suspicious_businesses = (
    spark.read.parquet(str(SUSPICIOUS_BUSINESSES_PATH))
    .select(
        "business_id",
        "spike_date",
        col("review_count").alias("spike_day_review_count"),
        "baseline_mean",
        "abs_increase",
        "z_score",
        "suspicious_reviews_in_window",
        "suspicious_user_count",
    )
    .filter(col("business_id").isNotNull())
    .filter(col("spike_date").isNotNull())
    .dropDuplicates(["business_id", "spike_date"])
    .withColumn("spike_date", to_date(col("spike_date")))
)

joined = reviews.join(
    suspicious_businesses,
    on="business_id",
    how="inner",
)

windowed = joined.filter(
    (col("review_date") >= date_add(col("spike_date"), -WINDOW_DAYS))
    & (col("review_date") <= date_add(col("spike_date"), WINDOW_DAYS))
)

volume_impact = (
    windowed.groupBy(
        "business_id",
        "spike_date",
        "spike_day_review_count",
        "baseline_mean",
        "abs_increase",
        "z_score",
        "suspicious_reviews_in_window",
        "suspicious_user_count",
    )

    # Counts reviews in the 30 days before the suspicious spike date.
    .agg(
        count(
            when(col("review_date") < col("spike_date"), lit(1))
        ).alias("reviews_before_spike"),

        # Counts reviews on the suspicious spike date.
        count(
            when(col("review_date") == col("spike_date"), lit(1))
        ).alias("reviews_on_spike_date"),

        # Counts reviews in the 30 days after the suspicious spike date.
        count(
            when(col("review_date") > col("spike_date"), lit(1))
        ).alias("reviews_after_spike"),
    )

    # Difference between review volume after the spike and before the spike.
    .withColumn(
        "after_minus_before",
        col("reviews_after_spike") - col("reviews_before_spike"),
    )

    # Percent change from before-spike volume to after-spike volume.
    .withColumn(
        "after_before_pct_change",
        when(
            col("reviews_before_spike") > 0,
            spark_round(
                (
                    (col("reviews_after_spike") - col("reviews_before_spike"))
                    / col("reviews_before_spike")
                )
                * lit(100),
                2,
            ),
        ).otherwise(None),
    )

    # Total reviews counted in the before/on/after impact window.
    .withColumn(
        "total_reviews_in_impact_window",
        col("reviews_before_spike")
        + col("reviews_on_spike_date")
        + col("reviews_after_spike"),
    )

    # Share of reviews in this window that came from suspicious users.
    .withColumn(
        "suspicious_review_share",
        when(
            col("total_reviews_in_impact_window") > 0,
            spark_round(
                col("suspicious_reviews_in_window")
                / col("total_reviews_in_impact_window"),
                4,
            ),
        ).otherwise(lit(0.0)),
    )

    # Marks whether suspicious users were involved in the spike window.
    .withColumn(
        "has_suspicious_users",
        when(col("suspicious_user_count") > 0, lit(1)).otherwise(lit(0)),
    )

    # Labels whether review volume increased, decreased, or stayed the same after the spike.
    .withColumn(
        "volume_change_label",
        when(col("after_minus_before") > 0, lit("increased"))
        .when(col("after_minus_before") < 0, lit("decreased"))
        .otherwise(lit("no_change")),
    )

    # Marks larger post-spike increases as meaningful.
    .withColumn(
        "meaningful_increase_flag",
        when(
            (col("after_minus_before") >= 10)
            | (col("after_before_pct_change") >= 25),
            lit(1),
        ).otherwise(lit(0)),
    )

    # Adds the window size used for the comparison.
    .withColumn("window_days", lit(WINDOW_DAYS))
)

output = volume_impact.select(
    "business_id",
    "spike_date",
    "window_days",
    "spike_day_review_count",
    "baseline_mean",
    "abs_increase",
    "z_score",
    "suspicious_reviews_in_window",
    "suspicious_user_count",
    "total_reviews_in_impact_window",
    "suspicious_review_share",
    "has_suspicious_users",
    "reviews_before_spike",
    "reviews_on_spike_date",
    "reviews_after_spike",
    "after_minus_before",
    "after_before_pct_change",
    "volume_change_label",
    "meaningful_increase_flag",
)

print("\nReview Volume Impact Preview:")
output.orderBy(
    col("spike_day_review_count").desc(),
    col("suspicious_reviews_in_window").desc(),
).show(
    20,
    truncate=False,
)

print("\nVolume Impact Summary:")
output.select(
    "reviews_before_spike",
    "reviews_on_spike_date",
    "reviews_after_spike",
    "after_minus_before",
    "after_before_pct_change",
    "suspicious_review_share",
).summary().show(truncate=False)

print("\nVolume Change Label Counts:")
output.groupBy("volume_change_label").count().show()

print("\nMeaningful Increase Count:")
output.groupBy("meaningful_increase_flag").count().show()

print("\nSuspicious User Involvement Count:")
output.groupBy("has_suspicious_users").count().show()

print(f"\nWriting volume impact results to {OUTPUT_PATH}")
output.write.mode("overwrite").parquet(str(OUTPUT_PATH))

print(f"Saved successfully to {OUTPUT_PATH}")

spark.stop()