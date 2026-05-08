#!/usr/bin/env python3
"""
Rule-based suspicious user scoring baseline.

Reads:
  - data/processed/user_features.parquet

Writes:
  - outputs/rule_based_scores.parquet

This script creates a simple suspiciousness score based on:
  1. extreme rating behavior
  2. review burst / high review rate
  3. low friend count with enough review activity
"""

import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, round as spark_round, when


# Setup project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "user_features.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "rule_based_scores.parquet"

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")


# Rating behavior thresholds
LOW_AVG_RATING_THRESHOLD = 1.5
HIGH_AVG_RATING_THRESHOLD = 4.5
HIGH_SINGLE_RATING_PERCENT = 0.90

# Review burst thresholds
# User must have at least 20 reviews and average at least 0.02 reviews/day.
# 0.02 reviews/day is about 1 review every 50 days.
MIN_REVIEWS_FOR_BURST = 20
MIN_REVIEWS_PER_DAY = 0.02

# Low social activity thresholds
# User must have 0 or 1 friends AND at least 10 reviews.
# This avoids flagging casual users who simply do not use Yelp socially.
MAX_FRIENDS_FOR_LOW_SOCIAL = 1
MIN_REVIEWS_FOR_LOW_SOCIAL = 10

spark = (
    SparkSession.builder.appName("RuleBasedUserScore")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Reading user features from {INPUT_PATH}")
users = spark.read.parquet(str(INPUT_PATH))

# Fill missing values so the rules do not break if some user metadata is missing
users = users.fillna(
    {
        "avg_stars_given": 0.0,
        "pct_5_star_reviews": 0.0,
        "pct_1_star_reviews": 0.0,
        "review_count": 0,
        "reviews_per_day": 0.0,
        "account_age_days": 0,
        "num_friends": 0,
    }
)

scored = (
    users

    # Flag 1: Extreme rating behavior
    # Flags users who mostly give very low ratings, very high ratings, or almost always give only 1-star or only 5-star reviews.
    .withColumn(
        "extreme_rating_flag",
        when(
            (col("avg_stars_given") <= LOW_AVG_RATING_THRESHOLD)
            | (col("avg_stars_given") >= HIGH_AVG_RATING_THRESHOLD)
            | (col("pct_5_star_reviews") >= HIGH_SINGLE_RATING_PERCENT)
            | (col("pct_1_star_reviews") >= HIGH_SINGLE_RATING_PERCENT),
            lit(1),
        ).otherwise(lit(0)),
    )

    # Flag 2: Review burst / high review rate
    # Flags users who have written many reviews at a high rate; approximates bursty reviewing behavior using reviews_per_day.
    .withColumn(
        "review_burst_flag",
        when(
            (
                (col("review_count") >= MIN_REVIEWS_FOR_BURST)
                & (col("reviews_per_day") >= MIN_REVIEWS_PER_DAY)
            )
            | (
                (col("review_count") >= 50)
                & (col("num_friends") <= MAX_FRIENDS_FOR_LOW_SOCIAL)
            ),
            lit(1),
        ).otherwise(lit(0)),
    )

    # Flag 3: Low social activity
    # Flags users with very few friends, but only if they also wrote enough reviews.
    .withColumn(
        "low_friends_flag",
        when(
            (col("num_friends") <= MAX_FRIENDS_FOR_LOW_SOCIAL)
            & (col("review_count") >= MIN_REVIEWS_FOR_LOW_SOCIAL),
            lit(1),
        ).otherwise(lit(0)),
    )

    # Counts how many suspicious rules were triggered.
    .withColumn(
        "flag_count",
        col("extreme_rating_flag")
        + col("review_burst_flag")
        + col("low_friends_flag"),
    )

    # Combines the three rule flags into one final score from 0 to 1.
    .withColumn(
        "rule_based_score",
        spark_round(col("flag_count") / lit(3), 4),
    )

    # Marks a user as suspicious only if they have low social activity and at least one other suspicious behavior.
    .withColumn(
        "is_flagged_suspicious",
        when(
            (col("low_friends_flag") == 1)
            & (
                (col("extreme_rating_flag") == 1)
                | (col("review_burst_flag") == 1)
            ),
            lit(1),
        ).otherwise(lit(0)),
    )
)

output = scored.select(
    "user_id",
    "avg_stars_given",
    "pct_5_star_reviews",
    "pct_1_star_reviews",
    "review_count",
    "reviews_per_day",
    "account_age_days",
    "num_friends",
    "extreme_rating_flag",
    "review_burst_flag",
    "low_friends_flag",
    "flag_count",
    "rule_based_score",
    "is_flagged_suspicious",
)

print("\nTop Rule-Based Suspicious Users:")
output.orderBy(
    col("is_flagged_suspicious").desc(),
    col("rule_based_score").desc(),
    col("review_count").desc(),
).show(
    20,
    truncate=False,
)

print("\nScore Distribution:")
output.groupBy("rule_based_score").count().orderBy("rule_based_score").show()

print("\nFlagged Suspicious User Count:")
output.groupBy("is_flagged_suspicious").count().show()

print("\nIndividual Flag Counts:")
output.groupBy("extreme_rating_flag").count().show()
output.groupBy("review_burst_flag").count().show()
output.groupBy("low_friends_flag").count().show()

print("\nFlag Combination Counts:")
output.groupBy(
    "extreme_rating_flag",
    "review_burst_flag",
    "low_friends_flag",
).count().orderBy(col("count").desc()).show()

print(f"\nWriting rule-based scores to {OUTPUT_PATH}")
output.write.mode("overwrite").parquet(str(OUTPUT_PATH))

print(f"Saved successfully to {OUTPUT_PATH}")

spark.stop()