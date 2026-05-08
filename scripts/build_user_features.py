#!/usr/bin/env python3
"""
Build user-level features from restaurant reviews and cleaned users data.

Reads:
  - data/processed/restaurant_reviews.parquet
  - data/processed/users_base.parquet
Writes: 
  - data/processed/user_features.parquet
"""

import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    count,
    current_date,
    datediff,
    lit,
    lower,
    size,
    split,
    sum as spark_sum,
    to_date,
    trim,
    when,
)

# Setup project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

# Initialize Spark Session
spark = (
    SparkSession.builder.appName("BuildUserFeatures")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# Ensure output directory exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Define file paths
reviews_path = PROCESSED_DIR / "restaurant_reviews.parquet"
users_path = PROCESSED_DIR / "users_base.parquet"
output_path = PROCESSED_DIR / "user_features.parquet"

# --- 1. Process Reviews ---
print(f"Reading restaurant reviews from {reviews_path}")
reviews = spark.read.parquet(str(reviews_path)) \
    .select("user_id", "stars") \
    .filter(col("user_id").isNotNull()) \
    .filter(col("stars").isNotNull())

# Aggregate behavior per user
review_stats = (
    reviews.groupBy("user_id")
    .agg(
        avg("stars").alias("avg_stars_given"),
        (spark_sum(when(col("stars") == 5, lit(1)).otherwise(lit(0))) / count(lit(1))).alias("pct_5_star_reviews"),
        (spark_sum(when(col("stars") == 1, lit(1)).otherwise(lit(0))) / count(lit(1))).alias("pct_1_star_reviews"),
        count(lit(1)).alias("review_count"),
    )
)

# --- 2. Process User Metadata ---
print(f"Reading users from {users_path}")
users = spark.read.parquet(str(users_path)).select("user_id", "yelping_since", "friends")

# Calculate account age
users_with_age = users.withColumn(
    "account_age_days",
    datediff(current_date(), to_date(col("yelping_since")))
)

# Calculate number of friends for either array-typed or string-typed columns.
friends_dtype = users.schema["friends"].dataType
# Check the type of the friends column
print(f"friends column type: {friends_dtype}")

if friends_dtype.typeName() == "array":
    # if the friends column is an array, count the number of friends
    num_friends_expr = when(col("friends").isNull(), lit(0)).otherwise(size(col("friends")))
else:
    # if the friends column is a string, count the number of friends
    num_friends_expr = when(
        col("friends").isNull() | (trim(col("friends")) == "") | (lower(trim(col("friends"))) == "none"),
        lit(0),
    ).otherwise(size(split(trim(col("friends")), r"\s*,\s*")))

users_with_friends = users_with_age.withColumn("num_friends", num_friends_expr)

# Join and Calculate Final Metrics
user_features = (
    review_stats.join(
        users_with_friends.select("user_id", "account_age_days", "num_friends"), 
        on="user_id", 
        how="left"
    )
    .withColumn(
        "reviews_per_day",
        when(col("account_age_days") > 0, col("review_count") / col("account_age_days")).otherwise(lit(0.0)),
    )
    .select(
        "user_id",
        "avg_stars_given",
        "pct_5_star_reviews",
        "pct_1_star_reviews",
        "review_count",
        "reviews_per_day",
        "account_age_days",
        "num_friends",
    )
)

# Preview results
print("\nFinal User Features Preview:")
user_features.show(10, truncate=False)

# Write output
print(f"\nWriting results to {output_path}")
user_features.write.mode("overwrite").parquet(str(output_path))
print(f"Saved successfully to {output_path}")

spark.stop()