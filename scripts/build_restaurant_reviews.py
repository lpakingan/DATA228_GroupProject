#!/usr/bin/env python3
"""
Filter reviews_base down to just restaurant reviews.

Reads:
  - data/processed/restaurants_base.parquet
  - data/processed/reviews_base.parquet
Writes: data/processed/restaurant_reviews.parquet
"""

import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

spark = (
    SparkSession.builder.appName("BuildRestaurantReviews")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

restaurants_path = PROCESSED_DIR / "restaurants_base.parquet"
reviews_path = PROCESSED_DIR / "reviews_base.parquet"
output_path = PROCESSED_DIR / "restaurant_reviews.parquet"

print(f"Reading restaurants from {restaurants_path}")
restaurants = spark.read.parquet(str(restaurants_path)).select("business_id")

print(f"Reading reviews from {reviews_path}")
reviews = spark.read.parquet(str(reviews_path))

print("\nBroadcast joining reviews to restaurants")
restaurant_reviews = reviews.join(broadcast(restaurants), on="business_id", how="inner")

# partitioning by review year
print(f"\nWriting {output_path}")
restaurant_reviews.write.mode("overwrite").partitionBy("review_year").parquet(str(output_path))
print(f"Saved to {output_path}")

spark.stop()
