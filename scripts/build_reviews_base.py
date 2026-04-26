#!/usr/bin/env python3
"""
Clean the raw reviews table.

Reads:  data/parquet/yelp_academic_dataset_review
Writes: data/processed/reviews_base.parquet
"""

import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, trim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

spark = (
    SparkSession.builder.appName("BuildReviewsBase")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

input_path = PARQUET_DIR / "yelp_academic_dataset_review"
output_path = PROCESSED_DIR / "reviews_base.parquet"

print(f"Reading {input_path}")
df = spark.read.parquet(str(input_path))
df.printSchema()

# check to make sure user_id/business_id/text fields are not null
df = (
    df
    .withColumn("text", trim(col("text")))
    .filter(col("user_id").isNotNull())
    .filter(col("business_id").isNotNull())
    .filter(col("text").isNotNull() & (length(col("text")) > 0))
)

# partitioning by review year
print(f"\nWriting {output_path}")
df.write.mode("overwrite").partitionBy("review_year").parquet(str(output_path))
print(f"Saved to {output_path}")

spark.stop()