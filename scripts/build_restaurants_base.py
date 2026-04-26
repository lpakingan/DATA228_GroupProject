#!/usr/bin/env python3
"""
Filter the businesses table down to just restaurants.

Reads:  data/parquet/yelp_academic_dataset_business
Writes: data/processed/restaurants_base.parquet
"""

import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

spark = (
    SparkSession.builder.appName("BuildRestaurantsBase")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

input_path = PARQUET_DIR / "yelp_academic_dataset_business"
output_path = PROCESSED_DIR / "restaurants_base.parquet"

print(f"Reading {input_path}")
df = spark.read.parquet(str(input_path))
df.printSchema()

# filter null business_ids and keep only restaurants
df = (
    df
    .filter(col("business_id").isNotNull())
    .filter(col("categories").isNotNull() & lower(col("categories")).contains("restaurants"))
)

print(f"\nWriting {output_path}")
df.write.mode("overwrite").parquet(str(output_path))
print(f"Saved to {output_path}")

spark.stop()
