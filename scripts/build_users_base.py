#!/usr/bin/env python3
"""
Clean the raw users table.

Reads:  data/parquet/yelp_academic_dataset_user
Writes: data/processed/users_base.parquet
"""

import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

spark = (
    SparkSession.builder.appName("BuildUsersBase")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

input_path = PARQUET_DIR / "yelp_academic_dataset_user"
output_path = PROCESSED_DIR / "users_base.parquet"

print(f"Reading {input_path}")
df = spark.read.parquet(str(input_path))
df.printSchema()

# filter null user_ids and cast yelping_since to a timestamp
df = (
    df
    .filter(col("user_id").isNotNull())
    .withColumn("yelping_since", to_timestamp("yelping_since"))
)

print(f"\nWriting {output_path}")
df.write.mode("overwrite").parquet(str(output_path))
print(f"Saved to {output_path}")

spark.stop()
