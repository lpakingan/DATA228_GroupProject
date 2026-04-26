#!/usr/bin/env python3
"""
Yelp JSON -> Parquet conversion
"""

import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, year

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
WAREHOUSE_DIR = PROJECT_ROOT / "spark-warehouse"

DATASETS = ["yelp_academic_dataset_business", "yelp_academic_dataset_user", "yelp_academic_dataset_checkin", "yelp_academic_dataset_tip", "yelp_academic_dataset_review"]

os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

spark = (
    SparkSession.builder.appName("YelpJsonToParquet")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.warehouse.dir", str(WAREHOUSE_DIR))
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

for name in DATASETS:
    input_path = RAW_DIR / f"{name}.json"
    output_path = PARQUET_DIR / name

    if not input_path.exists():
        print(f"Skipping {name}: {input_path} not found")
        continue

    print(f"\nReading {name}.json")
    df = spark.read.json(str(input_path))
    df.printSchema()

    # partition reviews by year so easier to query
    if name.endswith("review") and "date" in df.columns:
        df = df.withColumn("date", to_timestamp("date")).withColumn("review_year", year("date"))
        df.write.mode("overwrite").partitionBy("review_year").parquet(str(output_path))
    else:
        df.write.mode("overwrite").parquet(str(output_path))

    print(f"Saved to {output_path}")

spark.stop()