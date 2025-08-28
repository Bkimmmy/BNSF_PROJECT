# Create src/etl_pipeline.py based on the user's notebook flow and earlier cells
import os, textwrap, json, pathlib

code = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETL Pipeline for Predictive Maintenance (MetroPT / IoT compressors)
-------------------------------------------------------------------
This script converts raw IoT logs into an engineered, model-ready Parquet with:
  - Failure labeling (strict: COMP==0 & MPG==0 & LPS==0)
  - Time binning (default: 2-minute bins)
  - Aggregations for numeric and binary signals
  - Rolling statistics (e.g., 30-min window on 2-min bins)
  - Lag and delta features
  - Next/last failure timestamps
  - Remaining Useful Life (RUL) in minutes

Designed to run on Databricks or any Spark environment with access to S3.

Usage (examples)
----------------
Databricks notebook cell or CLI:

  python src/etl_pipeline.py \
    --input s3://hqpsusu-ml-data-bucket/raw/iot/metropt.csv \
    --output s3://hqpsusu-ml-data-bucket/processed/df_final \
    --bin-minutes 2 \
    --roll-minutes 30

Outputs
-------
- Parquet dataset at --output (partitioned by nothing; ordered by timestamp_bin)
- Logged schema & row/column counts

Notes
-----
- Columns are selected dynamically if present in input.
- If COMP/MPG/LPS are missing, failure flag will be all zeros (warns).
"""

import argparse
from typing import List

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ---------------------------
# Helpers
# ---------------------------
def keep_existing(df, cols: List[str]) -> List[str]:
    """Return only columns that exist in df."""
    have = set(df.columns)
    return [c for c in cols if c in have]


def build_spark(app_name: str = "rail-etl") -> SparkSession:
    """Create a SparkSession. On Databricks, this simply returns the current session."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        # .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")  # if running outside Databricks
        .getOrCreate()
    )
    return spark


# ---------------------------
# Core ETL
# ---------------------------
def run_etl(input_path: str, output_path: str, bin_minutes: int = 2, roll_minutes: int = 30) -> None:
    spark = build_spark()

    print(f"🟦 Reading raw CSV: {input_path}")
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # -- Ensure timestamp as proper type
    if "timestamp" in df.columns:
        df = df.withColumn("timestamp", F.to_timestamp("timestamp"))
    else:
        raise ValueError("Input data must include a 'timestamp' column.")

    # -- Strict failure flag (if digital columns exist)
    has_comp = "COMP" in df.columns
    has_mpg  = "MPG" in df.columns
    has_lps  = "LPS" in df.columns

    if has_comp and has_mpg and has_lps:
        df = df.withColumn(
            "failure_raw",
            F.when((F.col("COMP") == 0) & (F.col("MPG") == 0) & (F.col("LPS") == 0), 1).otherwise(0)
        )
    else:
        print("⚠️  Missing one of COMP/MPG/LPS — setting failure_raw = 0 for all rows.")
        df = df.withColumn("failure_raw", F.lit(0))

    # -- Optional: drop rows without GPS if present
    if "gpsLat" in df.columns and "gpsLong" in df.columns:
        df = df.dropna(subset=["gpsLat", "gpsLong"])

    # -- Time bin (floor to N-minute buckets)
    sec = bin_minutes * 60
    df = df.withColumn("timestamp_bin", (F.col("timestamp").cast("long") / sec).cast("long") * sec)
    df = df.withColumn("timestamp_bin", F.from_unixtime("timestamp_bin").cast("timestamp"))

    # -----------------------------
    # Aggregations per time bin
    # -----------------------------
    numeric_cols_all = [
        "Motor_current", "Oil_temperature", "DV_pressure",
        "TP2", "TP3", "H1", "Reservoirs", "Flowmeter",
        "Caudal_impulses", "gpsSpeed"
    ]
    binary_cols_all = [
        "COMP", "MPG", "LPS", "Pressure_switch",
        "DV_eletric", "Towers", "Oil_level", "gpsQuality"
    ]

    numeric_cols = keep_existing(df, numeric_cols_all)
    binary_cols  = keep_existing(df, binary_cols_all)

    if not numeric_cols:
        raise ValueError("No numeric sensor columns found. Check your input schema.")

    agg_exprs = []
    for c in numeric_cols:
        agg_exprs += [
            F.avg(c).alias(f"{c}_avg"),
            F.stddev(c).alias(f"{c}_std"),
            F.min(c).alias(f"{c}_min"),
            F.max(c).alias(f"{c}_max"),
        ]
    for c in binary_cols:
        # cast to int in case they're boolean/string-like
        agg_exprs += [F.max(F.col(c).cast("int")).alias(f"{c}_flag")]

    # Carry failure to the bin
    agg_exprs += [F.max("failure_raw").alias("failure")]

    features = (
        df.groupBy("timestamp_bin")
          .agg(*agg_exprs)
          .orderBy("timestamp_bin")
    )

    # -----------------------------
    # Rolling stats / lag / deltas
    # -----------------------------
    w_order = Window.orderBy("timestamp_bin")
    # number of rows that corresponds to the rolling window (exclude current row)
    rows_back = max(int(roll_minutes / max(bin_minutes, 1)) - 1, 1)
    w_roll = w_order.rowsBetween(-rows_back, -1)

    key_bases_all = [
        "Motor_current_avg",
        "Oil_temperature_avg",
        "DV_pressure_avg",
    ]
    key_bases = keep_existing(features, key_bases_all)

    for base in key_bases:
        features = features.withColumn(f"{base}_roll_mean_{roll_minutes}m", F.avg(F.col(base)).over(w_roll))
        features = features.withColumn(f"{base}_roll_std_{roll_minutes}m",  F.stddev(F.col(base)).over(w_roll))
        features = features.withColumn(f"{base}_lag1", F.lag(F.col(base), 1).over(w_order))
        features = features.withColumn(f"{base}_delta", F.col(base) - F.col(f"{base}_lag1"))

    # -----------------------------
    # Next/last failure + RUL
    # -----------------------------
    w_fwd  = Window.orderBy("timestamp_bin").rowsBetween(0, Window.unboundedFollowing)
    w_back = Window.orderBy("timestamp_bin").rowsBetween(Window.unboundedPreceding, 0)

    features = features.withColumn(
        "next_failure_time",
        F.first(F.when(F.col("failure") == 1, F.col("timestamp_bin")), ignorenulls=True).over(w_fwd)
    )
    features = features.withColumn(
        "last_failure_time",
        F.last(F.when(F.col("failure") == 1, F.col("timestamp_bin")), ignorenulls=True).over(w_back)
    )
    features = features.withColumn(
        "RUL_minutes",
        F.when(F.col("failure") == 1, F.lit(0.0))
         .otherwise((F.unix_timestamp("next_failure_time") - F.unix_timestamp("timestamp_bin")) / 60.0)
    )
    features = features.withColumn(
        "minutes_since_last_failure",
        (F.unix_timestamp("timestamp_bin") - F.unix_timestamp("last_failure_time")) / 60.0
    )

    # Fill NA for early bins on rolling stats
    fill_zero_cols = [c for c in features.columns if c.endswith("_roll_mean_%dm" % roll_minutes) or
                      c.endswith("_roll_std_%dm" % roll_minutes) or
                      c.endswith("_lag1") or
                      c.endswith("_delta")]
    features = features.fillna(0, subset=fill_zero_cols)

    # -----------------------------
    # Finalize
    # -----------------------------
    df_final = features

    print("🟦 Output schema:")
    df_final.printSchema()

    print("🟦 Sample rows:")
    df_final.show(10, truncate=False)

    print("🟦 Row count:", df_final.count())
    print("🟦 Column count:", len(df_final.columns))

    print(f"🟦 Writing Parquet to: {output_path}")
    (df_final
        .repartition(1)  # small demo; adjust/remove for large data
        .write.mode("overwrite").parquet(output_path)
    )

    print("✅ ETL complete.")


def parse_args():
    p = argparse.ArgumentParser(description="ETL pipeline for rail predictive maintenance")
    p.add_argument("--input",  type=str,
                   default="s3://hqpsusu-ml-data-bucket/raw/iot/metropt.csv",
                   help="Input CSV (S3 or local path)")
    p.add_argument("--output", type=str,
                   default="s3://hqpsusu-ml-data-bucket/processed/df_final",
                   help="Output Parquet path (S3 or local)")
    p.add_argument("--bin-minutes", type=int, default=2, help="Bin size in minutes")
    p.add_argument("--roll-minutes", type=int, default=30, help="Rolling window span in minutes")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_etl(args.input, args.output, bin_minutes=args.bin_minutes, roll_minutes=args.roll_minutes)
'''

# ensure folder exists
os.makedirs("/mnt/data/src", exist_ok=True)
with open("/mnt/data/src/etl_pipeline.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Wrote file to /mnt/data/src/etl_pipeline.py")
