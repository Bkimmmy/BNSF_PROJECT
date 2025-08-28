# We'll generate a production-ready `src/train_models.py` script based on the user's notebooks.
# The script will:
# - load a prepared features dataset (CSV or Parquet; local or s3)
# - build the feature matrix (drop leakage cols, add optional time-of-day sin/cos)
# - do a time-ordered split
# - train RandomForest and XGBoost
# - evaluate with MAE, RMSE, R2, sMAPE, tolerance metrics
# - save models, metrics, and plots locally
# - optionally upload artifacts to S3 if --s3-bucket/--s3-prefix provided
#
# We'll place it under /mnt/data/src/train_models.py so the user can download it.


import os, textwrap, json, pathlib

os.makedirs("/mnt/data/src", exist_ok=True)

code = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_models.py — Production-style training script (RF & XGBoost) for RUL regression.

Usage (local artifacts):
    python src/train_models.py \
        --input s3://hqpsusu-ml-data-bucket/processed/df_final.parquet \
        --target RUL_minutes \
        --timestamp-col timestamp_bin \
        --drop-cols failure next_failure_time last_failure_time minutes_since_last_failure RUL_minutes timestamp_bin \
        --output-dir ./outputs \
        --add-time-of-day

Upload artifacts to S3 as well:
    python src/train_models.py ... \
        --s3-bucket hqpsusu-ml-data-bucket \
        --s3-prefix final_project/models

Notes:
- Expects a "prepared" features table similar to your df_final with a numeric target column (RUL_minutes).
- Will time-order split (no shuffle) if a timestamp column is provided.
- Saves: model pickles, metrics.json, feature importances, and several PNG charts.
"""

from __future__ import annotations

import os
import io
import re
import sys
import json
import math
import argparse
import pathlib
import warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

try:
    import xgboost as xgb
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False

try:
    import joblib
except Exception as e:
    raise SystemExit("joblib is required to save models. Please `pip install joblib`.") from e

# Optional S3
_S3_AVAILABLE = True
try:
    import boto3
except Exception:
    _S3_AVAILABLE = False

# ------------------------
# Utils
# ------------------------

def is_s3_path(p: str) -> bool:
    return isinstance(p, str) and p.lower().startswith("s3://")

def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load parquet or csv from local or S3.
    Requires `pyarrow` for parquet and `s3fs` if using S3 with pandas.
    """
    if path.lower().endswith(".parquet") or path.lower().endswith(".pq"):
        return pd.read_parquet(path)  # needs pyarrow or fastparquet
    elif path.lower().endswith(".csv"):
        return pd.read_csv(path)
    else:
        # try parquet by default
        return pd.read_parquet(path)

def add_time_of_day_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    if ts_col not in df.columns:
        warnings.warn(f"--add-time-of-day requested but timestamp col '{ts_col}' not found. Skipping.")
        return df
    if not np.issubdtype(df[ts_col].dtype, np.datetime64):
        # try to parse
        try:
            df = df.copy()
            df[ts_col] = pd.to_datetime(df[ts_col])
        except Exception:
            warnings.warn(f"Could not convert '{ts_col}' to datetime; skipping time-of-day features.")
            return df
    minutes = df[ts_col].dt.hour * 60 + df[ts_col].dt.minute
    df = df.copy()
    df["tod_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
    df["tod_cos"] = np.cos(2 * np.pi * minutes / 1440.0)
    return df

def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: List[str],
    timestamp_col: Optional[str],
    add_time_of_day: bool
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Return X, y, feat_names from a prepared df."""
    if add_time_of_day and timestamp_col:
        df = add_time_of_day_features(df, timestamp_col)

    # target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in input data.")
    y = df[target_col].astype(float).to_numpy()

    # drop leakage / non-features
    to_drop = set([target_col] + (drop_cols or []))
    feat_cols = [c for c in df.columns if c not in to_drop and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        raise ValueError("No numeric feature columns found after dropping leakage columns.")

    X = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    return X, y, feat_cols

def time_ordered_split(
    df: pd.DataFrame,
    test_size: float,
    timestamp_col: Optional[str]
) -> np.ndarray:
    """Return indices for train/test split without shuffling. If timestamp provided, sort by it first."""
    if timestamp_col and timestamp_col in df.columns:
        try:
            order = np.argsort(pd.to_datetime(df[timestamp_col]).values)
        except Exception:
            order = np.arange(len(df))
    else:
        order = np.arange(len(df))
    n = len(order)
    cut = int(round(n * (1.0 - test_size)))
    idx_train = order[:cut]
    idx_test  = order[cut:]
    return idx_train, idx_test

def impute_with_median(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    meds = X_train.median(numeric_only=True)
    return X_train.fillna(meds), X_test.fillna(meds), meds

def metrics(y_true, y_pred) -> Dict[str, float]:
    eps = 1e-9
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    err = yp - yt
    ae = np.abs(err)
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(mean_squared_error(yt, yp, squared=False)),
        "R2": float(r2_score(yt, yp)),
        "Bias": float(err.mean()),
        "sMAPE_%": float(100.0 * np.mean(2*ae / (np.abs(yt)+np.abs(yp)+eps))),
        "pct_within_5min": float(100.0 * np.mean(ae <= 5)),
        "pct_within_10min": float(100.0 * np.mean(ae <= 10)),
        "N": int(yt.size),
    }

def plot_pred_vs_actual(y_true, y_rf, y_xgb, out_png: str):
    plt.figure(figsize=(12,5))
    # subsample if huge
    for y_pred, label in [(y_rf, "RF"), (y_xgb, "XGB")]:
        if y_pred is None: 
            continue
        n = len(y_true)
        idx = np.linspace(0, n-1, min(n, 1000)).astype(int)
        plt.scatter(y_true[idx], y_pred[idx], alpha=0.4, label=label, s=12, edgecolor="none")
    mn, mx = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([mn, mx], [mn, mx], "k--", lw=1)
    plt.xlabel("Actual RUL (min)"); plt.ylabel("Predicted RUL (min)")
    plt.title("Predicted vs Actual")
    plt.grid(alpha=0.3); plt.legend()
    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def plot_error_hist(y_true, y_rf, y_xgb, out_png: str):
    plt.figure(figsize=(10,5))
    for y_pred, label in [(y_rf, "RF"), (y_xgb, "XGB")]:
        if y_pred is None: 
            continue
        plt.hist(y_pred - y_true, bins=60, alpha=0.5, label=label)
        mu = float((y_pred - y_true).mean())
        plt.axvline(mu, ls="--", lw=1, label=f"{label} bias={mu:.1f}")
    plt.axvline(0, color="k", lw=1)
    plt.xlabel("Error (min)"); plt.ylabel("Count")
    plt.title("Prediction Error (Pred - Actual)")
    plt.grid(alpha=0.3); plt.legend()
    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def plot_acc_vs_tol(y_true, y_rf, y_xgb, out_png: str):
    plt.figure(figsize=(8,5))
    tols = np.arange(0, 61, 2)
    for y_pred, label in [(y_rf, "RF"), (y_xgb, "XGB")]:
        if y_pred is None: 
            continue
        acc = [100.0*np.mean(np.abs(y_pred - y_true) <= t) for t in tols]
        plt.plot(tols, acc, label=label)
    plt.xlabel("Tolerance (± minutes)"); plt.ylabel("Within tolerance (%)")
    plt.title("Accuracy vs Tolerance")
    plt.grid(alpha=0.3); plt.legend()
    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def save_feature_importances(model, feat_names: List[str], out_csv: str):
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False)
        pathlib.Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        imp.to_csv(out_csv, header=["importance"])

def upload_dir_to_s3(local_dir: str, bucket: str, prefix: str):
    if not _S3_AVAILABLE:
        print("boto3 not available; skipping S3 upload.")
        return
    s3 = boto3.client("s3")
    local_dir = pathlib.Path(local_dir)
    for p in local_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            key = f"{prefix.rstrip('/')}/{rel}"
            s3.upload_file(str(p), bucket, key)
            print(f"Uploaded s3://{bucket}/{key}")

# ------------------------
# Training
# ------------------------

def train_rf(X_tr, y_tr) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_tr, y_tr)
    return model

def train_xgb(X_tr, y_tr):
    if not _HAVE_XGB:
        warnings.warn("xgboost not installed; skipping XGB model.")
        return None
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_tr, y_tr)
    return model

# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser(description="Train RF & XGBoost for RUL regression.")
    ap.add_argument("--input", required=True, help="Path to features table (CSV/Parquet, local or s3://).")
    ap.add_argument("--target", default="RUL_minutes", help="Target column (default: RUL_minutes).")
    ap.add_argument("--timestamp-col", default=None, help="Optional timestamp column for time-ordered split.")
    ap.add_argument("--drop-cols", nargs="*", default=[], help="Columns to drop (leakage/non-features).")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test fraction (default 0.2).")
    ap.add_argument("--output-dir", default="outputs", help="Local directory to write artifacts.")
    ap.add_argument("--add-time-of-day", action="store_true", help="Add sin/cos time-of-day features from timestamp.")
    ap.add_argument("--s3-bucket", default=None, help="If set, upload artifacts to this S3 bucket.")
    ap.add_argument("--s3-prefix", default=None, help="S3 key prefix under bucket.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading data from: {args.input}")
    df = load_dataframe(args.input)

    # Build matrix
    X, y, feat_cols = build_feature_matrix(
        df=df,
        target_col=args.target,
        drop_cols=args.drop_cols,
        timestamp_col=args.timestamp_col,
        add_time_of_day=args.add_time_of_day
    )

    # Split (time-ordered if timestamp provided)
    idx_train, idx_test = time_ordered_split(df, args.test_size, args.timestamp_col)
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    # Impute
    X_train, X_test, train_meds = impute_with_median(X_train, X_test)
    print(f"Train shape: {X_train.shape}  Test shape: {X_test.shape}  Features: {len(feat_cols)}")

    # Train models
    rf = train_rf(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    xgb_model = train_xgb(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test) if xgb_model is not None else None

    # Metrics
    metrics_rf = metrics(y_test, y_pred_rf)
    metrics_xgb = metrics(y_test, y_pred_xgb) if y_pred_xgb is not None else None

    # Save models
    models_dir = os.path.join(args.output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    rf_path = os.path.join(models_dir, "rf_model.pkl")
    joblib.dump(rf, rf_path)
    print(f"Saved RF → {rf_path}")

    if xgb_model is not None:
        xgb_path = os.path.join(models_dir, "xgb_model.pkl")
        joblib.dump(xgb_model, xgb_path)
        print(f"Saved XGB → {xgb_path}")

    # Save feature importances
    save_feature_importances(rf, feat_cols, os.path.join(args.output_dir, "rf_feature_importances.csv"))
    if xgb_model is not None:
        save_feature_importances(xgb_model, feat_cols, os.path.join(args.output_dir, "xgb_feature_importances.csv"))

    # Save metrics.json
    m = {"RandomForest": metrics_rf}
    if metrics_xgb is not None:
        m["XGBoost"] = metrics_xgb
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    print("Saved metrics.json")

    # Plots
    plot_pred_vs_actual(y_test, y_pred_rf, y_pred_xgb, os.path.join(args.output_dir, "pred_vs_actual.png"))
    plot_error_hist(y_test, y_pred_rf, y_pred_xgb, os.path.join(args.output_dir, "error_hist.png"))
    plot_acc_vs_tol(y_test, y_pred_rf, y_pred_xgb, os.path.join(args.output_dir, "accuracy_vs_tolerance.png"))
    print("Saved plots.")

    # Save split info + medians for reproducibility
    np.save(os.path.join(args.output_dir, "idx_train.npy"), idx_train)
    np.save(os.path.join(args.output_dir, "idx_test.npy"), idx_test)
    pd.Series(train_meds, name="train_medians").to_csv(os.path.join(args.output_dir, "train_medians.csv"))
    pd.Series(feat_cols, name="feature_name").to_csv(os.path.join(args.output_dir, "feature_list.csv"), index=False)

    # Optional S3 upload
    if args.s3_bucket and args.s3_prefix:
        upload_dir_to_s3(args.output_dir, args.s3_bucket, args.s3_prefix)

    print("✅ Done.")

if __name__ == "__main__":
    main()
'''

with open("/mnt/data/src/train_models.py", "w") as f:
    f.write(code)

print("Wrote: /mnt/data/src/train_models.py")
