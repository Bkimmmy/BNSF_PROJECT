#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline evaluation for trained models (RandomForest, XGBoost, optional LSTM).
- Rebuilds the SAME feature matrix & time-ordered split using saved artifacts.
- Loads models from S3 (or local), runs predictions, computes metrics,
  and saves plots + a metrics CSV under ./reports/.

Assumptions (defaults can be overridden via CLI flags):
- Artifacts saved by training exist at:
    s3://hqpsusu-ml-data-bucket/final_project/artifacts/
      ├─ feat_cols.json           # list[str] feature column names (order matters)
      ├─ train_medians.json       # dict[str -> float] for NaN/inf imputation
      ├─ cut.json                 # {"cut": <int>} index where test set begins
      ├─ seq_len.json             # {"seq_len": <int>}  (optional, for LSTM)
      ├─ seq_mu.npy, seq_std.npy  # (optional, for LSTM sequence normalization)
- Models saved by training exist at:
    s3://hqpsusu-ml-data-bucket/final_project/models/
      ├─ rf_model.pkl
      ├─ xgb_model.pkl
      └─ lstm.pt                 # (optional, PyTorch state_dict or scripted)
- Engineered dataset “df_final” is available as Parquet:
    s3://hqpsusu-ml-data-bucket/processed/df_final.parquet
"""

import argparse
import json
import os
import sys
import math
from io import BytesIO
from datetime import datetime
from typing import Dict, Tuple, List

import boto3
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------
# Small, safe helper utils
# -------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _is_s3(uri: str) -> bool:
    return uri.startswith("s3://")

def _split_s3(uri: str) -> Tuple[str, str]:
    # s3://bucket/key -> ("bucket", "key")
    assert uri.startswith("s3://")
    no_scheme = uri[len("s3://"):]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    key = "" if len(parts) == 1 else parts[1]
    return bucket, key

def _download_s3_to_bytes(uri: str) -> bytes:
    bucket, key = _split_s3(uri)
    buf = BytesIO()
    boto3.client("s3").download_fileobj(bucket, key, buf)
    buf.seek(0)
    return buf.getvalue()

def _download_s3_to_file(uri: str, local_path: str) -> str:
    bucket, key = _split_s3(uri)
    boto3.client("s3").download_file(bucket, key, local_path)
    return local_path

def _load_json(uri_or_path: str) -> dict:
    if _is_s3(uri_or_path):
        data = _download_s3_to_bytes(uri_or_path)
        return json.loads(data.decode("utf-8"))
    with open(uri_or_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_npy(uri_or_path: str) -> np.ndarray:
    if _is_s3(uri_or_path):
        data = _download_s3_to_bytes(uri_or_path)
        return np.load(BytesIO(data))
    return np.load(uri_or_path)

def _load_joblib(uri_or_path: str):
    if _is_s3(uri_or_path):
        data = _download_s3_to_bytes(uri_or_path)
        return joblib.load(BytesIO(data))
    return joblib.load(uri_or_path)

def _read_parquet(uri_or_path: str) -> pd.DataFrame:
    """
    Conservative reader: if S3, download to /tmp then read with pandas (no s3fs dependency).
    """
    if _is_s3(uri_or_path):
        tmp = f"/tmp/{os.path.basename(uri_or_path)}"
        _download_s3_to_file(uri_or_path, tmp)
        return pd.read_parquet(tmp)
    return pd.read_parquet(uri_or_path)

# -------------------------
# Feature & split rebuild
# -------------------------

DROP_COLS_DEFAULT = [
    "timestamp_bin",
    "failure",
    "next_failure_time",
    "last_failure_time",
    "minutes_since_last_failure",
    "RUL_minutes"
]

def build_feature_matrix(
    df_final: pd.DataFrame,
    feat_cols: List[str],
    use_time_of_day: bool = True
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Reconstruct X, y using the same engineering choices as training.
    - Adds cyclic time-of-day if `timestamp_bin` exists (same as training notebooks).
    - Filters numeric columns and then selects EXACT feat_cols order.
    - Target is RUL_minutes.
    """
    df = df_final.copy()

    if "RUL_minutes" not in df.columns:
        raise ValueError("df_final must contain 'RUL_minutes' as the regression target.")

    # time-of-day cyclic features (the training notebooks added these when timestamp_bin exists)
    if use_time_of_day and "timestamp_bin" in df.columns and np.issubdtype(df["timestamp_bin"].dtype, np.datetime64):
        tod_min = df["timestamp_bin"].dt.hour * 60 + df["timestamp_bin"].dt.minute
        df["tod_sin"] = np.sin(2 * np.pi * tod_min / 1440.0)
        df["tod_cos"] = np.cos(2 * np.pi * tod_min / 1440.0)

    # Keep only rows with defined RUL (as done in notebooks to avoid leakage)
    df = df[df["RUL_minutes"].notnull()]

    # Only take provided feature columns in the saved order
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns in df_final: {missing[:10]}...")

    X = df[feat_cols].copy()
    # Clean up weird values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    y = df["RUL_minutes"].astype(float).to_numpy()

    return X, y

def time_ordered_split(X: pd.DataFrame, y: np.ndarray, cut: int):
    """
    Same time-ordered split used during training. `cut` is the index where test begins.
    """
    if cut <= 0 or cut > len(y):
        raise ValueError(f"Invalid cut index: {cut} for length {len(y)}.")

    X_train = X.iloc[:cut].copy()
    X_test  = X.iloc[cut:].copy()
    y_train = y[:cut].copy()
    y_test  = y[cut:].copy()
    return X_train, X_test, y_train, y_test

def apply_medians(X_train: pd.DataFrame, X_test: pd.DataFrame, medians: Dict[str, float]):
    """
    Fill NaNs using TRAIN medians captured at training time (no leakage).
    Unknown columns gracefully ignored; missing medians default to overall train median.
    """
    train_meds = pd.Series(medians)
    # Any feature not present in medians → compute from current train (backward compatibility)
    for col in X_train.columns:
        if col not in train_meds.index:
            val = X_train[col].median(skipna=True)
            train_meds.loc[col] = val if math.isfinite(val) else 0.0

    X_train = X_train.fillna(train_meds)
    X_test  = X_test.fillna(train_meds)
    return X_train, X_test

# -------------------------
# Metrics & plots
# -------------------------

def metrics(y_true, y_pred) -> Dict[str, float]:
    eps = 1e-9
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    err = yp - yt
    ae = np.abs(err)

    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(mean_squared_error(yt, yp, squared=False)),
        "R2": float(r2_score(yt, yp)),
        "Bias": float(err.mean()),
        "sMAPE_pct": float(100.0 * np.mean(2*ae / (np.abs(yt)+np.abs(yp)+eps))),
        "Pct_within_5min": float(100.0 * np.mean(ae <= 5)),
        "Pct_within_10min": float(100.0 * np.mean(ae <= 10)),
        "N": int(yt.size)
    }

def save_plots(pairs: List[Tuple[str, np.ndarray, np.ndarray]], out_dir: str):
    _ensure_dir(out_dir)

    # Common axis for Pred vs Actual
    all_true = np.concatenate([yt for _, yt, _ in pairs])
    low, high = np.percentile(all_true, [1, 99])
    pad = 0.05 * (high - low + 1e-6)
    xy_min, xy_max = low - pad, high + pad

    # A) Per-model Pred vs Actual
    cols = len(pairs)
    fig, axes = plt.subplots(1, cols, figsize=(6*cols, 5))
    axes = np.atleast_1d(axes)
    for ax, (name, yt, yp) in zip(axes, pairs):
        n = len(yt)
        idx = np.linspace(0, n-1, min(n, 400)).astype(int)
        ax.scatter(yt[idx], yp[idx], alpha=0.5, s=18, edgecolor="none")
        ax.plot([xy_min, xy_max], [xy_min, xy_max], "k--", lw=1)
        ax.set_title(f"{name}: Pred vs Actual")
        ax.set_xlabel("Actual RUL (min)"); ax.set_ylabel("Predicted RUL (min)")
        ax.set_xlim(xy_min, xy_max); ax.set_ylim(xy_min, xy_max)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pred_vs_actual.png"), dpi=140)
    plt.close(fig)

    # B) Error histogram overlay
    fig = plt.figure(figsize=(9, 5))
    for (name, yt, yp) in pairs:
        err = yp - yt
        plt.hist(err, bins=70, alpha=0.45, label=name)
    plt.axvline(0, color="k", lw=1)
    plt.xlabel("Error (min)"); plt.ylabel("Count")
    plt.title("Prediction Error (Pred - Actual)")
    plt.grid(alpha=0.3); plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_hist.png"), dpi=140)
    plt.close(fig)

    # C) Accuracy vs Tolerance
    fig = plt.figure(figsize=(8,5))
    tols = np.arange(0, 61, 2)
    for (name, yt, yp) in pairs:
        acc = [float(np.mean(np.abs(yp - yt) <= t)*100.0) for t in tols]
        plt.plot(tols, acc, label=name)
    plt.xlabel("Tolerance (± minutes)"); plt.ylabel("Within-tolerance (%)")
    plt.title("Accuracy vs Tolerance (higher is better)")
    plt.grid(alpha=0.3); plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "acc_vs_tolerance.png"), dpi=140)
    plt.close(fig)

# -------------------------
# Optional LSTM loader
# -------------------------

def try_load_lstm_and_predict(
    lstm_path: str,
    X_test: pd.DataFrame,
    X_train: pd.DataFrame,
    seq_len_path: str = None,
    mu_path: str = None,
    std_path: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Best-effort LSTM inference (optional). Requires:
      - lstm_path (s3 or local)
      - seq_len (json), mu/std (npy) computed on TRAIN sequences during training
    Returns (y_true_seq_test, y_pred_seq_test) or raises on failure.
    """
    import torch
    # Load sequences meta
    if seq_len_path is None or mu_path is None or std_path is None:
        raise RuntimeError("Missing seq_len/mu/std artifacts for LSTM inference.")
    seq_len = _load_json(seq_len_path)["seq_len"]
    mu = _load_npy(mu_path)
    std = _load_npy(std_path)
    std[std == 0] = 1e-6

    # Rebuild full matrix in same order so test aligns with classic cut
    X_full = pd.concat([X_train, X_test], axis=0).to_numpy(dtype=np.float32)

    # Sliding windows with target at the window end
    Xs, ys, tgt = [], [], []
    # We do not have y_full here; but evaluation only needs to compare against
    # the classic y_test (same target). We will slice to Yseq_te using the same cut.
    # Build placeholder y_full = NaN; later we align to classic y_test length.
    y_full = np.full((len(X_full),), np.nan, dtype=np.float32)

    for i in range(len(X_full) - seq_len):
        Xs.append(X_full[i:i+seq_len])
        # placeholder for target; real y comes from classic split
        ys.append(np.nan)
        tgt.append(i + seq_len)
    Xs, tgt = np.asarray(Xs, np.float32), np.asarray(tgt)

    cut = len(X_train)  # index where classic test begins
    mask_test = tgt >= cut
    Xseq_te = Xs[mask_test]

    # Normalize with TRAIN-only stats captured at training time
    Xseq_te = (Xseq_te - mu) / std

    # Load torch model (either scripted or state_dict with a known class)
    bytes_or_path = lstm_path
    if _is_s3(lstm_path):
        tmp = "/tmp/lstm.pt"
        _download_s3_to_file(lstm_path, tmp)
        bytes_or_path = tmp

    try:
        # Try scripted/trace first
        model = torch.jit.load(bytes_or_path, map_location="cpu")
    except Exception:
        # Fallback: plain state_dict with a minimal LSTM head (must match training config)
        # This requires the training script to have saved a config JSON with n_features/hidden/layers.
        raise RuntimeError("Could not load LSTM (expected a TorchScript file).")

    model.eval()
    with torch.no_grad():
        y_pred_seq = model(torch.tensor(Xseq_te, dtype=torch.float32)).cpu().numpy().ravel()

    # y_true for LSTM test ⇒ classic y_test length; align by size
    return y_pred_seq  # caller will align against classic y_test by length

# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on the saved test split.")
    parser.add_argument("--data", default="s3://hqpsusu-ml-data-bucket/processed/df_final.parquet",
                        help="Path (s3 or local) to df_final.parquet with engineered features & RUL.")
    parser.add_argument("--artifacts", default="s3://hqpsusu-ml-data-bucket/final_project/artifacts/",
                        help="S3 prefix or local folder containing feat_cols.json, train_medians.json, cut.json, etc.")
    parser.add_argument("--models", default="s3://hqpsusu-ml-data-bucket/final_project/models/",
                        help="S3 prefix or local folder containing rf_model.pkl, xgb_model.pkl, optional lstm.pt")
    parser.add_argument("--out", default="reports",
                        help="Output folder for metrics CSV and plots.")
    parser.add_argument("--skip-lstm", action="store_true",
                        help="Skip LSTM evaluation even if artifacts exist.")
    args = parser.parse_args()

    _ensure_dir(args.out)
    figs_dir = os.path.join(args.out, "figures")
    _ensure_dir(figs_dir)

    # Resolve artifact paths
    def art(name):  # join artifacts prefix and name
        return args.artifacts.rstrip("/") + "/" + name
    def mdl(name):
        return args.models.rstrip("/") + "/" + name

    # 1) Load engineered dataset
    print(f"Loading df_final from: {args.data}")
    df_final = _read_parquet(args.data)

    # 2) Load artifacts
    print(f"Loading artifacts from: {args.artifacts}")
    feat_cols = _load_json(art("feat_cols.json"))
    medians   = _load_json(art("train_medians.json"))
    cut_json  = _load_json(art("cut.json"))
    cut = int(cut_json["cut"])

    # 3) Rebuild features & split
    X, y = build_feature_matrix(df_final, feat_cols=feat_cols, use_time_of_day=True)
    X_train, X_test, y_train, y_test = time_ordered_split(X, y, cut=cut)
    X_train, X_test = apply_medians(X_train, X_test, medians)

    # 4) Load models
    print(f"Loading models from: {args.models}")
    rf = _load_joblib(mdl("rf_model.pkl"))
    xgb = _load_joblib(mdl("xgb_model.pkl"))

    # 5) Predict & metrics
    results_rows = []
    pairs = []

    y_pred_rf = rf.predict(X_test)
    m_rf = metrics(y_test, y_pred_rf); m_rf["Model"] = "RandomForest"
    results_rows.append(m_rf)
    pairs.append(("RandomForest", y_test, y_pred_rf))

    y_pred_xgb = xgb.predict(X_test)
    m_xgb = metrics(y_test, y_pred_xgb); m_xgb["Model"] = "XGBoost"
    results_rows.append(m_xgb)
    pairs.append(("XGBoost", y_test, y_pred_xgb))

    # 6) Optional LSTM
    if not args.skip_lstm:
        try:
            lstm_uri = mdl("lstm.pt")
            seq_len_uri = art("seq_len.json")
            mu_uri = art("seq_mu.npy")
            std_uri = art("seq_std.npy")
            y_pred_lstm = try_load_lstm_and_predict(
                lstm_path=lstm_uri,
                X_test=X_test, X_train=X_train,
                seq_len_path=seq_len_uri, mu_path=mu_uri, std_path=std_uri
            )
            # Align length with classic y_test if needed
            if len(y_pred_lstm) != len(y_test):
                # best effort: match tail
                min_len = min(len(y_pred_lstm), len(y_test))
                y_pred_lstm = y_pred_lstm[-min_len:]
                y_test_lstm = y_test[-min_len:]
            else:
                y_test_lstm = y_test

            m_lstm = metrics(y_test_lstm, y_pred_lstm); m_lstm["Model"] = "LSTM (seq)"
            results_rows.append(m_lstm)
            pairs.append(("LSTM (seq)", y_test_lstm, y_pred_lstm))
        except Exception as e:
            print(f"ℹ️ Skipping LSTM: {e}")

    # 7) Save metrics table
    results_df = pd.DataFrame(results_rows).set_index("Model").sort_values("RMSE")
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    metrics_csv = os.path.join(args.out, f"metrics_{ts}.csv")
    results_df.to_csv(metrics_csv, float_format="%.4f")
    print("\n=== Metrics ===")
    print(results_df)
    print(f"\nSaved: {metrics_csv}")

    # 8) Save plots
    save_plots(pairs, figs_dir)
    print(f"Saved figures into: {figs_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
