#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# analyze_results.py
# Aggregates all benchmark CSVs under logs/ and prints comparison tables.
#
# Usage:
#   python models_examples/analyze_results.py --logs-dir logs/
#   python models_examples/analyze_results.py --logs-dir logs/ --csv results_summary.csv

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# ── Config name parsing ───────────────────────────────────────────────────────
_DIR_RE = re.compile(
    r"cifar10_(?P<model>[^_]+)_(?P<device>[^_]+)_bs(?P<batch>\d+)_(?P<prec>fp\d+)",
    re.IGNORECASE,
)

def _parse_config(dir_name: str) -> dict | None:
    m = _DIR_RE.search(dir_name)
    if not m:
        return None
    return {
        "model":  m.group("model").upper(),
        "device": m.group("device").lower(),
        "batch":  int(m.group("batch")),
        "prec":   m.group("prec").lower(),
    }


# ── Data loading ──────────────────────────────────────────────────────────────
def load_all_runs(logs_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(logs_dir.rglob("*_energy_metrics.csv")):
        config = _parse_config(csv_path.parent.name)
        if config is None:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        df_valid = df[df["epoch"] > 0]  # skip epoch 0 (CUDA warmup)
        if df_valid.empty:
            continue

        row = {**config}
        row["run"]                  = csv_path.stem.replace("_energy_metrics", "")
        row["epochs"]               = len(df_valid)
        row["mean_energy_j"]        = df_valid["total_energy_j"].mean()
        row["mean_energy_per_sample_j"] = df_valid["energy_per_sample_j"].mean()
        row["mean_duration_s"]      = df_valid["duration_s"].mean()
        row["mean_edp"]             = df_valid["edp"].mean()
        row["final_accuracy"]       = df["accuracy"].dropna().iloc[-1] if "accuracy" in df.columns else None
        co2_cols = [c for c in df.columns if c.startswith("co2_") and "europe" in c]
        if co2_cols:
            row["mean_co2_europe_g"] = df_valid[co2_cols[0]].mean()
        rows.append(row)

    if not rows:
        print("No *_energy_metrics.csv files found in", logs_dir)
        sys.exit(1)

    return pd.DataFrame(rows)


# ── Aggregation ───────────────────────────────────────────────────────────────
def aggregate(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    metrics = ["mean_energy_j", "mean_energy_per_sample_j", "mean_duration_s", "mean_co2_europe_g"]
    metrics = [m for m in metrics if m in df.columns]
    agg = df.groupby(group_cols)[metrics].agg(["mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg["n_reps"] = df.groupby(group_cols)["run"].count()
    return agg.reset_index()


# ── Printing ──────────────────────────────────────────────────────────────────
def print_table(title: str, df: pd.DataFrame) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(df.to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results")
    parser.add_argument("--logs-dir", default="logs", help="Root logs directory")
    parser.add_argument("--csv",      default=None,   help="Save summary to CSV")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        sys.exit(1)

    df = load_all_runs(logs_dir)
    print(f"\nLoaded {len(df)} runs from {logs_dir}")

    gpu_df  = df[df["device"] != "cpu"]
    cpu_df  = df[df["device"] == "cpu"]

    # 1. Architecture comparison (GPU FP32, batch=128)
    arch_df = gpu_df[(gpu_df["prec"] == "fp32") & (gpu_df["batch"] == 128)]
    if not arch_df.empty:
        print_table("Architecture comparison — GPU FP32, batch=128",
                    aggregate(arch_df, ["model"]))

    # 2. Batch-size effect (GPU FP32, ResNet18)
    bs_df = gpu_df[(gpu_df["prec"] == "fp32") & (gpu_df["model"] == "RESNET18")]
    if not bs_df.empty:
        print_table("Batch-size effect — GPU FP32, ResNet18",
                    aggregate(bs_df, ["batch"]))

    # 3. FP32 vs FP16 (GPU, batch=128)
    prec_df = gpu_df[gpu_df["batch"] == 128]
    if not prec_df.empty:
        print_table("FP32 vs FP16 — GPU, batch=128",
                    aggregate(prec_df, ["model", "prec"]))

    # 4. GPU vs CPU (batch=128, FP32)
    if not cpu_df.empty:
        comp_df = pd.concat([
            gpu_df[(gpu_df["prec"] == "fp32") & (gpu_df["batch"] == 128)],
            cpu_df,
        ])
        print_table("GPU vs CPU — FP32, batch=128",
                    aggregate(comp_df, ["model", "device"]))

    # 5. Full summary
    print_table("Full summary — all runs",
                aggregate(df, ["model", "device", "batch", "prec"]))

    if args.csv:
        summary = aggregate(df, ["model", "device", "batch", "prec"])
        summary.to_csv(args.csv, index=False)
        print(f"\nSummary saved to: {args.csv}")


if __name__ == "__main__":
    main()
