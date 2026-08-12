#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_master_dataset.py

Reconstruye benchmark_master_dataset.json desde los CSV de energía en
logs/CIFAR10_*.

Naming esperado (una carpeta = una corrida, trazabilidad 1:1):

    CIFAR10_{model}_{device}_bs{batch}_{prec}_{mode}_rep{N}

    p.ej. CIFAR10_ResNet18_cuda0_bs256_fp16_control_rep1

El modo y la repetición se leen DIRECTAMENTE del nombre del directorio, sin
depender del orden temporal de los ficheros (que fallaba con >1 repetición).

Compatibilidad: si encuentra carpetas con el naming antiguo (sin _mode_repN) y
exactamente 3 CSVs, aplica el criterio legacy de orden temporal
(Control, Zeus_Only, Full_Optimized) para no romper datasets previos.

Salida: benchmark_master_dataset.json en la raíz del proyecto.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
OUT_FILE = BASE_DIR / "benchmark_master_dataset.json"

# tag de modo en el nombre  ->  nombre canónico + flags
MODE_TAG_TO_CANON = {
    "control": "Control",
    "zeus":    "Zeus_Only",
    "full":    "Full_Optimized",
}
MODE_FLAGS = {
    "Control":        {"power_optimize": False, "early_stopping": False},
    "Zeus_Only":      {"power_optimize": True,  "early_stopping": False},
    "Full_Optimized": {"power_optimize": True,  "early_stopping": True},
}
MODE_ORDER = {"Control": 0, "Zeus_Only": 1, "Full_Optimized": 2}

# naming nuevo: ..._bs{batch}_{prec}_{mode}_rep{N}
DIR_RE_NEW = re.compile(
    r"CIFAR10_(?P<model>[^_]+)_(?P<device>[^_]+)_bs(?P<batch>\d+)_(?P<prec>fp\d+)"
    r"_(?P<mode>control|zeus|full)_rep(?P<rep>\d+)$"
)
# naming legacy: ..._bs{batch}_{prec}   (sin modo ni repetición)
DIR_RE_LEGACY = re.compile(
    r"CIFAR10_(?P<model>[^_]+)_(?P<device>[^_]+)_bs(?P<batch>\d+)_(?P<prec>fp\d+)$"
)
LEGACY_SEQUENCE = ["Control", "Zeus_Only", "Full_Optimized"]


def summarize_run(csv_path: Path, mode: str, batch: int, rep: int) -> dict | None:
    """Resume una corrida a partir de su CSV de métricas por época.

    Se agregan TODAS las épocas registradas (la numeración del CSV empieza en 0),
    porque la época 0 es una época de entrenamiento real cuya energía se consumió
    de hecho. Excluirla falsearía a la baja la energía total, el CO2 y el recuento
    de épocas: en las corridas cortas con parada temprana el sesgo llega al 10-14 %.
    """
    df = pd.read_csv(csv_path)
    df_train = df.dropna(subset=["epoch"])
    if df_train.empty:
        return None

    energy_total = float(df_train["total_energy_j"].sum())
    energy_mean_epoch = float(df_train["total_energy_j"].mean())
    j_sample_mean = float(df_train["energy_per_sample_j"].mean())
    j_sample_std = float(df_train["energy_per_sample_j"].std(ddof=1)) if len(df_train) > 1 else 0.0
    final_acc = float(df_train["accuracy"].dropna().iloc[-1])
    co2_g_total = float(df_train["co2_spain_g"].sum()) if "co2_spain_g" in df_train.columns else 0.0

    epochs_history = []
    for _, row in df_train.iterrows():
        epochs_history.append({
            "epoch": int(row["epoch"]),
            "duration_s": float(row.get("duration_s", 0.0)),
            "energy_j": float(row.get("total_energy_j", 0.0)),
            "energy_per_sample_j": float(row.get("energy_per_sample_j", 0.0)),
            "accuracy": float(row.get("accuracy", 0.0)) if pd.notna(row.get("accuracy")) else 0.0,
            "edp": float(row.get("edp", 0.0)),
            "gpu_energy_wh": float(row.get("gpu_energy_wh", 0.0)),
            "cpu_energy_wh": float(row.get("cpu_energy_wh", 0.0)),
        })

    return {
        "batch_size": batch,
        "rep": rep,
        "mode": mode,
        "optimization_mode": mode,
        "flags": MODE_FLAGS[mode],
        "epochs_completed": int(len(df_train)),
        "summary": {
            "energy_j_total":      round(energy_total, 4),
            "energy_j_mean_epoch": round(energy_mean_epoch, 4),
            "j_sample_mean":       round(j_sample_mean, 6),
            "j_sample_std":        round(j_sample_std, 6),
            "final_accuracy":      round(final_acc, 6),
            "co2_g_total":         round(co2_g_total, 4),
            "energy_grade":        None,
        },
        "epochs_history": epochs_history,
        "source_csv": csv_path.name,
    }


def main():
    if not LOGS_DIR.exists():
        print(f"ERROR: {LOGS_DIR} no existe.", file=sys.stderr)
        sys.exit(1)

    by_model: dict[str, list] = {}
    n_new = n_legacy = n_skipped = 0

    for d in sorted(LOGS_DIR.iterdir()):
        if not d.is_dir():
            continue

        csvs = sorted(d.glob("*_energy_metrics.csv"))

        m_new = DIR_RE_NEW.match(d.name)
        if m_new:
            # naming nuevo: 1 carpeta = 1 corrida
            if len(csvs) == 0:
                print(f"⚠️  {d.name}: sin CSV de energía. Saltando.", file=sys.stderr)
                n_skipped += 1
                continue
            if len(csvs) > 1:
                print(f"⚠️  {d.name}: {len(csvs)} CSVs (esperaba 1). Uso el más reciente.",
                      file=sys.stderr)
            mode = MODE_TAG_TO_CANON[m_new.group("mode")]
            run = summarize_run(csvs[-1], mode, int(m_new.group("batch")), int(m_new.group("rep")))
            if run:
                by_model.setdefault(m_new.group("model"), []).append(run)
                n_new += 1
            continue

        m_leg = DIR_RE_LEGACY.match(d.name)
        if m_leg:
            # naming legacy: 3 CSVs por orden temporal
            if len(csvs) != 3:
                print(f"⚠️  {d.name} (legacy): {len(csvs)} CSVs (esperaba 3). Saltando.",
                      file=sys.stderr)
                n_skipped += 1
                continue
            for csv_path, mode in zip(csvs, LEGACY_SEQUENCE):
                run = summarize_run(csv_path, mode, int(m_leg.group("batch")), 1)
                if run:
                    by_model.setdefault(m_leg.group("model"), []).append(run)
                    n_legacy += 1
            continue

        # nombre no reconocido
        if csvs:
            print(f"⚠️  {d.name}: naming no reconocido. Saltando.", file=sys.stderr)
            n_skipped += 1

    # orden estable dentro de cada modelo: (batch, modo, rep)
    for rs in by_model.values():
        rs.sort(key=lambda r: (r["batch_size"], MODE_ORDER.get(r["mode"], 9), r.get("rep", 1)))

    dataset = [{"model_name": m, "runs": rs} for m, rs in sorted(by_model.items())]

    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    total_runs = sum(len(rs) for rs in by_model.values())
    print(f"✅ Master dataset escrito: {OUT_FILE}")
    print(f"   Corridas leídas: naming-nuevo={n_new}  legacy={n_legacy}  saltadas={n_skipped}")
    print(f"   Modelos: {len(by_model)}   Runs totales: {total_runs}")
    for m, rs in by_model.items():
        modes: dict[str, int] = {}
        for r in rs:
            modes[r["mode"]] = modes.get(r["mode"], 0) + 1
        print(f"   - {m}: {len(rs)} runs  {modes}")


if __name__ == "__main__":
    main()
