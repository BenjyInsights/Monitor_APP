#!/usr/bin/env bash
###############################################################################
# run_vit_5rep.sh
#
# Extensión del benchmark a un transformer (ViT compacto) para demostrar que la
# metodología (medición, EES, Pareto) generaliza más allá de las CNN.
#
#   Matriz: 1 modelo (ViT) × 4 batch × 3 modos × 5 repeticiones = 60 corridas.
#   Ejecución: SECUENCIAL en cuda:0 (mismas condiciones que las CNN del benchmark).
#   Trazabilidad y estructura idénticas a run_full_benchmark_5rep.sh.
#   Reanudable: salta corridas cuyo CSV ya exista.
###############################################################################

set -uo pipefail

MODELS=("ViT")
BATCH_SIZES=(32 64 128 256)
MODES=("control" "zeus" "full")
REPS=5
EPOCHS=50
GPU_DEVICE="cuda:0"
DEV_TAG="cuda0"

PYTHON="${PYTHON:-$(command -v python3.11 || command -v python3)}"
[ -x "$PYTHON" ] || PYTHON="python3"
MAIN="models_examples/pytorch-cifar/main.py"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs results

TOTAL=$(( ${#MODELS[@]} * ${#BATCH_SIZES[@]} * ${#MODES[@]} * REPS ))
COUNTER=0; DONE=0; SKIP=0; FAIL=0
START_TS=$(date +%s)

log()  { echo "[VIT $(date '+%Y-%m-%d %H:%M:%S')] $*"; }
line() { echo "──────────────────────────────────────────────────────────────────────"; }

log "PASO A — Purga de entorno (Clean Slate)"
sudo -n "$PYTHON" scripts/clean_environment.py || { log "ERROR: purga falló. Abortando."; exit 1; }

log "PASO B — Batería ViT: $TOTAL corridas (5 rep) en $GPU_DEVICE"
line

run_one() {
    local model="$1" bs="$2" mode="$3" rep="$4"
    COUNTER=$((COUNTER+1))
    local run_dir="logs/CIFAR10_${model}_${DEV_TAG}_bs${bs}_fp16_${mode}_rep${rep}"
    local id="${model}_bs${bs}_${mode}_rep${rep}"
    local model_log="logs/${model}.log"

    if compgen -G "${run_dir}/*_energy_metrics.csv" > /dev/null; then
        log "[$COUNTER/$TOTAL] SKIP (ya existe): $id"; SKIP=$((SKIP+1)); return 0
    fi

    local flags="--fp16"
    case "$mode" in
        zeus) flags="$flags --power-optimize" ;;
        full) flags="$flags --power-optimize --early-stopping" ;;
    esac

    local cmd="sudo -n $PYTHON -u $MAIN --model $model --batch-size $bs --epochs $EPOCHS \
--device $GPU_DEVICE --rep $rep --quiet $flags"

    {
        echo "=== BENCHMARK RUN (ViT) ============================================"
        echo "ID: $id   MODELO: $model   BATCH: $bs   MODO: $mode   REP: $rep/$REPS"
        echo "DIR: $run_dir"; echo "COMANDO: $cmd"; echo "INICIO: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================================================"
    } >> "$model_log"

    log "[$COUNTER/$TOTAL] EJECUTANDO: $id"
    local t0; t0=$(date +%s)
    if $cmd >> "$model_log" 2>&1; then
        DONE=$((DONE+1)); log "[$COUNTER/$TOTAL] OK ($(( $(date +%s)-t0 ))s): $id"
    else
        FAIL=$((FAIL+1)); log "[$COUNTER/$TOTAL] FALLO ($(( $(date +%s)-t0 ))s): $id (continúa)"
    fi
    line
}

for model in "${MODELS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        for mode in "${MODES[@]}"; do
            for rep in $(seq 1 "$REPS"); do
                run_one "$model" "$bs" "$mode" "$rep"
            done
        done
    done
done

line
log "Batería ViT completada. OK=$DONE SKIP=$SKIP FAIL=$FAIL (de $TOTAL)"
log "PASO D — Reconstruyendo master dataset (incluye ViT)"
"$PYTHON" tools/build_master_dataset.py || log "AVISO: build_master_dataset falló."
"$PYTHON" tools/generate_benchmark_analysis.py || log "AVISO: generate_benchmark_analysis falló."
TOT=$(( $(date +%s) - START_TS ))
log "FIN ViT. Tiempo: $((TOT/3600))h $(((TOT%3600)/60))m."
