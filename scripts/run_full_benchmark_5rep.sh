#!/usr/bin/env bash
###############################################################################
# run_full_benchmark_5rep.sh
#
# Batería maestra del banco de pruebas, con repeticiones para estadística robusta.
#
#   Matriz: 6 modelos × 4 batch × 3 modos × 5 repeticiones = 360 corridas.
#   Ejecución: SECUENCIAL en cuda:0 (condiciones idénticas a los datos previos).
#   Trazabilidad: 1 corrida = 1 carpeta  logs/CIFAR10_<m>_cuda0_bs<b>_fp16_<modo>_rep<N>/
#                 con su NDJSON + *_energy_metrics.csv + *_layer_profile.csv.
#   Reanudable: salta cualquier corrida cuyo CSV de energía ya exista.
#
# Modos:
#   control -> baseline (sin optimización)
#   zeus    -> --power-optimize            (REQUIERE root: fija power cap vía NVML)
#   full    -> --power-optimize --early-stopping (root + EES)
#
# Requiere sudoers NOPASSWD para el intérprete (ver README de lanzamiento).
###############################################################################

set -uo pipefail   # NO -e: una corrida fallida NO debe abortar la batería

# ── Configuración ────────────────────────────────────────────────────────────
MODELS=("ResNet18" "ResNet50" "VGG19" "MobileNetV2" "DenseNet121" "EfficientNetB0")
BATCH_SIZES=(32 64 128 256)
MODES=("control" "zeus" "full")
REPS=5
EPOCHS=50
GPU_DEVICE="cuda:0"
DEV_TAG="cuda0"
COUNTRY="Spain"

PYTHON="${PYTHON:-$(command -v python3.11 || command -v python3)}"
[ -x "$PYTHON" ] || PYTHON="python3"
MAIN="models_examples/pytorch-cifar/main.py"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs results

TOTAL=$(( ${#MODELS[@]} * ${#BATCH_SIZES[@]} * ${#MODES[@]} * REPS ))
COUNTER=0
DONE=0; SKIP=0; FAIL=0
START_TS=$(date +%s)

log()  { echo "[BENCH $(date '+%Y-%m-%d %H:%M:%S')] $*"; }
line() { echo "──────────────────────────────────────────────────────────────────────"; }

# ── Purga / Clean Slate (con root para poder restaurar el power cap) ─────────
log "PASO A — Purga de entorno (Clean Slate)"
sudo -n "$PYTHON" scripts/clean_environment.py || {
    log "ERROR: la purga falló. Abortando."; exit 1; }

log "PASO B — Iniciando batería: $TOTAL corridas (5 rep) en $GPU_DEVICE"
log "         Modelos=${#MODELS[@]}  Batch=${#BATCH_SIZES[@]}  Modos=${#MODES[@]}  Rep=$REPS  Épocas=$EPOCHS"
line

run_one() {
    local model="$1" bs="$2" mode="$3" rep="$4"
    COUNTER=$((COUNTER+1))

    local run_dir="logs/CIFAR10_${model}_${DEV_TAG}_bs${bs}_fp16_${mode}_rep${rep}"
    local id="${model}_bs${bs}_${mode}_rep${rep}"
    local model_log="logs/${model}.log"

    # ── Reanudación: saltar si ya está completa ──
    if compgen -G "${run_dir}/*_energy_metrics.csv" > /dev/null; then
        log "[$COUNTER/$TOTAL] SKIP (ya existe): $id"
        SKIP=$((SKIP+1)); return 0
    fi

    # ── Flags por modo ──
    local flags="--fp16"
    case "$mode" in
        control) ;;
        zeus)    flags="$flags --power-optimize" ;;
        full)    flags="$flags --power-optimize --early-stopping" ;;
    esac

    # ── Elevación: TODOS los modos bajo root para acceso uniforme a RAPL (CPU
    #    energy vía /sys/.../energy_uj requiere root) y al power cap NVML (Zeus/Full).
    #    Misma metodología que el resto de la batería. ──
    local cmd="sudo -n $PYTHON -u $MAIN --model $model --batch-size $bs --epochs $EPOCHS \
--device $GPU_DEVICE --rep $rep --quiet $flags"

    {
        echo "=== BENCHMARK RUN ==================================================="
        echo "ID:      $id"
        echo "MODELO:  $model   BATCH: $bs   MODO: $mode   REP: $rep/$REPS"
        echo "DIR:     $run_dir"
        echo "COMANDO: $cmd"
        echo "INICIO:  $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================================================"
    } >> "$model_log"

    log "[$COUNTER/$TOTAL] EJECUTANDO: $id"
    local t0 t1
    t0=$(date +%s)

    if $cmd >> "$model_log" 2>&1; then
        t1=$(date +%s)
        DONE=$((DONE+1))
        log "[$COUNTER/$TOTAL] OK ($((t1-t0))s): $id"
        echo "FIN OK: $(date '+%Y-%m-%d %H:%M:%S')  duración=$((t1-t0))s" >> "$model_log"
    else
        t1=$(date +%s)
        FAIL=$((FAIL+1))
        log "[$COUNTER/$TOTAL] FALLO ($((t1-t0))s): $id  (continúa la batería)"
        echo "FIN FALLO: $(date '+%Y-%m-%d %H:%M:%S')  duración=$((t1-t0))s" >> "$model_log"
    fi

    # progreso global + ETA
    local elapsed avg remain eta
    elapsed=$(( $(date +%s) - START_TS ))
    if [ "$((DONE+FAIL))" -gt 0 ]; then
        avg=$(( elapsed / (DONE+FAIL) ))
        remain=$(( (TOTAL - COUNTER) * avg ))
        eta=$(date -d "+${remain} seconds" '+%Y-%m-%d %H:%M' 2>/dev/null || echo "?")
        log "     Progreso: OK=$DONE  SKIP=$SKIP  FAIL=$FAIL  |  media=${avg}s/run  ETA≈$eta"
    fi
    line
}

# ── Bucle principal: modelo → batch → modo → repetición ─────────────────────
for model in "${MODELS[@]}"; do
    log ">> Suite modelo: $model"
    for bs in "${BATCH_SIZES[@]}"; do
        for mode in "${MODES[@]}"; do
            for rep in $(seq 1 "$REPS"); do
                run_one "$model" "$bs" "$mode" "$rep"
            done
        done
    done
done

line
log "PASO C — Batería completada.  OK=$DONE  SKIP=$SKIP  FAIL=$FAIL  (de $TOTAL)"

# ── Análisis post-hoc (best-effort) ─────────────────────────────────────────
log "PASO D — Análisis post-hoc"
log "   -> Reconstruyendo master dataset (modo+rep desde naming)..."
"$PYTHON" tools/build_master_dataset.py || log "   AVISO: build_master_dataset falló."
log "   -> Agregando resultados a CSV..."
"$PYTHON" tools/analyze_results.py --logs-dir logs/ --csv results/results_summary.csv 2>/dev/null || log "   AVISO: analyze_results falló."
log "   -> Generando tablas del benchmark..."
"$PYTHON" tools/generate_benchmark_analysis.py || log "   AVISO: generate_benchmark_analysis falló."
log "   -> Generando resumen ejecutivo..."
"$PYTHON" tools/generate_executive_summary.py || log "   AVISO: generate_executive_summary falló."

TOTAL_ELAPSED=$(( $(date +%s) - START_TS ))
log "FIN. Tiempo total: $((TOTAL_ELAPSED/3600))h $(((TOTAL_ELAPSED%3600)/60))m."
log "Datos en: logs/CIFAR10_*_<modo>_rep<N>/   |   Tablas en: results/"
