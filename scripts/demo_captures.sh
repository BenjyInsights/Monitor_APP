#!/usr/bin/env bash
# =============================================================================
# demo_captures.sh — Demostraciones interactivas del framework
#
# Lanza tres entrenamientos cortos, cada uno configurado para ejercitar un
# componente distinto de la interfaz en tiempo real:
#
#   1  Panel de telemetría (Rich Display)
#   2  Asesor de optimización (OptimizerAdvisor)
#   3  Frontera de Pareto del optimizador de potencia (GpuPowerOptimizer)
#
# Uso:
#   bash scripts/demo_captures.sh 1
#   bash scripts/demo_captures.sh 2
#   sudo bash scripts/demo_captures.sh 3
#   bash scripts/demo_captures.sh all
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CIFAR_DIR="${PROJECT_ROOT}/models_examples/pytorch-cifar"

# Resolución del intérprete.
#
# Bajo sudo, sudoers impone secure_path, de modo que `command -v python3`
# devuelve el Python del sistema y no el del entorno virtual activo. Por eso el
# intérprete se resuelve a partir de $SUDO_USER cuando procede.
_has_torch() { [[ -n "$1" && -x "$1" ]] && "$1" -c "import torch" 2>/dev/null; }

_resolve_python() {
    # Se devuelve el primer candidato que realmente tenga PyTorch, en lugar del
    # primero que exista: en un sistema Linux /bin/python3 siempre existe, y una
    # variable de entorno sin definir se expandiría justo a esa ruta.
    local cands=()
    [[ -n "${DEMO_PYTHON:-}" ]] && cands+=("$DEMO_PYTHON")

    # Bajo sudo, sudoers impone secure_path y el intérprete del entorno del
    # usuario deja de estar en PATH; se recupera a partir de $SUDO_USER.
    if [[ -n "${SUDO_USER:-}" ]]; then
        local home_dir; home_dir=$(getent passwd "$SUDO_USER" | cut -d: -f6)
        for d in "$home_dir"/anaconda3/envs/*/bin/python3 \
                 "$home_dir"/miniconda3/envs/*/bin/python3 \
                 "$home_dir"/.venv/bin/python3; do
            [[ -e "$d" ]] && cands+=("$d")
        done
    fi

    # Entorno activo y, por último, lo que resuelva PATH.
    [[ -n "${VIRTUAL_ENV:-}" ]] && cands+=("$VIRTUAL_ENV/bin/python3")
    [[ -n "${CONDA_PREFIX:-}" ]] && cands+=("$CONDA_PREFIX/bin/python3")
    cands+=("$(command -v python3.11 2>/dev/null || true)")
    cands+=("$(command -v python3 2>/dev/null || true)")

    for c in "${cands[@]}"; do
        _has_torch "$c" && { echo "$c"; return 0; }
    done
    return 1
}

PYTHON_CMD="$(_resolve_python || true)"

if [[ -z "$PYTHON_CMD" ]]; then
    echo "ERROR: no se ha encontrado ningún intérprete de Python con PyTorch instalado." >&2
    echo "       Actívalo antes de lanzar el script, o indícalo de forma explícita:" >&2
    echo "         DEMO_PYTHON=/ruta/a/python3 bash $0 $*" >&2
    echo "       Con sudo:" >&2
    echo "         sudo DEMO_PYTHON=\$(command -v python3) bash $0 $*" >&2
    exit 1
fi
echo "  Intérprete: $PYTHON_CMD"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
cd "$CIFAR_DIR"

DEVICE="${DEVICE:-cuda:0}"

# -----------------------------------------------------------------------------
# 1 — Panel de telemetría en tiempo real
#
# ResNet18 con precisión mixta y parada por eficiencia. El panel se actualiza al
# cierre de cada época con la telemetría de hardware, el progreso del modelo y
# el coste energético acumulado. Duración aproximada: 3-4 min.
# -----------------------------------------------------------------------------
run_demo1() {
    echo ""
    echo "[1/3] Panel de telemetría en tiempo real"
    echo "      ResNet18 · lote 128 · FP16 · parada por eficiencia activa"
    echo ""
    "$PYTHON_CMD" main.py \
        --model       ResNet18 \
        --epochs      6 \
        --batch-size  128 \
        --fp16 \
        --early-stopping \
        --display \
        --device      "$DEVICE"
}

# -----------------------------------------------------------------------------
# 2 — Asesor de optimización
#
# MobileNetV2 con lote reducido y precisión simple. Esta configuración activa dos
# reglas del asesor: infrautilización por tamaño de lote y ausencia de precisión
# mixta. Las sugerencias se emiten al cierre de la primera o segunda época.
# Duración aproximada: 2-3 min.
# -----------------------------------------------------------------------------
run_demo2() {
    echo ""
    echo "[2/3] Asesor de optimización"
    echo "      MobileNetV2 · lote 32 · FP32 — configuración deliberadamente subóptima"
    echo ""
    "$PYTHON_CMD" main.py \
        --model       MobileNetV2 \
        --epochs      5 \
        --batch-size  32 \
        --device      "$DEVICE"
}

# -----------------------------------------------------------------------------
# 3 — Frontera de Pareto del optimizador de potencia
#
# ResNet18 con el optimizador activo. Las cinco primeras épocas exploran los
# límites de potencia candidatos y al terminar se imprime la tabla de candidatos
# con el punto seleccionado. Duración aproximada: 4-5 min.
#
# Requiere privilegios de administrador para modificar el límite de potencia.
# Sin ellos el módulo conmuta a modo solo-sugerencias: la tabla se imprime
# igualmente, pero todas las épocas se ejecutan al mismo límite, de modo que los
# candidatos no resultan distinguibles entre sí.
# -----------------------------------------------------------------------------
run_demo3() {
    echo ""
    echo "[3/3] Frontera de Pareto del optimizador de potencia"
    echo "      ResNet18 · lote 128 · FP16 · presupuesto temporal ±10 %"
    if [[ $EUID -ne 0 ]]; then
        echo ""
        echo "      Aviso: sin privilegios de administrador. El optimizador operará en"
        echo "      modo solo-sugerencias y los candidatos no serán distinguibles."
    fi
    echo ""
    "$PYTHON_CMD" main.py \
        --model          ResNet18 \
        --epochs         8 \
        --batch-size     128 \
        --fp16 \
        --power-optimize \
        --time-budget    0.10 \
        --device         "$DEVICE"
}

case "${1:-help}" in
    1|dashboard) run_demo1 ;;
    2|advisor)   run_demo2 ;;
    3|pareto)    run_demo3 ;;
    all)
        run_demo1
        echo ""; read -rp ">>> Pulsa Enter para continuar con la demostración 2... "
        run_demo2
        echo ""; read -rp ">>> Pulsa Enter para continuar con la demostración 3... "
        run_demo3 ;;
    *)
        echo "Uso: bash scripts/demo_captures.sh [1|2|3|all]"
        echo "  1    Panel de telemetría en tiempo real"
        echo "  2    Asesor de optimización"
        echo "  3    Frontera de Pareto (requiere sudo para control activo)"
        echo "  all  Las tres en secuencia" ;;
esac
