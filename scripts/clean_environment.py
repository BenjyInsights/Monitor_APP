#!/usr/bin/env python3
"""clean_environment.py — Purga de estado previa al benchmark (Clean Slate).

Verifica que la GPU está en un estado limpio y reproducible antes de lanzar la
batería de experimentos:

  1. Inicializa NVML y comprueba que la GPU objetivo es accesible.
  2. Restaura el límite de potencia (power cap) al valor por defecto, para que
     una ejecución previa interrumpida no deje la GPU capada y sesgue las medidas.
  3. Avisa si hay procesos de cómputo ajenos ocupando la GPU.
  4. Garantiza que existen los directorios logs/ y results/.

NO borra datos: el respaldo de logs/results previos se hace por separado
(logs_backup_*/). Best-effort: si falta permiso root para restaurar el power cap,
lo advierte pero NO aborta (los modos que necesitan root ya se gestionan aparte).

Salida 0 = entorno listo. Salida != 0 = fallo bloqueante (GPU inaccesible).
"""
import os
import sys

GPU_INDEX = int(os.environ.get("BENCH_GPU_INDEX", "0"))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _log(msg: str) -> None:
    print(f"[CLEAN] {msg}", flush=True)


def main() -> int:
    _log("Iniciando purga de entorno (Clean Slate)...")

    # ── Directorios ────────────────────────────────────────────────────────
    for d in ("logs", "results"):
        path = os.path.join(_ROOT, d)
        os.makedirs(path, exist_ok=True)
    _log("Directorios logs/ y results/ verificados.")

    # ── NVML ───────────────────────────────────────────────────────────────
    try:
        import pynvml
    except ImportError:
        _log("ERROR: pynvml no disponible en el entorno. Abortando.")
        return 1

    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR: no se pudo inicializar NVML ({exc}). ¿Driver cargado? Abortando.")
        return 1

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        _log(f"GPU {GPU_INDEX} accesible: {name}")

        # Procesos de cómputo ajenos
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if procs:
                _log(f"AVISO: {len(procs)} proceso(s) de cómputo activos en la GPU "
                     f"(PIDs: {[p.pid for p in procs]}). Podrían contaminar las medidas.")
            else:
                _log("Sin procesos de cómputo ajenos en la GPU.")
        except Exception as exc:  # noqa: BLE001
            _log(f"AVISO: no se pudieron enumerar procesos ({exc}).")

        # Restaurar power cap por defecto (best-effort, requiere root)
        try:
            default_w = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle) // 1000
            current_w = pynvml.nvmlDeviceGetPowerManagementLimit(handle) // 1000
            if current_w != default_w:
                pynvml.nvmlDeviceSetPowerManagementLimit(handle, default_w * 1000)
                _log(f"Power cap restaurado: {current_w} W -> {default_w} W (por defecto).")
            else:
                _log(f"Power cap ya en el valor por defecto ({default_w} W).")
        except pynvml.NVMLError as exc:  # noqa: BLE001
            _log(f"AVISO: no se pudo restaurar el power cap ({exc}). "
                 f"Sin root los modos Zeus/Full operarán en advisor-only.")
    finally:
        pynvml.nvmlShutdown()

    _log("Entorno listo. Clean Slate OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
