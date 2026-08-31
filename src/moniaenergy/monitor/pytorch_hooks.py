# moniaenergy/monitor/pytorch_hooks.py #
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2026  Benjamín Sánchez Calza
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import json
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

import pandas as pd

# green_grader import removed for architectural purity

# Approximate idle (no-load) GPU power baseline for an RTX 6000 Ada generation.
# This is subtracted from total board power before process attribution to obtain
# the *dynamic* power driven by the training workload.  Override by setting the
# MONITOR_GPU_IDLE_W environment variable (float).
_GPU_IDLE_POWER_W: float = float(os.environ.get("MONITOR_GPU_IDLE_W", "15.0"))

class EpochTracker:
    """Writes semantic epoch-boundary events into the NDJSON monitoring log.

    Injecting epoch boundaries allows compute_energy_metrics() to attribute
    energy readings collected by the background monitor to individual training
    epochs, yielding per-epoch energy, carbon, and efficiency metrics.

    Epoch events are written to a sidecar file (<stem>.events.ndjson) instead
    of the main log, avoiding write-position conflicts with the monitor worker
    process.

    Attributes
    ----------
    _log_file_path : str
        Path to the main NDJSON monitoring log file.
    _events_path : str
        Path to the sidecar events file.

    Examples
    --------
    >>> with MonitorContext(log_file_path="logs/resnet_train") as mon:
    ...     tracker = EpochTracker(mon.monitor.log_file_path)
    ...     for epoch in range(num_epochs):
    ...         tracker.on_epoch_start(epoch)
    ...         loss, acc = train_one_epoch(loader, model, optimizer)
    ...         tracker.on_epoch_end(epoch, samples=len(dataset), loss=loss, accuracy=acc)
    """

    def __init__(self, log_file_path: str) -> None:
        """Initialize the epoch tracker.

        Parameters
        ----------
        log_file_path : str
            Path to the NDJSON monitoring log file (ending in '.ndjson').
            Epoch events are written to a sidecar file with suffix
            '.events.ndjson'.
        """
        self._log_file_path = log_file_path
        # Sidecar: same stem, different suffix
        stem = log_file_path[:-len(".ndjson")] if log_file_path.endswith(".ndjson") else log_file_path
        self._events_path = stem + ".events.ndjson"

    def on_epoch_start(self, epoch: int) -> None:
        """Record the start of a training epoch.

        Writes an epoch_start event to the sidecar events file with a
        timestamp.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        """
        self._append({
            "event": "epoch_start",
            "epoch": int(epoch),
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })

    def on_epoch_end(
        self,
        epoch: int,
        samples: int = 0,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
    ) -> None:
        """Record the end of a training epoch.

        Writes an epoch_end event to the sidecar events file with optional
        training metrics.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index (must match corresponding on_epoch_start).
        samples : int, optional
            Number of training samples processed in this epoch. Default is 0.
        loss : float, optional
            Training loss at the end of the epoch. Default is None.
        accuracy : float, optional
            Training accuracy (0-1 or 0-100, user convention).
            Default is None.
        """
        event: Dict[str, Any] = {
            "event": "epoch_end",
            "epoch": int(epoch),
            "samples": int(samples),
            "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        if loss is not None:
            event["loss"] = float(loss)
        if accuracy is not None:
            event["accuracy"] = float(accuracy)
        self._append(event)

    def _append(self, data: Dict[str, Any]) -> None:
        """Append an event to the sidecar events file.

        Parameters
        ----------
        data : dict
            Event data to serialize as JSON and write.
        """
        log_dir = os.path.dirname(self._events_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self._events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            f.flush()


class LayerEnergyProfiler:
    """Measures per-layer forward-pass compute time within a PyTorch model.

    Uses torch.cuda.Event (microsecond precision, non-blocking) on CUDA, or
    time.perf_counter() on CPU. Profiles only leaf modules (no child submodules)
    to avoid double-counting in nested containers.

    Per-layer timings can be combined with per-epoch GPU energy from
    compute_energy_metrics() to attribute energy to individual layers using
    their fractional compute time.

    Attributes
    ----------
    _cuda_available : bool
        Whether CUDA is available and should be used for timing.
    _handles : list
        List of registered hook handles to be removed on cleanup.
    _timings : dict
        Accumulated timing records per layer. Maps layer name to list of
        floats (CPU timing, ms) or list of tuples (CUDA events).
    _pending_local : threading.local
        Thread-local storage for pre/post-hook communication.

    Examples
    --------
    >>> profiler = LayerEnergyProfiler(model, device="cuda")
    >>> # ... run training epochs ...
    >>> df = profiler.get_summary()
    >>> profiler.remove()
    """

    def __init__(self, model: Any, device: str = "cuda") -> None:
        """Initialize the layer profiler.

        Registers forward pre-hooks and post-hooks on all leaf modules.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model whose forward passes will be profiled.
        device : str, optional
            Device type: "cuda" or "cpu". Default is "cuda".
            Determines whether torch.cuda.Event (CUDA) or
            time.perf_counter() (CPU) is used for timing.

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        """
        try:
            import torch
            self._torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch must be installed to use LayerEnergyProfiler. "
                "Install it via: pip install torch"
            )

        self._cuda_available = (
            device == "cuda" and self._torch.cuda.is_available()
        )
        self._handles: List[Any] = []
        self._timings: Dict[str, List[Any]] = defaultdict(list)
        self._pending_local = threading.local()

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf module only
                h_pre = module.register_forward_pre_hook(self._make_pre_hook(name))
                h_post = module.register_forward_hook(self._make_post_hook(name))
                self._handles.extend([h_pre, h_post])

    def _make_pre_hook(self, name: str) -> Any:
        """Create a forward pre-hook for a named module.

        Parameters
        ----------
        name : str
            Module name for timing identification.

        Returns
        -------
        callable
            Pre-hook function to register with the module.
        """
        def pre_hook(module: Any, input: Any) -> None:
            if not hasattr(self._pending_local, 'data'):
                self._pending_local.data = {}
            if self._cuda_available:
                start = self._torch.cuda.Event(enable_timing=True)
                start.record()
                self._pending_local.data[name] = start
            else:
                self._pending_local.data[name] = time.perf_counter()
        return pre_hook

    def _make_post_hook(self, name: str) -> Any:
        """Create a forward post-hook for a named module.

        Parameters
        ----------
        name : str
            Module name for timing identification.

        Returns
        -------
        callable
            Post-hook function to register with the module.
        """
        def post_hook(module: Any, input: Any, output: Any) -> None:
            if not hasattr(self._pending_local, 'data'):
                return
            if self._cuda_available:
                if name in self._pending_local.data:
                    end = self._torch.cuda.Event(enable_timing=True)
                    end.record()
                    self._timings[name].append((self._pending_local.data.pop(name), end))
            else:
                if name in self._pending_local.data:
                    elapsed_ms = (time.perf_counter() - self._pending_local.data.pop(name)) * 1000.0
                    self._timings[name].append(elapsed_ms)
        return post_hook

    def reset(self) -> None:
        """Clear all accumulated timing data without removing hooks."""
        self._timings.clear()
        if hasattr(self._pending_local, 'data'):
            self._pending_local.data.clear()

    def remove(self) -> None:
        """Remove all registered hooks from the model."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_summary(self) -> pd.DataFrame:
        """Get a summary of per-layer compute times.

        Returns a DataFrame sorted by total compute time (descending).
        CUDA events are synchronized lazily here for efficiency.
        Multi-GPU safe: synchronizes each device that owns recorded events.

        Returns
        -------
        pd.DataFrame
            One row per profiled layer with columns:
            - layer: Layer name
            - calls: Number of forward passes
            - total_ms: Total compute time (ms)
            - avg_ms: Average compute time per call (ms)
            - min_ms: Minimum compute time (ms)
            - max_ms: Maximum compute time (ms)
            - time_fraction: Layer's fractional share of total compute time
        """
        if not self._timings:
            return pd.DataFrame(
                columns=["layer", "calls", "total_ms", "avg_ms", "min_ms", "max_ms", "time_fraction"]
            )

        if self._cuda_available:
            # Synchronize every device that has pending events (multi-GPU safe)
            synced_devices: set = set()
            for records in self._timings.values():
                for start, _end in records:
                    try:
                        dev = start.device if hasattr(start, 'device') else None
                        if dev not in synced_devices:
                            if dev is not None:
                                self._torch.cuda.synchronize(dev)
                            else:
                                self._torch.cuda.synchronize()
                            synced_devices.add(dev)
                    except Exception:
                        # Fallback: synchronize current device
                        try:
                            self._torch.cuda.synchronize()
                        except Exception:
                            pass
                        break

        rows = []
        for layer, records in self._timings.items():
            if self._cuda_available:
                times = [start.elapsed_time(end) for start, end in records]
            else:
                times = records
            rows.append({
                "layer": layer,
                "calls": len(times),
                "total_ms": round(sum(times), 3),
                "avg_ms": round(sum(times) / len(times), 3),
                "min_ms": round(min(times), 3),
                "max_ms": round(max(times), 3),
            })

        df = pd.DataFrame(rows).sort_values("total_ms", ascending=False).reset_index(drop=True)
        total = df["total_ms"].sum()
        df["time_fraction"] = (df["total_ms"] / total).round(4) if total > 0 else 0.0
        return df


def compute_energy_metrics(log_path: str) -> pd.DataFrame:
    """Compute per-epoch energy and carbon metrics from NDJSON logs.

    Reads the main monitoring log and sidecar epoch events file, then
    sums hardware metrics (GPU/CPU energy, carbon emissions) within each
    epoch's time window.

    Parameters
    ----------
    log_path : str
        Path to the .ndjson monitoring log file (written by BaseMonitor).

    Returns
    -------
    pd.DataFrame
        One row per complete epoch with columns:

        - epoch: Zero-based epoch index
        - duration_s: Wall-clock epoch duration (seconds)
        - samples: Number of training samples (from epoch_end event)
        - loss: Training loss (if provided, else None)
        - accuracy: Training accuracy (if provided, else None)
        - gpu_energy_wh: GPU energy consumed (Wh)
        - cpu_energy_wh: CPU process energy consumed (Wh, via RAPL)
        - total_energy_j: Total energy (gpu + cpu) in joules
        - energy_per_sample_j: total_energy_j / samples (None if samples=0)
        - edp: Energy-Delay Product (J·s)
        - energy_grade: Letter grade (A++ to F)
        - co2_<continent>_g: Cumulative CO₂ per continent (grams)

        Returns empty DataFrame if no epoch events found.
    """
    metric_entries: List[Dict[str, Any]] = []
    epoch_events: List[Dict[str, Any]] = []
    fmt = '%Y-%m-%d %H:%M:%S'

    # Epoch events live in a sidecar file to avoid write conflicts
    stem = log_path[:-len(".ndjson")] if log_path.endswith(".ndjson") else log_path
    events_path = stem + ".events.ndjson"

    def _parse_lines(path: str, collect_events: bool) -> None:
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if collect_events and "event" in obj and obj["event"] in ("epoch_start", "epoch_end"):
                    try:
                        obj["_ts"] = datetime.strptime(obj["Timestamp"], fmt)
                        epoch_events.append(obj)
                    except (KeyError, ValueError):
                        pass
                elif "Timestamp" in obj and "event" not in obj:
                    try:
                        obj["_ts"] = datetime.strptime(obj["Timestamp"], fmt)
                        metric_entries.append(obj)
                    except ValueError:
                        pass

    _parse_lines(log_path, collect_events=False)   # metric entries only
    _parse_lines(events_path, collect_events=True)  # epoch events from sidecar

    if not epoch_events:
        print(
            "[compute_energy_metrics] No epoch events found. "
            f"Expected sidecar file: {events_path}\n"
            "Make sure EpochTracker.on_epoch_start() / on_epoch_end() were called."
        )
        return pd.DataFrame()

    starts = {ev["epoch"]: ev for ev in epoch_events if ev["event"] == "epoch_start"}
    ends = {ev["epoch"]: ev for ev in epoch_events if ev["event"] == "epoch_end"}
    complete_epochs = sorted(set(starts.keys()) & set(ends.keys()))

    if not complete_epochs:
        print("[compute_energy_metrics] No complete epoch pairs (start + end) found.")
        return pd.DataFrame()

    rows = []
    for epoch in complete_epochs:
        start_ts = starts[epoch]["_ts"]
        end_ts = ends[epoch]["_ts"]
        end_ev = ends[epoch]

        window = [e for e in metric_entries if start_ts <= e["_ts"] <= end_ts]

        # --- GPU energy (Wh) and CO2 (g CO2) per continent ---
        gpu_energy_wh = 0.0
        co2_by_continent: Dict[str, float] = defaultdict(float)

        for e in window:
            gpu_section = e.get("NVIDIA GPU Info")
            if not isinstance(gpu_section, dict):
                continue
            for gpu_data in gpu_section.values():
                if not isinstance(gpu_data, dict):
                    continue
                # Prefer process-specific energy when available
                _proc_wh = gpu_data.get("Process Power Usage Interval (Wh)")
                gpu_energy_wh += (
                    _proc_wh
                    if _proc_wh is not None
                    else gpu_data.get("Power Usage Interval (Wh)", 0.0)
                ) or 0.0
                # Process-fraction CO2 if available
                _proc_co2 = gpu_data.get("Process Carbon Emissions Estimated (g CO2)")
                carbon = (
                    _proc_co2
                    if isinstance(_proc_co2, dict)
                    else gpu_data.get("Carbon Emissions Estimated (g CO2)", {})
                )
                if isinstance(carbon, dict):
                    for continent, val in carbon.items():
                        if val is not None:
                            co2_by_continent[continent] += float(val)

        # --- CPU process energy (trapezoidal integration) ---
        cpu_energy_wh = 0.0
        cpu_series = [
            (e["_ts"], e["CPU RAPL Process Package Power (W)"])
            for e in window
            if "CPU RAPL Process Package Power (W)" in e
            and e["CPU RAPL Process Package Power (W)"] is not None
        ]
        for i in range(1, len(cpu_series)):
            ts_prev, pwr_prev = cpu_series[i - 1]
            ts_curr, pwr_curr = cpu_series[i]
            dt_h = (ts_curr - ts_prev).total_seconds() / 3600.0
            cpu_energy_wh += ((pwr_prev + pwr_curr) / 2.0) * dt_h

        # --- Derived metrics ---
        duration_s = (end_ts - start_ts).total_seconds()
        total_energy_j = (gpu_energy_wh + cpu_energy_wh) * 3600.0
        samples = int(end_ev.get("samples") or 0)
        energy_per_sample_j = (total_energy_j / samples) if samples > 0 else None
        edp = (total_energy_j * duration_s) if duration_s > 0 else None

        row: Dict[str, Any] = {
            "epoch": epoch,
            "duration_s": round(duration_s, 2),
            "samples": samples,
            "loss": end_ev.get("loss"),
            "accuracy": end_ev.get("accuracy"),
            "gpu_energy_wh": round(gpu_energy_wh, 6),
            "cpu_energy_wh": round(cpu_energy_wh, 6),
            "total_energy_j": round(total_energy_j, 4),
            "energy_per_sample_j": round(energy_per_sample_j, 6) if energy_per_sample_j is not None else None,
            "edp": round(edp, 4) if edp is not None else None,
        }

        # Energy grade computation has been moved to the post-hoc analysis pipeline
        # to ensure it can receive the correct model parameters for the Green AI Grade.

        for continent, val in co2_by_continent.items():
            col = "co2_" + continent.lower().replace(" ", "_").replace("-", "_") + "_g"
            row[col] = round(val, 6)

        rows.append(row)

    return pd.DataFrame(rows)


class EnergyEarlyStopping:
    """Stops training when accuracy-per-joule efficiency decays too much.

    The threshold ratio auto-calibrates from the first productive epoch:
        threshold = min_efficiency_ratio × first_epoch_efficiency.
    This makes the controller hardware-agnostic, adapting automatically to
    each training's energy scales and hardware.

    Prints progress to stdout every epoch for transparency during training.

    Attributes
    ----------
    _log_file_path : str
        Path to the NDJSON monitoring log.
    _ratio : float
        Calibration ratio (ignored when min_efficiency is set explicitly).
    _min_efficiency : float or None
        Energy efficiency threshold (ΔAccuracy/J). Auto-calibrated if None.
    _patience : int
        Consecutive inefficient epochs before stopping.
    _bad_epochs : int
        Counter of recent inefficient epochs.
    _prev_accuracy : float or None
        Previous epoch's accuracy (for delta computation).

    Examples
    --------
    >>> ees = EnergyEarlyStopping(
    ...     log_file_path=mon.monitor.log_file_path,
    ...     min_efficiency_ratio=0.05,
    ...     patience=3,
    ... )
    >>> for epoch in range(num_epochs):
    ...     mon.epoch_start(epoch)
    ...     loss, acc = train_one_epoch(...)
    ...     if mon.epoch_end(epoch, samples=N, loss=loss, accuracy=acc):
    ...         break
    """

    def __init__(
        self,
        log_file_path: str,
        min_efficiency_ratio: float = 0.05,
        min_efficiency: Optional[float] = None,
        patience: int = 3,
    ) -> None:
        """Initialize the early stopping controller.

        Parameters
        ----------
        log_file_path : str
            Path to the NDJSON monitoring log file (same as EpochTracker).
        min_efficiency_ratio : float, optional
            Calibration ratio. Used only when min_efficiency is None.
            threshold = ratio × first_epoch_efficiency. Default is 0.05.
        min_efficiency : float, optional
            Absolute efficiency threshold (ΔAccuracy/J). When set, disables
            auto-calibration. Default is None (auto-calibrate).
        patience : int, optional
            Consecutive inefficient epochs before stopping. Default is 3.
        """
        self._log_file_path = log_file_path
        self._ratio = min_efficiency_ratio
        self._min_efficiency = min_efficiency
        self._patience = patience
        self._bad_epochs: int = 0
        self._prev_accuracy: Optional[float] = None

    def step(self, epoch: int, accuracy: float) -> bool:
        """Evaluate early stopping criterion after an epoch.

        Must be called after EpochTracker.on_epoch_end() so the energy
        metrics have been computed.

        Prints status line with efficiency and threshold comparisons.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index (must match EpochTracker).
        accuracy : float
            Training accuracy for this epoch (0-1 or 0-100 scale, must be
            consistent throughout training).

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """
        df = compute_energy_metrics(self._log_file_path)

        if df.empty or epoch not in df["epoch"].values:
            self._prev_accuracy = accuracy
            return False

        energy_j = float(df.loc[df["epoch"] == epoch, "total_energy_j"].iloc[0])

        if self._prev_accuracy is None or energy_j <= 0:
            self._prev_accuracy = accuracy
            return False

        delta_acc  = accuracy - self._prev_accuracy
        self._prev_accuracy = accuracy
        efficiency = delta_acc / energy_j

        # Auto-calibrate threshold from the first observed efficiency
        if self._min_efficiency is None:
            if efficiency > 0:
                self._min_efficiency = self._ratio * efficiency
                print(
                    f"[EnergyEarlyStopping] threshold auto-set = {self._min_efficiency:.3e}  "
                    f"({self._ratio:.0%} of epoch-{epoch} efficiency {efficiency:.3e})"
                )
            else:
                return False

        inefficient = efficiency < self._min_efficiency

        print(
            f"[EnergyEarlyStopping] epoch={epoch:3d}  "
            f"Δacc={delta_acc:+.4f}  energy={energy_j:.0f}J  "
            f"eff={efficiency:.3e}  thr={self._min_efficiency:.3e}  "
            f"{'!! INEFF' if inefficient else 'ok':8s}  "
            f"patience={self._bad_epochs + (1 if inefficient else 0)}/{self._patience}"
        )

        if inefficient:
            self._bad_epochs += 1
            if self._bad_epochs >= self._patience:
                print(f"[EnergyEarlyStopping] Training stopped at epoch {epoch}.")
                return True
        else:
            self._bad_epochs = 0

        return False
