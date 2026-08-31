# moniaenergy/facade.py #
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

"""
facade.py — High-level ``monitor_train`` context manager.

Provides a single-import, single-``with`` entry point that activates the
full monitoring stack (hardware metrics, epoch tracking, energy grading,
optimizer advisor) without requiring the user to manage individual components.

Quick-start
-----------
>>> from moniaenergy import monitor_train
>>>
>>> with monitor_train(model, "my_run", country="Spain", batch_size=128) as mon:
...     for epoch in range(30):
...         mon.epoch_start(epoch)
...         loss, acc = train_one_epoch(loader, model, optimizer)
...         mon.epoch_end(epoch, samples=50000, loss=loss, accuracy=acc)

That is all.  At the end of the run the terminal prints a colour-coded
energy grade, and a full NDJSON + CSV report is saved automatically.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from .monitor.inline_monitor import MonitorContext
from .monitor.pytorch_hooks import (
    EpochTracker,
    EnergyEarlyStopping,
    compute_energy_metrics,
)
from .metrics.green_grader import calibrate_reference
from .monitor.optimizer_advisor import OptimizerAdvisor
from .monitor.gpu_power_optimizer import GpuPowerOptimizer

if TYPE_CHECKING:
    import torch.nn


# ---------------------------------------------------------------------------
# Grade → ANSI colour mapping for terminal pretty-printing
# ---------------------------------------------------------------------------
_GRADE_COLOURS = {
    "A++": "\033[1;92m",  # bright green
    "A+":  "\033[92m",
    "A":   "\033[32m",
    "B":   "\033[36m",    # cyan
    "C":   "\033[33m",    # yellow
    "D":   "\033[33m",
    "E":   "\033[31m",    # red
    "F":   "\033[1;31m",  # bright red
}
_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Session handle returned to the caller inside the ``with`` block
# ---------------------------------------------------------------------------

class MonitorSession:
    """Thin handle exposing epoch lifecycle methods to the user during training.

    Provides a simple interface for notifying the monitoring system about epoch
    boundaries, accepting metrics (loss, accuracy), and checking for early
    stopping criteria or active power optimization.

    Attributes
    ----------
    _tracker : EpochTracker
        Writes epoch boundary events to the NDJSON log.
    _log_path : str
        Path to the main monitoring NDJSON file.
    _ees : EnergyEarlyStopping or None
        Early stopping controller, or None if disabled.
    _advisor : OptimizerAdvisor
        Emits per-epoch energy optimization suggestions.
    _country : str
        Country for carbon intensity lookup.
    _parameters : int or None
        Number of model parameters (used in grade computation).
    _power_optimizer : GpuPowerOptimizer or None
        JIT power optimizer, or None if disabled.
    _last_grade : str or None
        Most recently computed energy grade.
    _epoch_start_time : float or None
        Monotonic timestamp of epoch start (used for epoch time measurement).
    """

    def __init__(
        self,
        tracker: EpochTracker,
        log_file_path: str,
        ees: Optional[EnergyEarlyStopping],
        advisor: OptimizerAdvisor,
        country: str,
        parameters: Optional[int],
        power_optimizer: Optional[GpuPowerOptimizer] = None,
    ) -> None:
        """Initialize a monitor session.

        Parameters
        ----------
        tracker : EpochTracker
            Epoch event tracker.
        log_file_path : str
            Path to NDJSON monitoring log.
        ees : EnergyEarlyStopping or None
            Early stopping controller, or None.
        advisor : OptimizerAdvisor
            Energy optimization advisor.
        country : str
            Country name for carbon lookups.
        parameters : int or None
            Model parameter count.
        power_optimizer : GpuPowerOptimizer, optional
            GPU power optimizer, or None. Default is None.
        """
        self._tracker = tracker
        self._log_path = log_file_path
        self._ees = ees
        self._advisor = advisor
        self._country = country
        self._parameters = parameters
        self._power_optimizer = power_optimizer
        self._last_grade: Optional[str] = None
        self._epoch_start_time: Optional[float] = None

    def epoch_start(self, epoch: int) -> None:
        """Signal the start of a training epoch.

        Records the current timestamp and notifies the epoch tracker.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        """
        import time as _time
        self._epoch_start_time = _time.monotonic()
        self._tracker.on_epoch_start(epoch)

    def epoch_end(
        self,
        epoch: int,
        samples: int = 0,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
    ) -> bool:
        """Signal the end of a training epoch and compute metrics.

        Notifies the tracker, computes energy metrics, updates the energy grade,
        and runs optimizer logic (either JIT power tuning or advisor suggestions).

        Checks EnergyEarlyStopping if active and returns early-stopping verdict.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        samples : int, optional
            Number of training samples in this epoch. Default is 0.
        loss : float, optional
            Epoch loss value. Default is None.
        accuracy : float, optional
            Epoch accuracy value (0.0 to 1.0 or percentage). Default is None.

        Returns
        -------
        bool
            True if EnergyEarlyStopping decided to stop training,
            False otherwise or if EES is disabled.
        """
        self._tracker.on_epoch_end(epoch, samples=samples, loss=loss, accuracy=accuracy)

        # Compute energy metrics and update grade
        df = compute_energy_metrics(self._log_path)
        if not df.empty and epoch in df["epoch"].values:
            row = df[df["epoch"] == epoch].iloc[0]
            # Grade computation removed: will be handled in post-hoc analysis

            # Run JIT power optimizer or Advisor (mutually exclusive)
            if self._power_optimizer is not None:
                import time as _time
                epoch_time_s = (
                    _time.monotonic() - self._epoch_start_time
                    if self._epoch_start_time is not None
                    else 0.0
                )
                j_sample = float(row.get("energy_per_sample_j", 0.0))
                self._power_optimizer.on_epoch_end(
                    j_sample=j_sample,
                    epoch_time_s=epoch_time_s,
                    epoch=epoch,
                )
            else:
                # Run optimizer advisor only if not actively tuning power
                self._advisor.step(epoch, df)

        # EnergyEarlyStopping check
        if self._ees and accuracy is not None:
            return self._ees.step(epoch=epoch, accuracy=float(accuracy))
        return False

    @property
    def last_grade(self) -> Optional[str]:
        """Most recently computed energy grade.

        Returns
        -------
        str or None
            Grade string (e.g., "A++", "A", "C") or None if not yet computed.
        """
        return self._last_grade

    @property
    def current_power_cap_w(self) -> Optional[int]:
        """Currently active GPU power limit (watts) from the optimizer.

        Returns
        -------
        int or None
            Power cap in watts, or None if optimizer is inactive.
        """
        if self._power_optimizer is not None:
            return self._power_optimizer.current_cap_w
        return None


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@contextmanager
def monitor_train(
    model: "torch.nn.Module | None" = None,
    run_name: str = "monitor_run",
    *,
    country: str = "Spain",
    batch_size: int = 128,
    fp16: bool = False,
    interval: float = 1.0,
    log_dir: str = "logs",
    early_stopping: bool = False,
    min_efficiency_ratio: float = 0.05,
    patience: int = 3,
    parameters: Optional[int] = None,
    frequency_unit: str = "MHz",
    memory_unit: str = "MB",
    temperature_unit: str = "ºC",
    power_optimize: bool = False,
    gpu_index: int = 0,
    time_budget_pct: float = 0.10,
    min_efficiency: Optional[float] = None,
):
    """Full monitoring context for ML training — one-line integration.

    Parameters
    ----------
    model:
        PyTorch model (used only to count parameters when ``parameters``
        is not supplied).  Safe to pass as ``None``.
    run_name:
        Descriptive name used to build the log file path.
    country:
        Country name matching a row in ``2025_Country_Carbon_Emissions.csv``
        (e.g. ``"Spain"``, ``"United States"``, ``"China"``).  Falls back to
        ``"World Average"`` when not found.
    batch_size:
        Current training batch size — passed to OptimizerAdvisor for
        context-aware suggestions.
    fp16:
        Whether mixed-precision is active.  Informs the advisor.
    interval:
        Hardware sampling interval in seconds.
    log_dir:
        Root directory under which NDJSON logs are saved.
    early_stopping:
        Enable EnergyEarlyStopping.
    min_efficiency_ratio:
        Threshold ratio for EnergyEarlyStopping (see its docstring).
    patience:
        Patience for EnergyEarlyStopping.
    parameters:
        Number of model parameters.  Auto-detected when ``model`` is given.
    frequency_unit / memory_unit / temperature_unit:
        Units forwarded to MonitorContext.

    Yields
    ------
    MonitorSession
        Call ``mon.epoch_start(epoch)`` and ``mon.epoch_end(...)`` inside your
        training loop.
    """
    # --- Auto-detect parameter count ----------------------------------------
    # New params documented below in the extended docstring.
    if parameters is None and model is not None:
        try:
            parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except Exception:
            parameters = None

    # --- Build log path -------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_basename = os.path.join(run_dir, f"run_{timestamp}")
    log_ndjson = log_basename + ".ndjson"

    # --- Start hardware monitor -----------------------------------------------
    with MonitorContext(
        context=run_name,
        interval=interval,
        log_file_path=log_basename,
        frequency_unit=frequency_unit,
        memory_unit=memory_unit,
        temperature_unit=temperature_unit,
    ) as mon:
        tracker = EpochTracker(mon.monitor.log_file_path)
        ees = EnergyEarlyStopping(
            log_file_path=mon.monitor.log_file_path,
            min_efficiency_ratio=min_efficiency_ratio,
            min_efficiency=min_efficiency,
            patience=patience,
        ) if early_stopping else None
        advisor = OptimizerAdvisor(
            batch_size=batch_size,
            fp16=fp16,
            early_stopping_active=early_stopping,
        )
        power_opt = (
            GpuPowerOptimizer(
                gpu_index=gpu_index,
                time_budget_pct=time_budget_pct,
            )
            if power_optimize
            else None
        )
        session = MonitorSession(
            tracker=tracker,
            log_file_path=mon.monitor.log_file_path,
            ees=ees,
            advisor=advisor,
            country=country,
            parameters=parameters,
            power_optimizer=power_opt,
        )
        try:
            yield session
        finally:
            if power_opt is not None:
                power_opt.restore()
            _print_final_report(session, log_ndjson, run_name, parameters)


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def _print_final_report(
    session: MonitorSession,
    log_ndjson: str,
    run_name: str,
    parameters: Optional[int],
) -> None:
    """Print a colour-coded energy grade summary at the end of training.

    Reads the accumulated energy metrics from the log file, computes mean
    efficiency, computes the final energy grade, and prints a formatted
    summary block with colour-coded grade, energy per sample, total energy,
    and estimated CO₂ emissions.

    Parameters
    ----------
    session : MonitorSession
        The completed monitoring session (used to retrieve last_grade).
    log_ndjson : str
        Path to the main NDJSON monitoring log file.
    run_name : str
        Name of the training run (for display).
    parameters : int or None
        Number of model parameters, or None if unknown.
    """
    df = compute_energy_metrics(log_ndjson)
    if df.empty:
        return

    # Use the mean efficiency across epochs (excluding epoch 0 warmup)
    valid = df[df["epoch"] > 0] if len(df) > 1 else df
    mean_energy_j = valid["total_energy_j"].mean()
    mean_acc = valid["accuracy"].dropna().mean() if "accuracy" in valid.columns else None
    mean_eps = valid["energy_per_sample_j"].dropna().mean() if "energy_per_sample_j" in valid.columns else None

    # La calificación se calcula en el pipeline de análisis post-hoc, que dispone
    # de la referencia calibrada sobre el conjunto completo de ejecuciones.
    grade_str = "N/A (Post-hoc)"

    sep = "═" * 66
    print(f"\n{sep}")
    print(f"  monitor_train — Final Report: {run_name}")
    print(sep)
    print(f"  Energy Grade:             {grade_str}")
    if mean_eps is not None:
        print(f"  J/sample (mean):          {mean_eps:.4f}")
    if mean_energy_j:
        print(f"  Total energy (mean/ep):   {mean_energy_j / 1000:.2f} kJ")
    co2_cols = [c for c in df.columns if c.startswith("co2_") and c.endswith("_g")]
    for col in co2_cols:
        total_co2 = df[col].sum()
        label = col[4:-2].replace("_", " ").title()
        print(f"  CO₂ estimated ({label:<15}): {total_co2:.2f} g")
    print(f"  Log saved to:             {log_ndjson}")
    print(f"{sep}\n")
