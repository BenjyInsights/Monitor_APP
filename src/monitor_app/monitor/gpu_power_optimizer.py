# monitor/gpu_power_optimizer.py #
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
gpu_power_optimizer.py — JIT GPU Power Optimizer (Zeus-style).

Automatically adjusts the GPU power cap (W) at the end of each epoch
using a two-phase Explore → Exploit strategy inspired by the Zeus framework
(ML.Energy Initiative, University of Michigan).

Key differences vs Zeus
-----------------------
- **Human-readable justifications**: Every decision is printed in plain
  language explaining *why* the cap changed and what J/sample improvement
  was achieved.
- **White-box Pareto**: Instead of a black-box bandit, we expose the Pareto
  evaluation as a printable table so the developer understands the tradeoff.
- **Graceful degradation**: Without root privileges,
  ``nvmlDeviceSetPowerManagementLimit`` raises ``NVMLError_NoPermission``.
  The optimizer automatically falls back to *advisor-only* mode: it prints the
  same analysis but does not touch hardware.
- **Signal-safe restore**: Registers SIGTERM/SIGINT handlers to restore the
  original power limit even on abrupt termination.

Algorithm
---------
**Phase 1 — Exploration** (first ``len(power_caps_w)`` epochs):
    Each epoch probes a different power cap from the candidate list, records
    (cap_w, j_sample, epoch_time_s) and restores nothing.

**Phase 2 — Exploitation** (remaining epochs):
    Selects the Pareto-optimal cap: the lowest cap_w such that
    ``epoch_time_s <= best_time * (1 + time_budget_pct)``.  Applies it once
    and holds it for the rest of training.

Usage
-----
>>> with monitor_train(model, "run", power_optimize=True) as mon:
...     for epoch in range(30):
...         mon.epoch_start(epoch)
...         loss, acc = train_one_epoch(...)
...         if mon.epoch_end(epoch, samples=N, loss=loss, accuracy=acc):
...             break
"""

from __future__ import annotations

import signal
import atexit
import math
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal data containers
# ---------------------------------------------------------------------------

@dataclass
class _ProbeResult:
    """Stores the outcome of probing a single power cap."""
    cap_w: int
    j_sample: float
    epoch_time_s: float
    epoch: int


# ---------------------------------------------------------------------------
# Colour helpers (ANSI — no extra dependency)
# ---------------------------------------------------------------------------

_G  = "\033[92m"   # green
_Y  = "\033[93m"   # yellow
_R  = "\033[91m"   # red
_C  = "\033[96m"   # cyan
_B  = "\033[1m"    # bold
_RS = "\033[0m"    # reset


def _bar(ratio: float, width: int = 20) -> str:
    """ASCII progress bar from 0.0 to 1.0."""
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GpuPowerOptimizer:
    """JIT GPU power cap optimizer with human-readable decision justifications.

    Parameters
    ----------
    gpu_index:
        NVML device index (0 = first GPU).
    time_budget_pct:
        Maximum tolerated slowdown when the cap is reduced.
        E.g. ``0.10`` means "allow up to 10 % longer epochs for energy savings".
    exploration_factor:
        Fraction of the power range to probe (generates candidate caps as
        ``max_cap * f`` for f in linspace from ``min_frac`` to 1.0).
    min_frac:
        Minimum power cap as a fraction of the hardware max (default: 0.60).
    verbose:
        Print detailed Pareto table at the end of exploration.
    """

    #: Separator for terminal blocks
    _SEP = "━" * 68

    def __init__(
        self,
        gpu_index: int = 0,
        time_budget_pct: float = 0.10,
        min_frac: float = 0.60,
        n_steps: int = 5,
        verbose: bool = True,
    ) -> None:
        self._gpu_index = gpu_index
        self._time_budget_pct = time_budget_pct
        self._verbose = verbose
        self._device_name = f"GPU:{gpu_index}"

        # Operational mode: "active" | "advisor-only" | "unavailable"
        self.mode: str = "unavailable"

        self._handle = None
        self._original_cap_w: Optional[int] = None
        self.min_cap_w: int = 0
        self.max_cap_w: int = 0
        self.power_caps_w: List[int] = []

        self._probes: List[_ProbeResult] = []
        self._exploit_cap_w: Optional[int] = None   # chosen cap after exploration
        self._exploit_applied: bool = False

        if not _NVML_AVAILABLE:
            print(
                f"\n{_Y}[PowerOptimizer] pynvml not available — mode disabled.{_RS}"
            )
            return

        try:
            # pynvml may already be initialized by NVIDIAGPUMetrics; calling
            # nvmlInit() again is a no-op if already initialized.
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

            min_mw, max_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(
                self._handle
            )
            cur_mw = pynvml.nvmlDeviceGetPowerManagementLimit(self._handle)

            self.min_cap_w = int(min_mw / 1000)
            self.max_cap_w = int(max_mw / 1000)
            self._original_cap_w = int(cur_mw / 1000)

            # Build candidate list from min_frac*max to max in n_steps
            step = max(1, int((self.max_cap_w - int(self.max_cap_w * min_frac)) / (n_steps - 1)))
            caps = sorted(set(
                min(self.max_cap_w, int(self.max_cap_w * min_frac) + i * step)
                for i in range(n_steps)
            ))
            # Always include the full cap so we have a baseline
            if self.max_cap_w not in caps:
                caps.append(self.max_cap_w)
            self.power_caps_w = caps

            # Try setting the limit at the base value to check permissions
            try:
                pynvml.nvmlDeviceSetPowerManagementLimit(
                    self._handle, cur_mw  # restore immediately — just checking
                )
                self.mode = "active"
            except Exception:
                self.mode = "advisor-only"

            self._register_signal_handlers()
            self._print_startup_banner()

        except Exception as exc:
            print(
                f"\n{_Y}[PowerOptimizer] NVML not available ({exc}) "
                f"— mode disabled.{_RS}"
            )

    @property
    def current_cap_w(self) -> int:
        """Returns the currently active power cap (W)."""
        # During exploration, it's the probe cap.
        # During exploitation, it's the exploit cap.
        # Fallback to original.
        if self._probes and len(self._probes) <= len(self.power_caps_w):
            return self._probes[-1].cap_w
        return self._exploit_cap_w or self._original_cap_w or 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def on_epoch_end(
        self,
        j_sample: float,
        epoch_time_s: float,
        epoch: int,
    ) -> bool:
        """Evaluate epoch metrics and optionally change the GPU power cap.

        Parameters
        ----------
        j_sample:
            Energy per sample (J) for this epoch.
        epoch_time_s:
            Wall-clock duration of the epoch (seconds).
        epoch:
            Zero-based epoch index.

        Returns
        -------
        bool
            ``True`` if the power cap was changed.
        """
        if self._handle is None or self.mode == "unavailable":
            return False

        # ── Phase 1: Exploration ──────────────────────────────────────────
        probe_idx = epoch  # epoch 0 → cap[0], epoch 1 → cap[1], …
        if probe_idx < len(self.power_caps_w):
            cap_w = self.power_caps_w[probe_idx]
            self._probes.append(_ProbeResult(
                cap_w=cap_w,
                j_sample=j_sample,
                epoch_time_s=epoch_time_s,
                epoch=epoch,
            ))
            changed = self._apply_cap(cap_w)
            action = "probing" if changed else "suggested (no privileges)"
            self._print_exploration(epoch, probe_idx, cap_w, j_sample, epoch_time_s, action)

            # If last probe epoch → compute Pareto and announce decision
            if probe_idx == len(self.power_caps_w) - 1:
                self._select_exploit_cap()
            return changed

        # ── Phase 2: Exploitation ─────────────────────────────────────────
        if not self._exploit_applied and self._exploit_cap_w is not None:
            changed = self._apply_cap(self._exploit_cap_w)
            self._exploit_applied = True
            self._print_exploit_decision(self._exploit_cap_w)
            return changed

        return False

    def restore(self) -> None:
        """Restore the original power limit. Call from ``finally`` block."""
        if self._handle is None or self._original_cap_w is None:
            return
        self._apply_cap(self._original_cap_w, silent=True)
        print(
            f"\n{_C}[PowerOptimizer] Power limit restored to "
            f"{self._original_cap_w} W.{_RS}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_cap(self, cap_w: int, *, silent: bool = False) -> bool:
        """Set the GPU power limit. Returns True on success."""
        if self._handle is None:
            return False
        try:
            pynvml.nvmlDeviceSetPowerManagementLimit(
                self._handle, cap_w * 1000  # mW
            )
            return True
        except Exception as exc:
            if not silent:
                if "NoPermission" in type(exc).__name__ or "Insufficient" in str(exc):
                    if self.mode == "active":
                        self.mode = "advisor-only"
                        print(
                            f"\n{_Y}[PowerOptimizer] No root privileges — "
                            f"mode switched to: advisor-only.{_RS}\n"
                            f"  Hint: run with sudo for active control.\n"
                        )
            return False

    def _select_exploit_cap(self) -> None:
        """Compute Pareto-optimal cap from probe results."""
        if not self._probes:
            return

        # Baseline: cap with the highest power (full power = baseline time)
        baseline = max(self._probes, key=lambda p: p.cap_w)
        time_budget = baseline.epoch_time_s * (1 + self._time_budget_pct)

        # Eligible: within time budget.
        # El propio baseline satisface siempre t <= t*(1+delta) para delta >= 0, de
        # modo que este conjunto nunca queda vacío: el cap elegido jamás excede el
        # presupuesto temporal. La rama de relajación se conserva como guarda
        # defensiva por si en el futuro el baseline dejara de derivarse de _probes.
        eligible = [p for p in self._probes if p.epoch_time_s <= time_budget]
        if not eligible:  # pragma: no cover - inalcanzable con baseline ∈ _probes
            eligible = self._probes

        # Best: lowest J/sample among eligible
        best = min(eligible, key=lambda p: p.j_sample)
        self._exploit_cap_w = best.cap_w

        if self._verbose:
            self._print_pareto_table(baseline, best)

    def _print_startup_banner(self) -> None:
        mode_str = (
            f"{_G}ACTIVE (hardware control){_RS}"
            if self.mode == "active"
            else f"{_Y}ADVISOR-ONLY (no root){_RS}"
        )
        print(f"\n{self._SEP}")
        print(f"{_B}[PowerOptimizer] Started — GPU {self._gpu_index}{_RS}")
        print(f"  Mode:          {mode_str}")
        print(f"  Current limit: {self._original_cap_w} W")
        print(f"  HW range:      {self.min_cap_w} W – {self.max_cap_w} W")
        print(f"  Caps to probe: {self.power_caps_w} W")
        print(f"  Time budget:   ±{self._time_budget_pct*100:.0f}%")
        print(self._SEP + "\n")

    def _print_exploration(
        self,
        epoch: int,
        probe_idx: int,
        cap_w: int,
        j_sample: float,
        epoch_time_s: float,
        action: str,
    ) -> None:
        bar = _bar((probe_idx + 1) / len(self.power_caps_w))
        print(
            f"{_C}[PowerOptimizer]{_RS} epoch={epoch}  "
            f"Exploration [{bar}] {probe_idx+1}/{len(self.power_caps_w)}\n"
            f"  Cap {action}: {_B}{cap_w} W{_RS}  |  "
            f"J/sample={j_sample:.5f}  |  tiempo={epoch_time_s:.1f}s"
        )

    def _print_pareto_table(
        self, baseline: _ProbeResult, best: _ProbeResult
    ) -> None:
        print(f"\n{self._SEP}")
        print(f"{_B}[PowerOptimizer] Pareto analysis — exploration complete{_RS}")
        print(f"  {'Cap (W)':<10} {'J/sample':<12} {'Time (s)':<12} {'ΔEnergy':<12} {'ΔTime':<10} {'Viable'}")
        print("  " + "-" * 62)
        for p in sorted(self._probes, key=lambda x: x.cap_w):
            delta_e = (baseline.j_sample - p.j_sample) / baseline.j_sample * 100
            delta_t = (p.epoch_time_s - baseline.epoch_time_s) / baseline.epoch_time_s * 100
            viable_budget = baseline.epoch_time_s * (1 + self._time_budget_pct)
            viable = p.epoch_time_s <= viable_budget
            marker = f"{_G}✓{_RS}" if viable else f"{_R}✗{_RS}"
            best_mark = f" {_B}← OPTIMAL{_RS}" if p.cap_w == best.cap_w else ""
            e_col = f"{_G}" if delta_e > 0 else f"{_R}"
            t_col = f"{_G}" if delta_t <= 0 else (_Y if delta_t <= 10 else _R)
            print(
                f"  {p.cap_w:<10} {p.j_sample:<12.5f} {p.epoch_time_s:<12.1f} "
                f"{e_col}{delta_e:+.1f}%{_RS:<11} {t_col}{delta_t:+.1f}%{_RS:<9} {marker}{best_mark}"
            )
        savings_pct = (baseline.j_sample - best.j_sample) / baseline.j_sample * 100
        print(f"\n  {_G}→ Decision: cap={best.cap_w} W saves {savings_pct:.1f}% energy "
              f"within the time budget (±{self._time_budget_pct*100:.0f}%).{_RS}")
        print(self._SEP + "\n")

    def _print_exploit_decision(self, cap_w: int) -> None:
        baseline = max(self._probes, key=lambda p: p.cap_w)
        best = next((p for p in self._probes if p.cap_w == cap_w), None)
        if best is None:
            return
        savings_e = (baseline.j_sample - best.j_sample) / baseline.j_sample * 100
        delta_t   = (best.epoch_time_s - baseline.epoch_time_s) / baseline.epoch_time_s * 100
        action_str = (
            f"applied in hardware ({_G}active control{_RS})"
            if self.mode == "active"
            else f"suggested ({_Y}no root privileges{_RS})"
        )
        print(f"\n{self._SEP}")
        print(f"{_B}[PowerOptimizer] Exploitation started — optimal cap fixed{_RS}")
        print(f"  Cap {action_str}: {_B}{cap_w} W{_RS}  (from {baseline.cap_w} W)")
        print(f"  Energy saving:   {_G}{savings_e:.1f}%{_RS} in J/sample")
        print(f"  Time cost:       {delta_t:+.1f}% per epoch")
        print(f"  Rationale: cap={cap_w}W cut J/sample from {baseline.j_sample:.5f} "
              f"to {best.j_sample:.5f} for only {delta_t:+.1f}% extra time.")
        print(self._SEP + "\n")

    def _register_signal_handlers(self) -> None:
        """Register SIGINT/SIGTERM handlers AND atexit hook for restoration.

        Ensures the original power limit is restored even on abrupt termination:
        - Signal handlers (SIGINT, SIGTERM) call restore() immediately.
        - atexit hook provides a fallback for non-signal termination.

        Both mechanisms attempt restoration, so the power limit is guaranteed
        to be restored under any normal process exit condition.

        Gracefully skips registration if signals cannot be installed (e.g.,
        multi-threaded context where signals are unsafe).
        """
        import logging
        logger = logging.getLogger(__name__)

        # --- Signal handler ---
        def _signal_handler(signum: int, frame) -> None:  # noqa: ANN001
            """Restore power limit and exit cleanly on signal."""
            sig_name = signal.Signals(signum).name
            try:
                logger.info(
                    f"[{self._device_name}] Signal {sig_name} ({signum}) received; "
                    f"restoring original power limit ({self._original_cap_w}W)"
                )
                self.restore()
            except Exception as e:
                logger.error(
                    f"[{self._device_name}] Error during signal-triggered restore: {e}"
                )
            finally:
                raise SystemExit(0)

        # --- Atexit hook (fallback) ---
        def _atexit_handler() -> None:
            """Fallback restoration on normal process exit (non-signal)."""
            if self._original_cap_w is not None and self._handle is not None:
                try:
                    logger.debug(
                        f"[{self._device_name}] atexit hook: restoring "
                        f"power limit to {self._original_cap_w}W"
                    )
                    self._apply_cap(self._original_cap_w, silent=True)
                except Exception as e:
                    logger.warning(
                        f"[{self._device_name}] atexit restore failed: {e}"
                    )

        # Try to register SIGINT/SIGTERM handlers
        try:
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
            logger.debug(
                f"[{self._device_name}] Signal handlers (SIGINT/SIGTERM) registered"
            )
        except (RuntimeError, ValueError) as e:
            # Signals are unsafe in multi-threaded contexts
            logger.warning(
                f"[{self._device_name}] Cannot register signal handlers: {e}. "
                f"Relying on atexit hook for restoration."
            )

        # Always register atexit hook as a fallback
        try:
            atexit.register(_atexit_handler)
            logger.debug(
                f"[{self._device_name}] atexit hook registered for fallback restoration"
            )
        except Exception as e:
            logger.error(
                f"[{self._device_name}] Failed to register atexit hook: {e}"
            )
