# moniaenergy/monitor/optimizer_advisor.py #
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
optimizer_advisor.py — Active energy-efficiency advisor for training runs.

Analyses the per-epoch energy metrics accumulated by EpochTracker /
compute_energy_metrics() and emits actionable, context-aware suggestions
to the terminal whenever a significant improvement is possible (>10 % gain).

Suggestions are printed in a clearly formatted block, annotated with
estimated impact, and include a before/after grade forecast when a
GradeResult is available.

Typical output
--------------
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[MonitorAdvisor] epoch=8  eff_trend=↓  J/sample=0.071  grade=C
  💡 SUGGESTION: current batch_size (32) is suboptimal for this GPU.
     ↗ Raising it to batch=128 would cut J/sample by ~40 % (own benchmarks).
  💡 SUGGESTION: FP32 active — try --fp16 to cut energy use by ~13 %.
  💡 SUGGESTION: efficiency dropping >20 % over the last 3 epochs.
     → Activa EnergyEarlyStopping con --early-stopping (patience=3).
[MonitorAdvisor] Potencial: C → A+  (3 mejoras disponibles)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

# Minimum improvement threshold to emit a suggestion (10 %)
_MIN_IMPROVEMENT_FRACTION = 0.10
# Minimum number of epochs before trend analysis is possible
_MIN_EPOCHS_FOR_TREND = 3
# How many consecutive epochs of declining efficiency trigger the EES hint
_EES_PATIENCE_TRIGGER = 3

# Known benchmark J/sample values derived from run_benchmarks.sh data
# Keys: (batch_size, precision).  Values: mean J/sample from benchmarks.
_BENCHMARK_J_SAMPLE: dict[tuple, float] = {
    (32,  "fp32"): 0.0646,
    (64,  "fp32"): 0.0679,
    (128, "fp32"): 0.0406,
    (256, "fp32"): 0.0316,
    (32,  "fp16"): 0.0352,  # ResNet18 reference
    (64,  "fp16"): 0.0526,
    (128, "fp16"): 0.0352,
    (256, "fp16"): 0.0259,
}


class OptimizerAdvisor:
    """Emits per-epoch energy optimisation suggestions to stdout.

    Usage
    -----
    advisor = OptimizerAdvisor(
        batch_size=32,
        fp16=False,
        early_stopping_active=False,
    )
    for epoch in range(num_epochs):
        tracker.on_epoch_start(epoch)
        train_one_epoch(...)
        tracker.on_epoch_end(epoch, samples=N, ...)
        energy_df = compute_energy_metrics(log_path)
        advisor.step(epoch, energy_df)
    """

    def __init__(
        self,
        batch_size: int = 128,
        fp16: bool = False,
        early_stopping_active: bool = False,
        current_grade: Optional[str] = None,
        min_improvement: float = _MIN_IMPROVEMENT_FRACTION,
    ):
        """
        Parameters
        ----------
        batch_size:
            The current training batch size.
        fp16:
            Whether mixed precision (FP16/AMP) is already active.
        early_stopping_active:
            Whether EnergyEarlyStopping is already enabled.
        current_grade:
            Current energy grade string (e.g. "C").  Used in the summary line.
        min_improvement:
            Minimum relative improvement required to emit a suggestion.
        """
        self._batch_size = batch_size
        self._fp16 = fp16
        self._ees_active = early_stopping_active
        self._current_grade = current_grade
        self._min_improvement = min_improvement
        self._prev_eff: list[float] = []  # history of J/sample per epoch
        self._suggestions_emitted: set[str] = set()  # avoid repeat suggestions

    def step(self, epoch: int, energy_df: pd.DataFrame) -> None:
        """Evaluate the epoch and print suggestions if warranted.

        Parameters
        ----------
        epoch:
            Zero-based epoch index.
        energy_df:
            DataFrame returned by ``compute_energy_metrics()``.
            May include optional columns: sm_utilization_pct (GPU SM occupancy).
        """
        if energy_df is None or energy_df.empty:
            return

        row = energy_df[energy_df["epoch"] == epoch]
        if row.empty:
            return

        j_sample = row["energy_per_sample_j"].iloc[0]
        if j_sample is None or pd.isna(j_sample):
            return

        self._prev_eff.append(float(j_sample))

        suggestions: list[str] = []
        potential_grade: Optional[str] = None

        # -- Suggestion 1: batch_size suboptimal (based on SM utilization) ----
        best_bs, savings_bs, sm_util = self._check_batch_size(j_sample, row)
        if best_bs and savings_bs >= self._min_improvement and "batch" not in self._suggestions_emitted:
            sm_note = ""
            if sm_util is not None and sm_util < 75.0:
                sm_note = f"(SM utilization: {sm_util:.0f}%, target ~80-90%)"
            suggestions.append(
                f"  💡 SUGGESTION: current batch_size ({self._batch_size}) is suboptimal for this GPU.\n"
                f"     ↗ Raising it to batch={best_bs} would cut J/sample by ~{savings_bs * 100:.0f}% "
                f"(own benchmarks) {sm_note}."
            )
            self._suggestions_emitted.add("batch")

        # -- Suggestion 2: FP16 not active -------------------------------------
        savings_fp16 = self._check_fp16(j_sample)
        if savings_fp16 >= self._min_improvement and not self._fp16 and "fp16" not in self._suggestions_emitted:
            suggestions.append(
                f"  💡 SUGGESTION: FP32 active — try --fp16 to cut energy use by "
                f"~{savings_fp16 * 100:.0f}%."
            )
            self._suggestions_emitted.add("fp16")

        # -- Suggestion 3: Declining efficiency trend → suggest EES -----------
        if (
            len(self._prev_eff) >= _MIN_EPOCHS_FOR_TREND
            and not self._ees_active
            and "ees" not in self._suggestions_emitted
        ):
            recent = self._prev_eff[-_EES_PATIENCE_TRIGGER:]
            if all(recent[i] > recent[i - 1] * (1 + self._min_improvement)
                   for i in range(1, len(recent))):
                suggestions.append(
                    f"  💡 SUGGESTION: efficiency dropping >20% over the last "
                    f"{_EES_PATIENCE_TRIGGER} epochs.\n"
                    f"     → Enable EnergyEarlyStopping with --early-stopping "
                    f"(patience={_EES_PATIENCE_TRIGGER})."
                )
                self._suggestions_emitted.add("ees")

        if not suggestions:
            return

        # -- Determine trend symbol --------------------------------------------
        if len(self._prev_eff) >= 2:
            trend = "↓" if self._prev_eff[-1] > self._prev_eff[-2] else "↑"
        else:
            trend = "~"

        # -- Estimate potential grade ------------------------------------------
        potential_grade = self._estimate_potential_grade(j_sample, suggestions)

        # -- Print block -------------------------------------------------------
        sep = "━" * 66
        print(f"\n{sep}")
        print(
            f"[MonitorAdvisor] epoch={epoch}  "
            f"eff_trend={trend}  "
            f"J/sample={j_sample:.4f}  "
            f"grade={self._current_grade or '?'}"
        )
        for s in suggestions:
            print(s)
        summary = (
            f"[MonitorAdvisor] Potencial: {self._current_grade} → {potential_grade}  "
            f"({len(suggestions)} improvement{'s' if len(suggestions) != 1 else ''} available)"
        ) if potential_grade else (
            f"[MonitorAdvisor] {len(suggestions)} suggestion(s) to improve efficiency."
        )
        print(summary)
        print(f"{sep}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_batch_size(
        self, current_j_sample: float, row: "pd.Series | None" = None
    ) -> tuple[Optional[int], float, Optional[float]]:
        """Return the best alternative batch size, its relative saving, and observed SM util.

        Parameters
        ----------
        current_j_sample : float
            Current epoch's energy per sample (J).
        row : pd.Series, optional
            DataFrame row for this epoch. May contain 'sm_utilization_pct' column.

        Returns
        -------
        tuple(int or None, float, float or None)
            (best_batch_size, relative_saving, observed_sm_utilization_pct)
        """
        prec = "fp16" if self._fp16 else "fp32"
        current_ref = _BENCHMARK_J_SAMPLE.get((self._batch_size, prec), current_j_sample)
        best_bs = None
        best_saving = 0.0

        # Extract SM utilization if available
        sm_util: Optional[float] = None
        if row is not None:
            try:
                sm_util = float(row.get("sm_utilization_pct"))
            except (TypeError, ValueError, AttributeError):
                sm_util = None

        for (bs, p), ref_j in _BENCHMARK_J_SAMPLE.items():
            if p != prec or bs <= self._batch_size:
                continue
            saving = (current_ref - ref_j) / current_ref if current_ref > 0 else 0
            if saving > best_saving:
                best_saving = saving
                best_bs = bs

        return best_bs, best_saving, sm_util

    def _check_fp16(self, current_j_sample: float) -> float:
        """Return the estimated relative energy saving from enabling FP16."""
        prec_fp32 = "fp32"
        prec_fp16 = "fp16"
        ref_fp32 = _BENCHMARK_J_SAMPLE.get((self._batch_size, prec_fp32))
        ref_fp16 = _BENCHMARK_J_SAMPLE.get((self._batch_size, prec_fp16))
        if ref_fp32 and ref_fp16 and ref_fp32 > 0:
            return max(0.0, (ref_fp32 - ref_fp16) / ref_fp32)
        return 0.0

    def _estimate_potential_grade(
        self, current_j_sample: float, suggestions: list[str]
    ) -> Optional[str]:
        """Rough estimate of achievable grade after applying all suggestions."""
        grade_map = {"A++": 0, "A+": 1, "A": 2, "B": 3, "C": 4, "D": 5, "E": 6, "F": 7}
        rev_map = {v: k for k, v in grade_map.items()}
        if self._current_grade not in grade_map:
            return None
        current_idx = grade_map[self._current_grade]
        steps = len(suggestions)
        potential_idx = max(0, current_idx - steps)
        return rev_map.get(potential_idx, "A++")
