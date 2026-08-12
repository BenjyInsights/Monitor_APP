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
tests/test_gpu_power_optimizer.py — Unit tests for GpuPowerOptimizer Pareto logic.

Tests the Pareto selection algorithm, exploration/exploitation phases, and
advisor-only mode without requiring actual GPU hardware (pynvml is mocked).
"""

import unittest
from unittest.mock import patch, MagicMock

from monitor_app.monitor.gpu_power_optimizer import GpuPowerOptimizer, _ProbeResult


class TestParetoSelection(unittest.TestCase):
    """Test the internal _select_exploit_cap() Pareto selection logic."""

    def _make_optimizer(self) -> GpuPowerOptimizer:
        """Create a GpuPowerOptimizer in a testable state (no NVML)."""
        with patch("monitor_app.monitor.gpu_power_optimizer._NVML_AVAILABLE", False):
            opt = GpuPowerOptimizer(gpu_index=0)
        opt.mode = "advisor-only"
        opt.power_caps_w = [180, 210, 240, 270, 300]
        opt.max_cap_w = 300
        opt.min_cap_w = 180
        opt._original_cap_w = 300
        opt._handle = MagicMock()
        return opt

    def test_selects_lowest_energy_within_time_budget(self) -> None:
        """Pareto should select the cap with lowest J/sample within time budget."""
        opt = self._make_optimizer()
        opt._time_budget_pct = 0.10  # 10%

        opt._probes = [
            _ProbeResult(cap_w=180, j_sample=0.040, epoch_time_s=55.0, epoch=0),
            _ProbeResult(cap_w=210, j_sample=0.035, epoch_time_s=52.0, epoch=1),
            _ProbeResult(cap_w=240, j_sample=0.038, epoch_time_s=50.5, epoch=2),
            _ProbeResult(cap_w=270, j_sample=0.042, epoch_time_s=50.2, epoch=3),
            _ProbeResult(cap_w=300, j_sample=0.045, epoch_time_s=50.0, epoch=4),
        ]

        opt._select_exploit_cap()

        # Baseline = 300W (50.0s). Budget = 50.0 * 1.10 = 55.0s.
        # Eligible: all (cap 180 at 55.0 is exactly on the boundary).
        # Best J/sample among eligible: 210W at 0.035.
        self.assertEqual(opt._exploit_cap_w, 210)

    def test_baseline_is_highest_cap(self) -> None:
        """Baseline should always be the probe with the highest power cap."""
        opt = self._make_optimizer()
        opt._probes = [
            _ProbeResult(cap_w=180, j_sample=0.040, epoch_time_s=55.0, epoch=0),
            _ProbeResult(cap_w=300, j_sample=0.050, epoch_time_s=48.0, epoch=1),
            _ProbeResult(cap_w=240, j_sample=0.038, epoch_time_s=50.0, epoch=2),
        ]

        opt._select_exploit_cap()

        # Baseline should be 300W (highest cap), even if it wasn't the last probe.
        # Budget = 48.0 * 1.10 = 52.8s
        # Eligible: cap 300 (48.0s), cap 240 (50.0s). Cap 180 (55.0s) is over budget.
        # Best J/sample: cap 240 (0.038)
        self.assertEqual(opt._exploit_cap_w, 240)

    def test_time_budget_filters_slow_probes(self) -> None:
        """Probes exceeding time budget should be excluded from selection."""
        opt = self._make_optimizer()
        opt._time_budget_pct = 0.05  # Very tight 5% budget

        opt._probes = [
            _ProbeResult(cap_w=180, j_sample=0.025, epoch_time_s=70.0, epoch=0),  # Best J but too slow
            _ProbeResult(cap_w=240, j_sample=0.038, epoch_time_s=52.0, epoch=1),  # Within budget
            _ProbeResult(cap_w=300, j_sample=0.045, epoch_time_s=50.0, epoch=2),  # Baseline
        ]

        opt._select_exploit_cap()

        # Budget = 50.0 * 1.05 = 52.5s
        # Eligible: cap 240 (52.0s ≤ 52.5), cap 300 (50.0 ≤ 52.5). Cap 180 excluded.
        # Best J/sample among eligible: cap 240 (0.038)
        self.assertEqual(opt._exploit_cap_w, 240)

    def test_keeps_baseline_when_no_alternative_fits_budget(self) -> None:
        """A restrictive budget must never be traded away for lower energy.

        The baseline (highest cap) always satisfies its own time budget, so the
        eligible set is never empty. When every lower cap is too slow, the
        optimizer must stay on the baseline even though a lower cap would use
        less energy per sample — that is the point of the time budget.
        """
        opt = self._make_optimizer()
        opt._time_budget_pct = 0.01  # 1% — very restrictive

        opt._probes = [
            _ProbeResult(cap_w=180, j_sample=0.025, epoch_time_s=60.0, epoch=0),
            _ProbeResult(cap_w=240, j_sample=0.035, epoch_time_s=55.0, epoch=1),
            _ProbeResult(cap_w=300, j_sample=0.045, epoch_time_s=50.0, epoch=2),
        ]

        opt._select_exploit_cap()

        # Budget = 50.0 * 1.01 = 50.5s. Only the 300 W baseline fits, so despite
        # 180 W having the lowest J/sample the selection stays at 300 W.
        self.assertEqual(opt._exploit_cap_w, 300)

    def test_selected_cap_always_respects_time_budget(self) -> None:
        """The selected cap must never exceed the time budget, for any budget."""
        probes = [
            _ProbeResult(cap_w=180, j_sample=0.025, epoch_time_s=60.0, epoch=0),
            _ProbeResult(cap_w=240, j_sample=0.035, epoch_time_s=55.0, epoch=1),
            _ProbeResult(cap_w=300, j_sample=0.045, epoch_time_s=50.0, epoch=2),
        ]
        by_cap = {p.cap_w: p for p in probes}

        for budget_pct in (0.0, 0.01, 0.05, 0.10, 0.20, 0.50):
            opt = self._make_optimizer()
            opt._time_budget_pct = budget_pct
            opt._probes = list(probes)

            opt._select_exploit_cap()

            baseline_time = by_cap[300].epoch_time_s
            selected_time = by_cap[opt._exploit_cap_w].epoch_time_s
            self.assertLessEqual(
                selected_time,
                baseline_time * (1 + budget_pct),
                msg=f"budget {budget_pct:.0%} violated by cap {opt._exploit_cap_w} W",
            )

    def test_empty_probes_no_crash(self) -> None:
        """No probes should result in no selection (no crash)."""
        opt = self._make_optimizer()
        opt._probes = []
        opt._select_exploit_cap()
        self.assertIsNone(opt._exploit_cap_w)

    def test_single_probe(self) -> None:
        """Single probe should select that probe as exploit cap."""
        opt = self._make_optimizer()
        opt._probes = [
            _ProbeResult(cap_w=240, j_sample=0.038, epoch_time_s=51.0, epoch=0),
        ]
        opt._select_exploit_cap()
        self.assertEqual(opt._exploit_cap_w, 240)


class TestOptimizerModes(unittest.TestCase):
    """Test the operational modes of GpuPowerOptimizer."""

    def test_unavailable_without_nvml(self) -> None:
        """Without pynvml, optimizer should be in 'unavailable' mode."""
        with patch("monitor_app.monitor.gpu_power_optimizer._NVML_AVAILABLE", False):
            opt = GpuPowerOptimizer(gpu_index=0)
        self.assertEqual(opt.mode, "unavailable")

    def test_on_epoch_end_returns_false_when_unavailable(self) -> None:
        """on_epoch_end should return False when mode is unavailable."""
        with patch("monitor_app.monitor.gpu_power_optimizer._NVML_AVAILABLE", False):
            opt = GpuPowerOptimizer(gpu_index=0)
        result = opt.on_epoch_end(j_sample=0.05, epoch_time_s=50.0, epoch=0)
        self.assertFalse(result)

    def test_restore_does_not_crash_without_handle(self) -> None:
        """restore() should be safe to call when no NVML handle exists."""
        with patch("monitor_app.monitor.gpu_power_optimizer._NVML_AVAILABLE", False):
            opt = GpuPowerOptimizer(gpu_index=0)
        opt.restore()  # Should not raise


if __name__ == "__main__":
    unittest.main()
