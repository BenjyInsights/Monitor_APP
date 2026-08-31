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
tests/test_energy_early_stopping.py — Unit tests for EnergyEarlyStopping.

Tests the auto-calibration logic, patience counter, and stopping criterion
using mocked compute_energy_metrics to avoid needing real NDJSON log files.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from moniaenergy.monitor.pytorch_hooks import EnergyEarlyStopping


def _mock_energy_df(epoch: int, energy_j: float) -> pd.DataFrame:
    """Create a minimal energy DataFrame for a single epoch."""
    return pd.DataFrame([{
        "epoch": epoch,
        "total_energy_j": energy_j,
        "duration_s": 50.0,
        "samples": 50000,
        "energy_per_sample_j": energy_j / 50000,
    }])


class TestEnergyEarlyStopping(unittest.TestCase):
    """Test EnergyEarlyStopping controller."""

    def _make_ees(self, patience: int = 3, ratio: float = 0.05) -> EnergyEarlyStopping:
        """Create an EES instance with a dummy log path."""
        return EnergyEarlyStopping(
            log_file_path="/tmp/dummy.ndjson",
            min_efficiency_ratio=ratio,
            patience=patience,
        )

    @patch("moniaenergy.monitor.pytorch_hooks.compute_energy_metrics")
    def test_does_not_stop_first_epoch(self, mock_cem: MagicMock) -> None:
        """First epoch should never trigger stopping (no previous accuracy)."""
        ees = self._make_ees()
        mock_cem.return_value = _mock_energy_df(0, 5000.0)
        result = ees.step(epoch=0, accuracy=0.50)
        self.assertFalse(result)

    @patch("moniaenergy.monitor.pytorch_hooks.compute_energy_metrics")
    def test_does_not_stop_with_good_efficiency(self, mock_cem: MagicMock) -> None:
        """Efficient epochs should not trigger stopping."""
        ees = self._make_ees(patience=3)

        # Epoch 0: baseline (no previous accuracy)
        mock_cem.return_value = _mock_energy_df(0, 5000.0)
        ees.step(epoch=0, accuracy=0.50)

        # Epoch 1: good efficiency (large accuracy gain, moderate energy)
        mock_cem.return_value = _mock_energy_df(1, 5000.0)
        result = ees.step(epoch=1, accuracy=0.70)
        self.assertFalse(result)

    @patch("moniaenergy.monitor.pytorch_hooks.compute_energy_metrics")
    def test_auto_calibrates_threshold(self, mock_cem: MagicMock) -> None:
        """Threshold should be auto-calibrated from the first productive epoch."""
        ees = self._make_ees(ratio=0.05)
        self.assertIsNone(ees._min_efficiency)

        # Epoch 0: baseline
        mock_cem.return_value = _mock_energy_df(0, 5000.0)
        ees.step(epoch=0, accuracy=0.40)

        # Epoch 1: first productive epoch with positive delta_acc
        mock_cem.return_value = _mock_energy_df(1, 5000.0)
        ees.step(epoch=1, accuracy=0.60)

        # Threshold should now be set
        self.assertIsNotNone(ees._min_efficiency)
        expected_eff = (0.60 - 0.40) / 5000.0
        self.assertAlmostEqual(ees._min_efficiency, 0.05 * expected_eff, places=10)

    @patch("moniaenergy.monitor.pytorch_hooks.compute_energy_metrics")
    def test_stops_after_patience_exhausted(self, mock_cem: MagicMock) -> None:
        """Should stop after `patience` consecutive inefficient epochs."""
        ees = self._make_ees(patience=2)

        # Epoch 0: baseline
        mock_cem.return_value = _mock_energy_df(0, 5000.0)
        ees.step(epoch=0, accuracy=0.40)

        # Epoch 1: productive (auto-calibrates)
        mock_cem.return_value = _mock_energy_df(1, 5000.0)
        ees.step(epoch=1, accuracy=0.60)

        # Epoch 2: inefficient (tiny delta_acc)
        mock_cem.return_value = _mock_energy_df(2, 5000.0)
        result = ees.step(epoch=2, accuracy=0.6001)
        self.assertFalse(result)  # patience = 1/2

        # Epoch 3: still inefficient
        mock_cem.return_value = _mock_energy_df(3, 5000.0)
        result = ees.step(epoch=3, accuracy=0.6002)
        self.assertTrue(result)  # patience exhausted: 2/2

    @patch("moniaenergy.monitor.pytorch_hooks.compute_energy_metrics")
    def test_resets_patience_on_efficient_epoch(self, mock_cem: MagicMock) -> None:
        """An efficient epoch should reset the bad_epochs counter."""
        ees = self._make_ees(patience=3)

        # Epoch 0: baseline
        mock_cem.return_value = _mock_energy_df(0, 5000.0)
        ees.step(epoch=0, accuracy=0.40)

        # Epoch 1: productive (auto-calibrates)
        mock_cem.return_value = _mock_energy_df(1, 5000.0)
        ees.step(epoch=1, accuracy=0.60)

        # Epoch 2: inefficient (bad_epochs = 1)
        mock_cem.return_value = _mock_energy_df(2, 5000.0)
        ees.step(epoch=2, accuracy=0.6001)
        self.assertEqual(ees._bad_epochs, 1)

        # Epoch 3: efficient again (resets)
        mock_cem.return_value = _mock_energy_df(3, 5000.0)
        ees.step(epoch=3, accuracy=0.80)
        self.assertEqual(ees._bad_epochs, 0)

    @patch("moniaenergy.monitor.pytorch_hooks.compute_energy_metrics")
    def test_returns_false_on_empty_df(self, mock_cem: MagicMock) -> None:
        """Empty DataFrame from compute_energy_metrics should return False."""
        ees = self._make_ees()
        mock_cem.return_value = pd.DataFrame()
        result = ees.step(epoch=0, accuracy=0.50)
        self.assertFalse(result)

    @patch("moniaenergy.monitor.pytorch_hooks.compute_energy_metrics")
    def test_explicit_threshold_skips_calibration(self, mock_cem: MagicMock) -> None:
        """When min_efficiency is set explicitly, auto-calibration should be skipped."""
        ees = EnergyEarlyStopping(
            log_file_path="/tmp/dummy.ndjson",
            min_efficiency=1e-5,
            patience=2,
        )
        self.assertEqual(ees._min_efficiency, 1e-5)

        # Epoch 0: baseline
        mock_cem.return_value = _mock_energy_df(0, 5000.0)
        ees.step(epoch=0, accuracy=0.40)

        # Epoch 1: should still use the explicit threshold
        mock_cem.return_value = _mock_energy_df(1, 5000.0)
        ees.step(epoch=1, accuracy=0.60)

        # Threshold should not have changed
        self.assertEqual(ees._min_efficiency, 1e-5)


if __name__ == "__main__":
    unittest.main()
