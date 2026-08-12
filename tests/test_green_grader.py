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
tests/test_green_grader.py — Unit tests for Green AI Grade computation.

Tests compute_grade() and calibrate_reference() from the green_grader module,
covering grade assignment, edge cases, accuracy normalisation, and calibration.
"""

import unittest
import math

from monitor_app.metrics.green_grader import compute_grade, calibrate_reference


class TestComputeGrade(unittest.TestCase):
    """Test the compute_grade() function."""

    def test_valid_inputs_returns_grade_result(self) -> None:
        """A valid call should return a GradeResult, not None."""
        result = compute_grade(total_energy_j=1000.0, accuracy=0.90, parameters=11_000_000)
        self.assertIsNotNone(result)
        self.assertIn(result.grade, ["A++", "A+", "A", "B", "C", "D", "E", "F"])

    def test_zero_energy_returns_none(self) -> None:
        """Zero energy is physically impossible — return None."""
        result = compute_grade(total_energy_j=0.0, accuracy=0.90)
        self.assertIsNone(result)

    def test_negative_energy_returns_none(self) -> None:
        """Negative energy is invalid — return None."""
        result = compute_grade(total_energy_j=-100.0, accuracy=0.90)
        self.assertIsNone(result)

    def test_zero_accuracy_returns_none(self) -> None:
        """Zero accuracy should return None (no useful model)."""
        result = compute_grade(total_energy_j=1000.0, accuracy=0.0)
        self.assertIsNone(result)

    def test_none_energy_returns_none(self) -> None:
        """None energy should return None."""
        result = compute_grade(total_energy_j=None, accuracy=0.90)
        self.assertIsNone(result)

    def test_none_accuracy_returns_none(self) -> None:
        """None accuracy should return None."""
        result = compute_grade(total_energy_j=1000.0, accuracy=None)
        self.assertIsNone(result)

    def test_accuracy_normalisation_from_percentage(self) -> None:
        """Accuracy given as percentage (e.g. 90.0) should be normalised to 0.90."""
        result_pct = compute_grade(total_energy_j=1000.0, accuracy=90.0, parameters=11_000_000)
        result_frac = compute_grade(total_energy_j=1000.0, accuracy=0.90, parameters=11_000_000)
        self.assertIsNotNone(result_pct)
        self.assertIsNotNone(result_frac)
        self.assertAlmostEqual(result_pct.eff_score, result_frac.eff_score, places=10)

    def test_accuracy_above_100_returns_none(self) -> None:
        """Accuracy > 100 (after normalisation > 1.0) should return None."""
        result = compute_grade(total_energy_j=1000.0, accuracy=150.0)
        self.assertIsNone(result)

    def test_no_parameters_uses_default(self) -> None:
        """When parameters is None, formula should use log10(10) = 1."""
        result = compute_grade(total_energy_j=1000.0, accuracy=0.90, parameters=None)
        self.assertIsNotNone(result)
        expected = 0.90 * math.log10(10) / 1000.0
        self.assertAlmostEqual(result.eff_score, expected, places=10)

    def test_higher_efficiency_gets_better_grade(self) -> None:
        """More efficient runs (lower energy, same accuracy) should get equal or better grade."""
        grade_order = {"A++": 0, "A+": 1, "A": 2, "B": 3, "C": 4, "D": 5, "E": 6, "F": 7}
        result_low_e = compute_grade(total_energy_j=10.0, accuracy=0.95, parameters=1_000_000)
        result_high_e = compute_grade(total_energy_j=10000.0, accuracy=0.95, parameters=1_000_000)
        self.assertIsNotNone(result_low_e)
        self.assertIsNotNone(result_high_e)
        self.assertLessEqual(grade_order[result_low_e.grade], grade_order[result_high_e.grade])

    def test_custom_reference_score(self) -> None:
        """Custom reference_score should change the grade assignment."""
        result_default = compute_grade(total_energy_j=1000.0, accuracy=0.90, parameters=1_000_000)
        result_custom = compute_grade(total_energy_j=1000.0, accuracy=0.90, parameters=1_000_000,
                                      reference_score=1e-10)
        self.assertIsNotNone(result_default)
        self.assertIsNotNone(result_custom)
        self.assertGreaterEqual(result_custom.pct_of_reference, result_default.pct_of_reference)

    def test_grade_result_fields(self) -> None:
        """GradeResult should contain all expected fields."""
        result = compute_grade(total_energy_j=500.0, accuracy=0.85, parameters=500_000)
        self.assertIsNotNone(result)
        self.assertIsInstance(result.grade, str)
        self.assertIsInstance(result.eff_score, float)
        self.assertIsInstance(result.reference_score, float)
        self.assertIsInstance(result.pct_of_reference, float)
        self.assertIsInstance(result.label, str)
        self.assertIn("Green Grade", result.label)


class TestCalibrateReference(unittest.TestCase):
    """Test the calibrate_reference() function."""

    def test_empty_list_returns_default(self) -> None:
        """Empty list should return the fallback default 1e-4."""
        self.assertAlmostEqual(calibrate_reference([]), 1e-4, places=10)

    def test_single_value(self) -> None:
        """Single-element list should return that value."""
        self.assertAlmostEqual(calibrate_reference([0.005]), 0.005, places=10)

    def test_median_odd_count(self) -> None:
        """Odd-length list should return the middle element."""
        self.assertAlmostEqual(calibrate_reference([0.001, 0.003, 0.005]), 0.003, places=10)

    def test_median_even_count(self) -> None:
        """Even-length list should return the average of two middle elements."""
        self.assertAlmostEqual(calibrate_reference([0.001, 0.002, 0.004, 0.008]), 0.003, places=10)

    def test_filters_none_and_zero(self) -> None:
        """None and zero values should be filtered out."""
        result = calibrate_reference([None, 0, 0.005, None, 0.010])
        self.assertAlmostEqual(result, 0.0075, places=10)

    def test_all_invalid_returns_default(self) -> None:
        """List of all None/zero should return fallback default."""
        self.assertAlmostEqual(calibrate_reference([None, 0, 0, None]), 1e-4, places=10)


if __name__ == "__main__":
    unittest.main()
