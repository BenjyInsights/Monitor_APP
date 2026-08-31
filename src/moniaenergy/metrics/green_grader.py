# moniaenergy/metrics/green_grader.py #
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
green_grader.py — Universal energy efficiency grader for ML training runs.

Computes a "Green AI Grade" (A++, A+, A, B, C, D, E, F) using a normalized
efficiency formula that is hardware-agnostic and dataset-independent:

    eff_score = (accuracy * log10(max(parameters, 10))) / total_energy_j

The grade is then determined by comparing eff_score against tier thresholds
derived from the distribution of eff_scores observed in your own benchmark
runs (or sensible defaults when no benchmarks are available).

This avoids hardcoding CIFAR-10-specific thresholds and works correctly across
datasets (ImageNet, CIFAR-100, etc.) and across model sizes (small CNNs to
large Transformers).
"""

import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Grade definition
# ---------------------------------------------------------------------------

#: Ordered from best to worst.  Each entry is (label, minimum_score_fraction).
#: Fractions are relative to the "reference" efficiency (A = 100 %).
#: These are auto-calibrated from benchmark data when available; the constants
#: below are sensible defaults that match the observed CIFAR-10 benchmark range.
_GRADE_TIERS: list[tuple[str, float]] = [
    ("A++", 8.0),   # ≥ 800 % of reference (top-of-class)
    ("A+",  4.0),   # ≥ 400 %
    ("A",   2.0),   # ≥ 200 %
    ("B",   1.0),   # ≥ 100 % (reference)
    ("C",   0.5),   # ≥  50 %
    ("D",   0.2),   # ≥  20 %
    ("E",   0.05),  # ≥   5 %
    ("F",   0.0),   # < 5 % (very inefficient)
]


@dataclass
class GradeResult:
    """Result returned by :func:`compute_grade`."""

    grade: str
    """Letter grade (A++, A+, A, B, C, D, E, F)."""

    eff_score: float
    """Raw efficiency score: (accuracy × log10(params)) / energy_j."""

    reference_score: float
    """The score that corresponds to grade B (the calibration reference)."""

    pct_of_reference: float
    """eff_score expressed as a percentage of reference_score."""

    label: str
    """Human-readable one-line summary."""


def compute_grade(
    total_energy_j: float,
    accuracy: float,
    parameters: Optional[int] = None,
    reference_score: Optional[float] = None,
) -> Optional[GradeResult]:
    """Compute the universal Green AI Grade for a training epoch or run.

    The formula normalises energy consumption by both the achieved accuracy and
    the model size (log-scale), so that a large model that achieves high
    accuracy can still earn a good grade if it does so efficiently, while a
    tiny model that wastes energy on low accuracy is penalised.

    Parameters
    ----------
    total_energy_j:
        Total energy consumed (Joules).  Must be > 0.
    accuracy:
        Training or test accuracy in **[0, 1]** range.  Values in [0, 100]
        are accepted and normalised automatically.
    parameters:
        Number of trainable model parameters.  When ``None`` the formula
        reduces to ``accuracy / energy_j`` (equivalent to assuming ~10 params,
        i.e. log10 = 1).
    reference_score:
        The score value that corresponds to grade **B**.  When ``None``, a
        sensible default of ``1e-4`` is used (calibrated on RTX 6000 Ada /
        CIFAR-10 benchmarks).  Pass the median score from your own benchmark
        database for maximum accuracy.

    Returns
    -------
    GradeResult or None
        ``None`` when inputs are invalid (zero energy, zero accuracy, etc.).
    """
    # --- Sanity checks -------------------------------------------------------
    if total_energy_j is None or total_energy_j <= 0:
        return None
    if accuracy is None or accuracy == 0:
        return None

    # Normalise accuracy to [0, 1]
    acc = float(accuracy)
    if acc > 1.0:
        acc /= 100.0
    if acc <= 0 or acc > 1:
        return None

    # --- Efficiency formula --------------------------------------------------
    n_params = max(int(parameters), 10) if parameters else 10
    log_params = math.log10(n_params)
    eff_score = (acc * log_params) / float(total_energy_j)

    # --- Calibration reference -----------------------------------------------
    ref = float(reference_score) if reference_score and reference_score > 0 else 1e-4
    pct = eff_score / ref

    # --- Assign grade --------------------------------------------------------
    grade = "F"
    for label, min_fraction in _GRADE_TIERS:
        if pct >= min_fraction:
            grade = label
            break

    human_label = (
        f"Green Grade: {grade}  "
        f"(eff={eff_score:.3e}, ref={ref:.3e}, {pct * 100:.1f}% of reference)"
    )

    return GradeResult(
        grade=grade,
        eff_score=eff_score,
        reference_score=ref,
        pct_of_reference=pct,
        label=human_label,
    )


def calibrate_reference(eff_scores: list[float]) -> float:
    """Derive a calibrated reference score from a list of observed values.

    Uses the **median** so that outliers (e.g. CPU runs) do not skew the
    reference.  A minimum of 3 observations is recommended.

    Parameters
    ----------
    eff_scores:
        List of raw ``eff_score`` values from past runs.

    Returns
    -------
    float
        The median value, or 1e-4 when the list is empty.
    """
    valid = sorted(v for v in eff_scores if v and v > 0)
    if not valid:
        return 1e-4
    mid = len(valid) // 2
    return valid[mid] if len(valid) % 2 != 0 else (valid[mid - 1] + valid[mid]) / 2
