# moniaenergy/metrics/cpu_power.py #
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

from typing import Optional

import psutil
import time
import sys
import os

from .cpu_usage import CPUProcessUsageMetric
from .base import BaseMetric
from ..config.metrics_units import MetricsUnits
from ..config.cpu_config import CPUConfiguration

class CPUPackagePowerMetric(BaseMetric):
    """
    Class to measure the CPU package power consumption using RAPL (Running Average Power Limit).

    This class calculates the power consumption of the CPU package by reading energy measurements
    from the RAPL interface. It computes the power in watts based on the change in energy over time.

    When the RAPL counters are not readable (they are root-only since Linux 5.10),
    the metric reports 0.0 W and marks itself unavailable rather than aborting.
    """

    _rapl_warned: bool = False

    def __init__(self, metrics_units: MetricsUnits, process: psutil.Process, monitor_interval_ms: int):
        """
        Initializes the CPUPackagePowerMetric with the given metrics units, process, and monitoring interval.

        Args:
            metrics_units (MetricsUnits): The instance of MetricsUnits holding the units configuration.
            process (psutil.Process): The process to monitor.
            monitor_interval_ms (int): The monitoring interval in milliseconds.
        
        Raises:
            ValueError: If metrics_units is None.
        """
        super().__init__(metrics_units, process, monitor_interval_ms)
        self._rapl_path = CPUConfiguration.get_rapl_path()
        self._energy_file = os.path.join(self._rapl_path, "energy_uj")
        self._max_range_file = os.path.join(self._rapl_path, "max_energy_range_uj")

        # Last values to estimate delta
        self._last_energy_uj = self._read_file_as_int(self._energy_file)
        self._available = self._last_energy_uj is not None
        self._last_time = time.time()

    @staticmethod
    def _read_file_as_int(path: str) -> Optional[int]:
        """
        Reads a file as an integer.

        This method opens a file specified by path and reads its contents as an integer.
        If the file cannot be opened due to permission errors, it prints an error message
        and exits the program.

        Args:
            path (str): Path to the file to read.

        Returns:
            int: The contents of the file as an integer.

        Returns None when the file exists but is not readable by this user,
        so callers can degrade gracefully instead of aborting.
        """
        try:
            with open(path, "r") as f:
                return int(f.read().strip())
        except PermissionError:
            # Since Linux 5.10 the RAPL counters are root-only (mitigation for
            # CVE-2020-8694, "PlatypUS"). Degrade to CPU-unavailable instead of
            # killing the monitoring subprocess: GPU telemetry via NVML needs no
            # privileges and must keep working.
            if not CPUPackagePowerMetric._rapl_warned:
                CPUPackagePowerMetric._rapl_warned = True
                print(
                    f"[monIAenergy] Warning: no permission to read {path}. "
                    "CPU energy will be reported as unavailable; GPU metrics are "
                    "unaffected. To enable CPU measurement run with sudo or "
                    f"grant read access (sudo chmod a+r {path}).",
                    file=sys.stderr,
                )
            return None
    
    def get_header(self) -> str:
        """
        Returns the header string for the CPU package power metric.

        The header is "CPU RAPL Package Power (W)", indicating the power consumption
        of the CPU package in watts.

        Returns:
            str: The header string.
        """
        return "CPU RAPL Package Power (W)"
    
    def get_value(self) -> float:
        """
        Calculates and returns the CPU package power consumption in watts.

        This method reads the current energy usage from the RAPL energy file
        and calculates the power based on the difference from the last recorded
        energy value. It accounts for potential wrap-around of the energy counter.
        The power is computed as the change in energy divided by the time interval
        since the last measurement. The result is rounded to two decimal places.

        Returns:
            float: The CPU package power consumption in watts, rounded to two decimal places.
        """
        # Get current energy and time #
        current_energy_uj = self._read_file_as_int(self._energy_file)
        if current_energy_uj is None or self._last_energy_uj is None:
            # RAPL not readable by this user: report 0.0 W so the sampling loop
            # (and GPU telemetry with it) keeps running.
            return 0.0
        current_time = time.time()

        # Get delta energy #
        delta_energy_uj = current_energy_uj - self._last_energy_uj

        # Check wrap-around #
        if delta_energy_uj < 0:
            max_uj = self._read_file_as_int(self._max_range_file)
            if max_uj is None:
                return 0.0
            delta_energy_uj = (max_uj - self._last_energy_uj) + current_energy_uj

        # Get delta time
        delta_time = current_time - self._last_time
        delta_joules = delta_energy_uj * 1e-6  # uJ to J

        # Calculate POWER #
        power_watts = delta_joules / delta_time if delta_time > 0 else 0.0

        # Update last values #
        self._last_energy_uj = current_energy_uj
        self._last_time = current_time

        return float(round(power_watts, 2))
    

class CPUPackageProcessPowerMetric(BaseMetric):
    def __init__(self, metrics_units: MetricsUnits, process: psutil.Process, monitor_interval_ms: int):
        """
        Initializes the CPUPackageProcessPowerMetric with the given metrics units configuration, process,
        and monitoring interval.

        Args:
            metrics_units (MetricsUnits): The instance of MetricsUnits holding the units configuration.
            process (psutil.Process): The process to monitor.
            monitor_interval_ms (int): The monitoring interval in milliseconds.

        Raises:
            ValueError: If metrics_units is None.
        """
        super().__init__(metrics_units, process, monitor_interval_ms)
        self.package_power = CPUPackagePowerMetric(metrics_units, process, monitor_interval_ms)
        self.cpu_process_usage = CPUProcessUsageMetric(metrics_units, process, monitor_interval_ms)

    def get_header(self) -> str:
        """
        Returns the header string for the CPU RAPL process package power metric.

        The header is formatted as "CPU RAPL Process Package Power (W)".

        Returns:
            str: The header string.
        """
        return "CPU RAPL Process Package Power (W)"

    def get_value(self) -> float:
        """
        Estimates the power used by the process by multiplying the total package power by the process's relative usage.

        The estimation is based on the assumption that the process's relative usage is similar to the package's relative usage.

        Returns:
            float: The estimated power used by the process in watts, rounded to two decimal places.
        """
        total_power = self.package_power.get_value()
        process_usage_percent = self.cpu_process_usage.get_value()
        estimated_power = total_power * (process_usage_percent / 100)

        return float(round(estimated_power, 2))
