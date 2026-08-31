# moniaenergy/metrics/cpu_usage.py #
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

import psutil

from .base import BaseMetric
from ..config.metrics_units import MetricsUnits
from ..utils.cpu_info import CPUInfo

class CPUProcessUsageMetric(BaseMetric):
    """Measures the CPU usage percentage of the monitored process.

    The value is normalised by the number of logical CPU threads so that a
    fully CPU-bound single-threaded process reports ~100 % regardless of the
    core count of the host machine.
    """

    def __init__(self, metrics_units:MetricsUnits, process:psutil.Process, monitor_interval_ms:int):
        """
        Initializes the CPUProcessUsageMetric with the given metrics units configuration, process 
        and monitoring interval.

        Args:
            metrics_units (MetricsUnits): The instance of MetricsUnits holding the units configuration.
            process (psutil.Process): The process to monitor.
            monitor_interval_ms (int): The monitoring interval in milliseconds.

        Raises:
            ValueError: If metrics_units is None.
        """
        if metrics_units is None:
            raise ValueError("A valid MetricsUnits instance is required")
        
        super().__init__(metrics_units, process, monitor_interval_ms)        
        
    def get_header(self) -> str:
        """
        Returns the header string for the CPU process usage metric.

        The header is formatted as "CPU Process Usage (%)", indicating the percentage of total CPU time used by the process.

        Returns:
            str: The header string.
        """
        return f"CPU Process Usage (%)"
    
    def get_value(self) -> float:
        """
        Calculates and returns the CPU usage percentage for the monitored process.

        The method computes the percentage of total CPU time used by the specific 
        process, normalized by the number of CPU threads, ensuring an accurate 
        measurement in multi-threaded environments.

        Returns:
            float: The CPU usage percentage as a float, rounded to two decimal places.
        """
        return float(round(self._process.cpu_percent(interval=None) / CPUInfo.get_cpu_threads(), 2))
    

class CPUUsagePerCoreMetric(BaseMetric):
    """Measures the system-wide CPU usage percentage for each logical core.

    Reports a dictionary mapping each core identifier (e.g. ``"Core 0"``) to
    its current usage percentage as returned by ``psutil.cpu_percent(percpu=True)``.
    """

    def __init__(self, metrics_units:MetricsUnits, process:psutil.Process, monitor_interval_ms:int):
        """
        Initializes the CPUUsagePerCoreMetric with the given metrics units configuration, process 
        and monitoring interval.

        Args:
            metrics_units (MetricsUnits): The instance of MetricsUnits holding the units configuration.
            process (psutil.Process): The process to monitor.
            monitor_interval_ms (int): The monitoring interval in milliseconds.

        Raises:
            ValueError: If metrics_units is None.
        """
        if metrics_units is None:
            raise ValueError("A valid MetricsUnits instance is required")
        
        super().__init__(metrics_units, process, monitor_interval_ms)

    def get_header(self) -> str:
        """
        Returns the header string for the CPU usage per core metric.

        The header is formatted as "CPU Global Usage per Core/Thread (%)", indicating the percentage of total CPU time used by the process on each core.

        Returns:
            str: The header string.
        """
        return f"CPU Global Usage per Core/Thread (%)"
    

    def get_value(self) -> dict:
        """Returns the current CPU usage percentage for each logical core.

        Returns:
            dict: Mapping of ``"Core <N>"`` to usage percentage (float, 2 d.p.).
        """
        return {f"Core {core_index}":
                float(round(usage, 2)) for core_index, usage in
                enumerate(psutil.cpu_percent(interval=None, percpu=True))}