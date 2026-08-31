# moniaenergy/metrics/memory.py #
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

class MemoryProcessUsageMetric(BaseMetric):
    """
    A metric for monitoring the memory usage of a specific process.

    This metric provides the memory usage of a process in a specified unit, 
    as defined by the MetricsUnits configuration. It measures the process's 
    Unique Set Size (USS), which represents the memory occupied exclusively 
    by the process (i.e., not shared with others).

    Attributes:
        _memory_unit (str): The unit of memory used for reporting (e.g., 'MB').
        _memory_factor (float): The factor used to convert raw memory usage to the specified unit.
    """
    def __init__(self, metrics_units: MetricsUnits, process:psutil.Process, monitor_interval_ms:int):
        """
        Initializes the MemoryProcessUsageMetric with the given metrics units configuration, process, CPU information 
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
        self._memory_unit, self._memory_factor = metrics_units.memory

    def get_header(self) -> str:
        """
        Returns the header string for the memory usage metric.

        The header is formatted as "Process Memory Usage (X)", where X is the configured memory unit.

        Returns:
            str: The header string.
        """
        return f"Process Memory Usage ({self._memory_unit})"
    
    def get_value(self) -> float:
        """
        Calculates and returns the memory usage of the process in the configured unit.

        The value is obtained by multiplying the process's unique set size (USS) by the conversion factor.

        Returns:
            float: The memory usage value, rounded to two decimal places.
        """
        return float(round(self._process.memory_full_info().uss * self._memory_factor, 2))
