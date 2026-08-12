# metrics/CPU_Metrics/cpu_frequency.py #
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

class CPUFrequencyMetric(BaseMetric):
    """
    Class to measure the CPU frequency in a specified unit.

    The CPU frequency is measured in MHz by default (using the psutil library)
    but can be converted to GHz by setting the frequency unit to 'GHz' in the MetricsUnits configuration.

    Attributes:
        _frequency_unit (str): The unit of frequency to use (Typically MHz or GHz).
        _frequency_factor (float): The factor to multiply the frequency by to get the desired unit.
    """
    def __init__(self, metrics_units:MetricsUnits, process:psutil.Process, monitor_interval_ms:int):
        """
        Initializes the CPUFrequencyMetric with the given metrics units configuration, process
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
        self._frequency_unit, self._frequency_factor = metrics_units.frequency
    
    def get_header(self) -> str:
        """
        Returns the header string for the CPU frequency metric.

        The header is formatted as "CPU Frequency (X)", where X is the configured frequency unit.

        Returns:
            str: The header string.
        """
        return f"CPU Frequency ({self._frequency_unit})"

    def get_value(self) -> float:
        """
        Calculates and returns the current CPU frequency in the configured unit.

        The frequency is obtained using the psutil library, which provides the current 
        frequency in MHz by default. This value is then scaled by the frequency factor to 
        convert it to the desired unit (e.g., GHz).

        Returns:
            float: The CPU frequency as a float, rounded to two decimal places.
        """
        return float(round(psutil.cpu_freq().current * self._frequency_factor, 2))


class CPUFrequencyPerCoreMetric(BaseMetric):
    """
    Class to measure the CPU frequency per core in a specified unit.

    The CPU frequency is measured in MHz by default (using the psutil library)
    but can be converted to GHz by setting the frequency unit to 'GHz' in the MetricsUnits configuration.

    Attributes:
        _frequency_unit (str): The unit of frequency to use (typically 'MHz' or 'GHz').
        _frequency_factor (float): The factor to multiply the frequency by to convert it to the desired unit.
    """
    def __init__(self, metrics_units:MetricsUnits, process:psutil.Process, monitor_interval_ms:int):
        """
        Initializes the CPUFrequencyPerCoreMetric with the given metrics units configuration, process
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
        self._frequency_unit, self._frequency_factor = metrics_units.frequency

    def get_header(self) -> str:
        """
        Returns the header string for the CPU frequency per core metric.

        The header is formatted as "CPU Frequency Per Core (X)", where X is the configured frequency unit.

        Returns:
            str: The header string.
        """
        return f"CPU Frequency Per Core/Thread ({self._frequency_unit})"

    def get_value(self) -> dict:
        """
        Calculates and returns the CPU frequency for each core in the configured unit.

        The frequency for each core is obtained using the psutil library, which provides the current 
        frequency per core in MHz by default. This value is then scaled by the frequency factor to 
        convert it to the desired unit (e.g., GHz).

        Returns:
            dict: A dictionary where the keys are core identifiers (e.g., "Core 0") and the values are 
                  the CPU frequencies as floats, rounded to two decimal places.
        """
        return {"Core " + str(i): 
                float(round(core_freq.current * self._frequency_factor, 2)) 
                for i, core_freq in enumerate(psutil.cpu_freq(percpu=True))}
