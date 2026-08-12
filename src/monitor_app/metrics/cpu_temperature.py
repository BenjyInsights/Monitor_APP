# metrics/CPU_Metrics/cpu_temperature.py #
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
import re

from .base import BaseMetric
from ..config.metrics_units import MetricsUnits
from ..config.cpu_config import CPUConfiguration
from ..utils.cpu_info import CPUInfo

from operator import itemgetter

class CPUPackageTemperatureMetric(BaseMetric):
    """Measures the CPU package (die) temperature.

    Reads the package-level temperature sensor via psutil, using the
    vendor-specific sensor name and label defined in ``CPUConfiguration``.
    Supports Celsius and Fahrenheit output units.
    """

    def __init__(self, metrics_units:MetricsUnits, process:psutil.Process, monitor_interval_ms:int):
        """
        Initializes the CPUPackageTemperatureMetric with the given metrics units configuration, process 
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
        # Temperature factor is not necessary because psutil library 
        # can read in Celsius or Fahrenheit with a parameter
        self._temperature_unit, _ = metrics_units.temperature 
        self.cpu_vendor = CPUInfo.get_cpu_vendor()

    def get_header(self) -> str:
        """
        Returns the header string for the CPU package temperature metric.

        The header is formatted as "CPU Package Temperature (X)", where X is the configured temperature unit.

        Returns:
            str: The header string.
        """
        return f"{CPUConfiguration.get_temperature_header(self.cpu_vendor, measurement_type='package')} ({self._temperature_unit})"
    
    def get_value(self) -> float:
        """
        Retrieves the current CPU package temperature.

        The temperature is measured using the psutil library and is returned in the unit
        configured in the metrics units. The method retrieves sensor data specific to the
        CPU vendor and converts the temperature to Fahrenheit if specified. 

        Returns:
            float: The CPU package temperature rounded to one decimal place. Returns 0.0 if
                the package label is not found.
        """
        fahrenheit_bool = self._metrics_units.temperature[0] == "ºF"
        sensor_name, package_label = itemgetter("sensor_name", "package_label")(CPUConfiguration.get_temperature_sensor_config(self.cpu_vendor))
        cpu_temps = psutil.sensors_temperatures(fahrenheit=fahrenheit_bool).get(sensor_name, [])

        return next((float(round(ct.current, 1)) for ct in cpu_temps if ct.label == package_label), 0.0)
    

class CPUCoresTemperatureMetric(BaseMetric):
    """Measures the per-core CPU temperatures.

    Returns a dictionary mapping each core label to its current temperature,
    sorted numerically by core index and excluding the package-level entry.
    Supports Celsius and Fahrenheit output units.
    """

    def __init__(self, metrics_units:MetricsUnits, process:psutil.Process, monitor_interval_ms:int):
        """
        Initializes the CPUCoresTemperatureMetric with the given metrics units configuration, process 
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
        # Temperature factor is not necessary because psutil library 
        # can read in Celsius or Fahrenheit with a parameter
        self._temperature_unit, _ = metrics_units.temperature 
        self.cpu_vendor = CPUInfo.get_cpu_vendor()

    def get_header(self) -> str:
        """
        Returns the header string for the CPU cores temperature metric.

        The header is retrieved based on the CPU vendor and measurement type,
        formatted as "CPU Cores Temperature (X)", where X is the configured
        temperature unit.

        Returns:
            str: The header string.
        """
        return f"{CPUConfiguration.get_temperature_header(self.cpu_vendor, measurement_type='cores')} ({self._temperature_unit})"
    
    def get_value(self) -> dict:
        """
        Retrieves the current CPU core temperatures.

        The temperatures are measured using the psutil library with per-core readings and 
        are returned in the unit configured in the metrics units. The method retrieves sensor 
        data specific to the CPU vendor and converts the temperature to Fahrenheit if specified. 
        The retrieved temperatures are sorted by core number and the package label is excluded. 

        Returns:
            dict: A dictionary where the keys are core identifiers (e.g., "Core 0") and the values are 
                the CPU core temperatures as floats, rounded to one decimal place. If no sensors are
                found, the method returns an empty dictionary.
        """
        fahrenheit_bool = self._metrics_units.temperature[0] == "ºF"
        sensor_name, package_label = itemgetter("sensor_name", "package_label")(CPUConfiguration.get_temperature_sensor_config(self.cpu_vendor))
        cpu_temps = psutil.sensors_temperatures(fahrenheit=fahrenheit_bool).get(sensor_name, [])
        sorted_temps = sorted(cpu_temps, key=lambda x: int(re.search(r"\d+", x.label).group()) if re.search(r"\d+", x.label) else 9999)

        return {temp.label: float(round(temp.current, 1)) for temp in sorted_temps if temp.label != package_label}