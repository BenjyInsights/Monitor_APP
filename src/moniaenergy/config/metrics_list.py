# moniaenergy/config/metrics_list.py #
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

from .metrics_units import MetricsUnits
from ..metrics.cpu_usage import CPUProcessUsageMetric, CPUUsagePerCoreMetric
from ..metrics.cpu_frequency import CPUFrequencyMetric, CPUFrequencyPerCoreMetric
from ..metrics.cpu_temperature import CPUPackageTemperatureMetric, CPUCoresTemperatureMetric
from ..metrics.cpu_power import CPUPackagePowerMetric, CPUPackageProcessPowerMetric
from ..metrics.memory import MemoryProcessUsageMetric
from ..metrics.gpu_metrics import NVIDIAGPUMetrics

class MetricsList:
    """
    A collection of metric factories.

    The metric factories are lambda functions that create instances of various 
    CPU and memory metrics. Each factory takes three parameters:
    - mu: An instance of MetricsUnits for units configuration.
    - proc: A psutil.Process object representing the process to monitor.
    - interval: An integer representing the monitoring interval in milliseconds.

    The class provides a method to create a list of metric instances using the 
    given metrics units, process and monitoring interval.
    """
    def __init__(self):
        """
        Initializes the MetricsList with a collection of metric factories.
        """
        self.metrics_factories = [
            lambda mu, proc, interval: CPUProcessUsageMetric(mu, proc, interval),
            lambda mu, proc, interval: CPUUsagePerCoreMetric(mu, proc, interval),
            lambda mu, proc, interval: CPUFrequencyMetric(mu, proc, interval),
            lambda mu, proc, interval: CPUFrequencyPerCoreMetric(mu, proc, interval),
            lambda mu, proc, interval: CPUPackageTemperatureMetric(mu, proc, interval),
            lambda mu, proc, interval: CPUCoresTemperatureMetric(mu, proc, interval),
            lambda mu, proc, interval: CPUPackagePowerMetric(mu, proc, interval),
            lambda mu, proc, interval: CPUPackageProcessPowerMetric(mu, proc, interval),
            lambda mu, proc, interval: MemoryProcessUsageMetric(mu, proc, interval),
            lambda mu, proc, interval: NVIDIAGPUMetrics(mu, proc, interval)
        ]

    def get_metrics(self, metrics_units:MetricsUnits, process:psutil.Process, monitor_interval_ms:int) -> list:
        """
        Returns a list of metric instances created from the metric factories in 
        'metrics_factories' using the given metrics units, process and monitoring interval.

        Args:
            metrics_units (MetricsUnits): The instance of MetricsUnits holding the units configuration.
            process (psutil.Process): The process to monitor.
            monitor_interval_ms (int): The monitoring interval in milliseconds.

        Returns:
            list: A list of metric instances.
        """
        return [factory(metrics_units, process, monitor_interval_ms) for factory in self.metrics_factories]

