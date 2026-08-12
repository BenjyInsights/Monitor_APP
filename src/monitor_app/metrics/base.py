# metrics/base.py #
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

from abc import ABC, abstractmethod
from typing import Any

from ..config.metrics_units import MetricsUnits

class BaseMetric(ABC):
    """Abstract class that defines the interface for the metrics.

    Metrics are used to measure properties of the system or of the process
    being monitored. They are used to generate log entries that can be
    written to a file.

    The interface for metrics consists of two methods: 
        - 'get_header': method should return a string thats
                        should be used as the key in the log entry for this metric.
        - 'get_value': method should return the value of the metric, which will
                       be used as the value in the log entry for this metric.
    """
    def __init__(self, metrics_units:MetricsUnits, process:psutil.Process, monitor_interval_ms:int):
        self._metrics_units = metrics_units
        self._process = process
        self._monitor_interval_ms = monitor_interval_ms

    @abstractmethod
    def get_header(self) -> str:
        """
        Returns a string that should be used as the header in the log file
        for this metric.

        This string should be a valid JSON key, and should describe the
        metric that is being logged.

        Returns:
            str: The header string.
        """
        pass

    @abstractmethod
    def get_value(self) -> Any:
        """
        Abstract method to retrieve the value of the metric.

        This method should be implemented by subclasses to return the
        metric's value in the desired format (e.g., str, float, dict).
        """
        pass

