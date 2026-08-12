# cli/parse_monitor.py #
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

import os
from .metrics_units import MetricsUnits

class ParseMonitorArguments:
    """
    This class is used to parse and validate the arguments passed to the `@monitor` decorator or to the `with Monitor()` context manager.

    The class is initialized with the following parameters:

    - `context`: A string representing the context of the monitoring.
    - `interval`: The monitoring interval in seconds.
    - `log_file_path`: The path to the log file for storing monitoring data.
    - `frequency_unit`: The unit for frequency measurement.
    - `memory_unit`: The unit for memory measurement.
    - `temperature_unit`: The unit for temperature measurement.

    The class validates the arguments and stores them in instance variables for later use.
    """
    def __init__(self,
                 context:str,
                 interval:int,
                 log_file_path:str,
                 frequency_unit:str,
                 memory_unit:str,
                 temperature_unit:str):
        """Validates and stores all monitoring parameters.

        Args:
            context (str): A string describing the monitoring context.
            interval (int): Monitoring interval in seconds; must be positive.
            log_file_path (str): Path to the output log file.
            frequency_unit (str): Unit for CPU frequency measurements.
            memory_unit (str): Unit for memory measurements.
            temperature_unit (str): Unit for temperature measurements.

        Raises:
            ValueError: If any parameter fails validation.
        """
        self.context = self._validate_context(context)
        self.interval = self._validate_interval(interval)
        self.log_file_path = self._validate_log_path(log_file_path)
        self.frequency_unit = self._validate_frequency(frequency_unit)
        self.memory_unit = self._validate_memory(memory_unit)
        self.temperature_unit = self._validate_temperature(temperature_unit)

    def _validate_context(self, context):
        """
        Validate and normalize the context value.

        Args:
            context (str): The context value to validate.

        Returns:
            str: The validated and normalized context value.

        Raises:
            ValueError: If the context value is not a string.
        """
        if not isinstance(context, str):
            raise ValueError("Context must be a string.")
        return context


    def _validate_interval(self, interval):
        """
        Validate and normalize the interval value.

        Args:
            interval (int): The interval value to validate.

        Returns:
            int: The validated interval value.

        Raises:
            ValueError: If the interval value is invalid.
        """
        try:
            ivalue = int(interval)
        except Exception:
            raise ValueError(f"Invalid interval value: {interval}. Must be an integer.")
        if ivalue <= 0:
            raise ValueError("Interval must be greater than 0.")
        return ivalue


    def _validate_log_path(self, path):
        """
        Validate and normalize the log file path.

        Args:
            path (str): The log file path to validate.

        Returns:
            str: The validated and normalized log file path.

        Raises:
            ValueError: If the log file path is not a string.
        """
        if not isinstance(path, str):
            raise ValueError("Log file path must be a string.")
        logs_dir = os.path.dirname(path)
        if logs_dir and not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)
        return path


    def _validate_frequency(self, unit):
        """
        Validates a frequency unit.

        Args:
            unit (str): The unit to validate.

        Raises:
            ValueError: If the unit is not one of the valid options.

        Returns:
            str: The validated unit.
        """
        valid = MetricsUnits.available_frequency_units()
        if unit not in valid:
            raise ValueError(f"Invalid frequency unit '{unit}'. Valid options are: {valid}")
        return unit
    

    def _validate_memory(self, unit):
        """
        Validates a memory unit.

        Args:
            unit (str): The unit to validate.

        Raises:
            ValueError: If the unit is not one of the valid options.

        Returns:
            str: The validated unit.
        """
        valid = MetricsUnits.available_memory_units()
        if unit not in valid:
            raise ValueError(f"Invalid memory unit '{unit}'. Valid options are: {valid}")
        return unit
    

    def _validate_temperature(self, unit):
        """
        Validates a temperature unit.

        Args:
            unit (str): The unit to validate.

        Raises:
            ValueError: If the unit is not one of the valid options.

        Returns:
            str: The validated unit.
        """
        valid = MetricsUnits.available_temperature_units()
        if unit not in valid:
            raise ValueError(f"Invalid temperature unit '{unit}'. Valid options are: {valid}")
        return unit