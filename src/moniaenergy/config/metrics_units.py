# moniaenergy/config/metrics_units.py #
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

from typing import Callable

class MetricsUnits:
    """
    Class to manage the different units of measurement for CPU frequency, memory and temperature.

    The units are defined in the class variables _FREQUENCY_UNIT, _MEMORY_UNIT and _TEMPERATURE_UNIT.
    The dictionaries have the unit as key and a factor as value. The factor is the value to multiply the measurement by to get the desired unit.
    """
    _FREQUENCY_UNIT = {"MHz": 1, "GHz": 1 / 1000}
    _MEMORY_UNIT = {"B": 1, "KB": 1 / 1024, "MB": 1 / (1024 ** 2), "GB": 1 / (1024 ** 3)}
    _TEMPERATURE_UNIT = {"ºC": lambda x: x, "ºF": lambda x: (x * (9 / 5)) + 32}

    @classmethod
    def available_frequency_units(cls) -> list[str]:
        """
        Returns a list of the available frequency units.

        Returns:
            list[str]: A list of the available frequency units.
        """
        return list(cls._FREQUENCY_UNIT.keys())

    @classmethod
    def available_memory_units(cls) -> list[str]:
        """
        Returns a list of the available memory units.

        Returns:
            list[str]: A list of the available memory units.
        """
        return list(cls._MEMORY_UNIT.keys())

    @classmethod
    def available_temperature_units(cls) -> list[str]:
        """
        Returns a list of the available temperature units.

        Returns:
            list[str]: A list of the available temperature units.
        """
        return list(cls._TEMPERATURE_UNIT.keys())

    def __init__(self, frequency_unit: str, memory_unit: str, temperature_unit: str):
        """
        Construct a MetricsUnits object.

        Args:
            frequency_unit (str): The unit of frequency to use. This should be one of the values returned by `available_frequency_units()`.
            memory_unit (str): The unit of memory to use. This should be one of the values returned by `available_memory_units()`.
            temperature_unit (str): The unit of temperature to use. This should be one of the values returned by `available_temperature_units()`.

        Returns:
            None
        """
        self.set_metrics_units(frequency_unit, memory_unit, temperature_unit)

    def set_metrics_units(self, frequency_unit: str, memory_unit: str, temperature_unit: str) -> None:
        """
        Sets the metrics units for frequency, memory, and temperature by updating the corresponding
        unit and factor attributes.

        Args:
            frequency_unit (str): The unit of frequency to use.
            memory_unit (str): The unit of memory to use.
            temperature_unit (str): The unit of temperature to use.

        Returns:
            None
        """
        self._frequency_unit = frequency_unit
        self._frequency_factor = self._FREQUENCY_UNIT.get(frequency_unit)
        self._memory_unit = memory_unit
        self._memory_factor = self._MEMORY_UNIT.get(memory_unit)
        self._temperature_unit = temperature_unit
        self._temperature_factor = self._TEMPERATURE_UNIT.get(temperature_unit)

    @property
    def frequency(self) -> tuple[str, float]:
        """
        The unit and factor for frequency.

        Returns:
            tuple[str, float]: A tuple containing the unit of frequency as a string and the factor to convert to that unit as a float.
        """
        return self._frequency_unit, self._frequency_factor

    @property
    def memory(self) -> tuple[str, float]:
        """
        The unit and factor for memory.

        Returns:
            tuple[str, float]: A tuple containing the unit of memory as a string and the factor to convert to that unit as a float.
        """
        return self._memory_unit, self._memory_factor

    @property
    def temperature(self) -> tuple[str, Callable[[float], float]]:
        """
        The unit and factor for temperature.

        Returns:
            tuple[str, callable]: A tuple containing the unit of temperature as a string and the factor to convert to that unit as a lambda function.
        """
        return self._temperature_unit, self._temperature_factor
