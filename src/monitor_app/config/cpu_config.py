# config/cpu_config.py #
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

class CPUConfiguration:
    """
    Configuration class for CPU-related settings and parameters.

    This class provides access to various CPU configuration details, including temperature sensor
    configurations and header strings for temperature metrics. It supports multiple CPU vendors 
    and provides methods to retrieve configurations specific to each vendor.

    Attributes:
        _cpu_configs (dict): A dictionary containing configurations for different CPU vendors.
        _rapl_path (str): The default path for accessing energy readings through the RAPL interface.

    Methods:
        get_supported_cpus(): Retrieves the list of supported CPU vendors.
        get_temperature_sensor_config(cpu_vendor): Returns the temperature sensor configuration for a given CPU vendor.
        get_temperature_header(cpu_vendor, measurement_type): Returns the header string for a CPU temperature metric.
        get_rapl_path(): Returns the RAPL path used for accessing energy readings.
    """
    _cpu_configs = {
        "GenuineIntel": {
            "temperature_sensor_config": {
                "sensor_name": "coretemp",
                "package_label": "Package id 0"
            },
            "headers": {
                "cores": "CPU Cores Temperatures",
                "package": "CPU Package Temperature"
            }
        },
        "AuthenticAMD": {
            "temperature_sensor_config": {
                "sensor_name": "k10temp",
                "package_label": "Tctl"
            },
            "headers": {
                "cores": "CPU CCD Temperatures",
                "package": "CPU CTL Temperature"
            }
        }
    }

    _rapl_path = "/sys/class/powercap/intel-rapl/intel-rapl:0"

    @classmethod
    def get_supported_cpus(cls) -> list[str]:
        """Retrieve the list of supported CPU vendors."""
        return list(cls._cpu_configs.keys())

    @classmethod
    def get_temperature_sensor_config(cls, cpu_vendor: str) -> dict:
        """
        Return the temperature sensor configuration for the given CPU vendor.
        
        Args:
            cpu_vendor (str): The CPU vendor.
        
        Returns:
            dict: The temperature sensor configuration dictionary.
        """
        return cls._cpu_configs.get(cpu_vendor, {}).get("temperature_sensor_config", {})

    @classmethod
    def get_temperature_header(cls, cpu_vendor: str, measurement_type: str) -> str:
        """
        Return the header string for a CPU temperature metric.
        
        Args:
            cpu_vendor (str): The CPU vendor.
            measurement_type (str): The type of measurement (either "cores" or "package").
        
        Returns:
            str: The header string for the CPU temperature metric.
        """
        return cls._cpu_configs.get(cpu_vendor, {}).get("headers", {}).get(measurement_type, "CPU Temperature")

    @classmethod
    def get_rapl_path(cls) -> str:
        """Return the RAPL path used for accessing energy readings."""
        return cls._rapl_path
