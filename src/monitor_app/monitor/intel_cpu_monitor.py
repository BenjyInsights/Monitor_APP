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

"""
intel_cpu_monitor.py — Intel CPU energy monitor via RAPL.

Concrete implementation of BaseEnergyMonitor for Intel CPUs using the
Running Average Power Limit (RAPL) interface. RAPL is available on most
Intel CPUs since Sandy Bridge (2011).

Graceful Degradation
---------------------
- **Mode.READ_ONLY**: Can read energy/power via RAPL sysfs (typical, no root).
- **Mode.UNAVAILABLE**: RAPL sysfs not found or CPU not supported.

Note: RAPL traditionally does not support power *limits* via sysfs on Intel
(that functionality exists in BIOS/firmware). The monitor provides read-only
access to energy metrics.

Examples
--------
>>> from monitor_app.monitor.intel_cpu_monitor import IntelCpuMonitor
>>> mon = IntelCpuMonitor()
>>> with mon:
...     reading = mon.read_metrics()
...     if reading:
...         print(f"Power: {reading.power_w} W")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from .base_energy_monitor import (
    BaseEnergyMonitor,
    EnergyReading,
    MonitorMode,
    MonitorError,
    HardwareNotFoundError,
)

logger = logging.getLogger(__name__)


class IntelCpuMonitor(BaseEnergyMonitor):
    """Intel CPU energy monitor using RAPL (Running Average Power Limit).

    Reads CPU package energy from sysfs RAPL interface. Only supports read
    access to energy/power metrics; does not implement power limits (typically
    configured in BIOS on Intel systems).

    Parameters
    ----------
    package_index : int, optional
        CPU package index (0 = first package, often the only one on single-socket).
        Default is 0.

    Examples
    --------
    >>> mon = IntelCpuMonitor(package_index=0)
    >>> with mon:
    ...     reading = mon.read_metrics()
    ...     print(f"Avg Power: {reading.power_w} W")
    """

    # RAPL sysfs path template for energy data
    _RAPL_SYSFS_BASE = "/sys/class/powercap/intel-rapl"
    _ENERGY_UJ_FILENAME = "energy_uj"
    _MAX_RANGE_UJ_FILENAME = "max_energy_range_uj"
    _NAME_FILENAME = "name"

    def __init__(self, package_index: int = 0) -> None:
        """Initialize Intel CPU RAPL monitor.

        Parameters
        ----------
        package_index : int, optional
            CPU socket index. Default is 0.
        """
        super().__init__(device_name=f"CPU-Package-{package_index}")
        self._package_index = package_index
        self._rapl_path: Optional[str] = None
        self._energy_file: Optional[str] = None
        self._max_range_file: Optional[str] = None

        # Cached values for power computation
        self._last_energy_uj: Optional[int] = None
        self._last_time: Optional[float] = None

    # -----------------------------------------------------------------------
    # BaseEnergyMonitor implementation
    # -----------------------------------------------------------------------

    def _initialize_hardware(self) -> None:
        """Initialize RAPL interface and locate package sysfs path.

        Gracefully sets mode to UNAVAILABLE if RAPL is not available.

        Raises
        ------
        MonitorError
            Should not raise; uses mode=UNAVAILABLE for any issues.
        """
        rapl_path = self._find_rapl_package_path()
        if rapl_path is None:
            logger.warning(f"[{self._device_name}] RAPL sysfs not found")
            self._mode = MonitorMode.UNAVAILABLE
            return

        self._rapl_path = rapl_path
        self._energy_file = os.path.join(rapl_path, self._ENERGY_UJ_FILENAME)
        self._max_range_file = os.path.join(rapl_path, self._MAX_RANGE_UJ_FILENAME)

        # Verify files are readable
        if not os.path.isfile(self._energy_file):
            logger.warning(f"[{self._device_name}] Energy file not found: {self._energy_file}")
            self._mode = MonitorMode.UNAVAILABLE
            return

        # Initialize cached energy/time
        try:
            self._last_energy_uj = self._read_energy_uj()
            self._last_time = time.time()
            self._mode = MonitorMode.READ_ONLY
            logger.info(f"[{self._device_name}] Initialized in READ_ONLY mode")
        except Exception as e:
            logger.warning(f"[{self._device_name}] Failed to read initial energy: {e}")
            self._mode = MonitorMode.UNAVAILABLE

    def _read_metrics(self) -> EnergyReading:
        """Read current CPU package energy via RAPL sysfs.

        Computes power (W) as delta-energy / delta-time since last read.

        Returns
        -------
        EnergyReading
            Populated with power_w (delta), energy_j (cumulative from sysfs),
            and metadata containing raw energy_uj and time interval.

        Raises
        ------
        MonitorError
            If sysfs read fails or hardware not initialized.
        """
        if not self._energy_file or not self._rapl_path:
            raise MonitorError(f"[{self._device_name}] Hardware not initialized")

        current_energy_uj = self._read_energy_uj()
        current_time = time.time()

        # Compute instantaneous power as delta-energy / delta-time
        power_w: Optional[float] = None
        if (self._last_energy_uj is not None and
            self._last_time is not None):
            delta_energy_uj = float(current_energy_uj - self._last_energy_uj)
            delta_time_s = current_time - self._last_time
            if delta_time_s > 0:
                # Convert µJ to J, then divide by delta_time to get W
                power_w = delta_energy_uj / (1_000_000 * delta_time_s)

        # Intervalo respecto a la lectura anterior; se calcula ANTES de refrescar
        # la caché (hacerlo después daba siempre 0,0 s en los metadatos).
        delta_time_s = current_time - (self._last_time or current_time)

        # Update cache
        self._last_energy_uj = current_energy_uj
        self._last_time = current_time

        # Energy in joules (sysfs provides cumulative energy since power-on)
        energy_j = current_energy_uj / 1_000_000.0

        metadata = {
            "energy_uj": current_energy_uj,
            "delta_time_s": delta_time_s,
        }

        return EnergyReading(
            timestamp=current_time,
            power_w=power_w,
            energy_j=energy_j,
            metadata=metadata,
        )

    def _shutdown(self) -> None:
        """Cleanup (Intel RAPL is read-only, no restoration needed)."""
        logger.debug(f"[{self._device_name}] Shutdown (read-only, no action)")
        pass

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def _find_rapl_package_path(self) -> str | None:
        """Locate the RAPL sysfs path for the target package.

        Searches /sys/class/powercap/intel-rapl/ for directories matching
        intel-rapl:{package_index}, e.g., intel-rapl:0 for the first package.

        Returns
        -------
        str or None
            Absolute path to the package directory, or None if not found.
        """
        if not os.path.isdir(self._RAPL_SYSFS_BASE):
            return None

        target_dir = os.path.join(
            self._RAPL_SYSFS_BASE,
            f"intel-rapl:{self._package_index}"
        )

        if os.path.isdir(target_dir):
            return target_dir

        return None

    def _read_energy_uj(self) -> int:
        """Read raw energy value from RAPL sysfs.

        Returns
        -------
        int
            Current energy in microjoules (µJ).

        Raises
        ------
        MonitorError
            If the energy file cannot be read.
        """
        if not self._energy_file:
            raise MonitorError(f"[{self._device_name}] Energy file not set")

        try:
            with open(self._energy_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                return int(content)
        except PermissionError as e:
            raise MonitorError(
                f"[{self._device_name}] Permission denied reading {self._energy_file}: {e}"
            )
        except (ValueError, OSError) as e:
            raise MonitorError(
                f"[{self._device_name}] Failed to read energy: {e}"
            )

    def get_cpu_package_name(self) -> str | None:
        """Retrieve the human-readable name of the CPU package from RAPL.

        Reads from the 'name' file in the RAPL sysfs directory, e.g.,
        "package-0", "psys" (for whole-system power on some platforms).

        Returns
        -------
        str or None
            CPU package name, or None if unavailable.
        """
        if not self._rapl_path:
            return None

        name_file = os.path.join(self._rapl_path, self._NAME_FILENAME)
        try:
            with open(name_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.debug(f"[{self._device_name}] Cannot read name file: {e}")
            return None

    def get_max_energy_range_uj(self) -> int | None:
        """Get the maximum energy range (overflow point) from RAPL sysfs.

        This value indicates the energy counter overflow point; after the
        counter reaches this value, it wraps around to 0.

        Returns
        -------
        int or None
            Max energy range in microjoules, or None if unavailable.
        """
        if not self._max_range_file or not os.path.isfile(self._max_range_file):
            return None

        try:
            with open(self._max_range_file, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except Exception as e:
            logger.debug(f"[{self._device_name}] Cannot read max range: {e}")
            return None
