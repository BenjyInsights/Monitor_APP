# moniaenergy/monitor/base_energy_monitor.py #
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
base_energy_monitor.py — Abstract interface for hardware energy monitors.

Provides a common abstraction layer for energy/power monitors (GPU, CPU, etc.)
with graceful degradation when hardware access fails (e.g., missing root
privileges) or hardware is unavailable.

This module enables pluggable monitor implementations and simplifies testing
via dependency injection and mock adapters.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any, Dict


logger = logging.getLogger(__name__)


class MonitorMode(Enum):
    """Operational mode of a hardware monitor."""

    ACTIVE = "active"
    """Monitor is active and can read/write settings."""

    READ_ONLY = "read_only"
    """Monitor can only read metrics due to permission restrictions."""

    UNAVAILABLE = "unavailable"
    """Hardware not detected or permanently unavailable."""


@dataclass
class EnergyReading:
    """A single instantaneous energy/power reading from hardware."""

    timestamp: float
    """Unix timestamp in seconds."""

    power_w: Optional[float] = None
    """Current power draw in watts. None if unavailable."""

    energy_j: Optional[float] = None
    """Accumulated energy in joules (time-integrated). None if unavailable."""

    temperature_c: Optional[float] = None
    """Current temperature in Celsius. None if unavailable."""

    utilization_pct: Optional[float] = None
    """Busy/active fraction [0, 100]. None if unavailable."""

    metadata: Dict[str, Any] | None = None
    """Optional extra fields (e.g., frequency, voltage, SM utilization)."""


class MonitorError(Exception):
    """Base exception for hardware monitor errors."""

    pass


class PermissionError(MonitorError):
    """Raised when operation requires elevated privileges (e.g., root)."""

    pass


class HardwareNotFoundError(MonitorError):
    """Raised when required hardware is not detected."""

    pass


class BaseEnergyMonitor(ABC):
    """Abstract base class for hardware energy monitors.

    Provides a unified interface for reading energy/power metrics from various
    hardware backends (GPU, CPU packages, etc.) with graceful degradation when
    hardware is unavailable or permissions are insufficient.

    Subclasses must implement:
        - ``_initialize_hardware()``: Probe and initialize hardware access.
        - ``_read_metrics()``: Fetch current meter values.
        - ``_shutdown()``: Release hardware resources.

    Optional methods for active power management:
        - ``set_power_limit()``: Enforce a power cap (requires root/elevated).
        - ``get_power_limit()``: Query current power limit.
        - ``restore_defaults()``: Revert to factory settings.

    Examples
    --------
    **Read-only usage (e.g., monitoring only)**:

    >>> monitor = NvidiaGpuMonitor(gpu_index=0)
    >>> monitor.initialize()
    >>> if monitor.is_active():
    ...     reading = monitor.read_metrics()
    ...     print(f"Power: {reading.power_w} W, Temp: {reading.temperature_c} C")
    >>> monitor.shutdown()

    **Active power management (requires root)**:

    >>> monitor = NvidiaGpuMonitor(gpu_index=0)
    >>> monitor.initialize()
    >>> if monitor.mode == MonitorMode.ACTIVE:
    ...     monitor.set_power_limit(250)  # Cap at 250W
    ...     # ... run training ...
    ...     monitor.restore_defaults()  # Restore original limit
    """

    def __init__(self, device_name: str | None = None) -> None:
        """Initialize the monitor instance.

        Parameters
        ----------
        device_name : str, optional
            Human-readable device identifier (e.g., "GPU-0", "CPU package-0").
            If not provided, defaults to the class name. Useful for logging.
        """
        self._device_name: str = device_name or self.__class__.__name__
        self._mode: MonitorMode = MonitorMode.UNAVAILABLE
        self._initialized: bool = False
        self._last_reading: Optional[EnergyReading] = None

    # -----------------------------------------------------------------------
    # Lifecycle management
    # -----------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize hardware access and probe capabilities.

        Should be called before any read operations. On success, sets
        self._initialized = True and determines self._mode based on hardware
        status and available privileges.

        Subclasses override _initialize_hardware() to implement specific logic.

        Raises
        ------
        MonitorError
            If initialization fails catastrophically (e.g., NVML init fails).
            Graceful degradations (missing root) use mode=READ_ONLY or
            mode=UNAVAILABLE without raising.
        """
        if self._initialized:
            return  # Idempotent

        try:
            self._initialize_hardware()
            self._initialized = True
        except Exception as e:
            logger.error(f"[{self._device_name}] Initialization failed: {e}")
            self._mode = MonitorMode.UNAVAILABLE
            self._initialized = True

    def shutdown(self) -> None:
        """Release hardware resources.

        Safe to call multiple times. Automatically restores power limits if
        they were modified (depends on subclass implementation).

        Subclasses override _shutdown() for device-specific cleanup.
        """
        if not self._initialized:
            return

        try:
            self._shutdown()
        except Exception as e:
            logger.warning(f"[{self._device_name}] Shutdown warning: {e}")
        finally:
            self._initialized = False

    # -----------------------------------------------------------------------
    # Abstract methods (subclasses must implement)
    # -----------------------------------------------------------------------

    @abstractmethod
    def _initialize_hardware(self) -> None:
        """Probe and initialize hardware access.

        Called by initialize(). Must set self._mode to one of:
        - ACTIVE: Full read/write access available.
        - READ_ONLY: Can read metrics but not modify settings (e.g., no root).
        - UNAVAILABLE: Hardware not detected or unusable.

        Raises
        ------
        MonitorError
            For catastrophic errors (e.g., library init failure).
            Graceful degradations use self._mode = READ_ONLY/UNAVAILABLE.
        """
        pass

    @abstractmethod
    def _read_metrics(self) -> EnergyReading:
        """Fetch current meter values from hardware.

        Called by read_metrics(). Must return valid EnergyReading with
        available fields populated. Fields that cannot be read should be None.

        Returns
        -------
        EnergyReading
            Instantaneous reading with power, temperature, utilization, etc.
        """
        pass

    @abstractmethod
    def _shutdown(self) -> None:
        """Release hardware resources.

        Called by shutdown(). Should clean up handles, restore original settings,
        and deregister signal handlers.
        """
        pass

    # -----------------------------------------------------------------------
    # Public read interface
    # -----------------------------------------------------------------------

    def read_metrics(self) -> EnergyReading | None:
        """Read current energy/power metrics from hardware.

        Returns the latest EnergyReading and caches it in self._last_reading.

        Returns
        -------
        EnergyReading or None
            Instantaneous meter values, or None if hardware is unavailable
            (self.mode == UNAVAILABLE).

        Raises
        ------
        RuntimeError
            If called before initialize().
        """
        if not self._initialized:
            raise RuntimeError(f"[{self._device_name}] Must call initialize() first")

        if self._mode == MonitorMode.UNAVAILABLE:
            return None

        try:
            reading = self._read_metrics()
            self._last_reading = reading
            return reading
        except Exception as e:
            logger.warning(f"[{self._device_name}] Read failed: {e}")
            return None

    def get_last_reading(self) -> EnergyReading | None:
        """Return the most recent successful reading.

        Useful for graceful degradation when a single read fails but you want
        to continue with the last-known valid state.

        Returns
        -------
        EnergyReading or None
            Last successful read, or None if no successful read yet.
        """
        return self._last_reading

    # -----------------------------------------------------------------------
    # Optional active power management (subclasses implement as needed)
    # -----------------------------------------------------------------------

    def set_power_limit(self, limit_w: int) -> bool:
        """Enforce a power limit on the device.

        Optional operation; only supported when mode=ACTIVE.

        Parameters
        ----------
        limit_w : int
            Power limit in watts. Typically [min_cap_w, max_cap_w].

        Returns
        -------
        bool
            True if successful, False if unsupported or failed gracefully.
        """
        return False

    def get_power_limit(self) -> int | None:
        """Query the current power limit.

        Optional operation; returns None if unsupported or unavailable.

        Returns
        -------
        int or None
            Current power limit in watts, or None if unavailable.
        """
        return None

    def restore_defaults(self) -> bool:
        """Restore original/factory power settings.

        Should always succeed when mode=ACTIVE, even on second call.

        Returns
        -------
        bool
            True if successful or already at default, False if failed.
        """
        return False

    # -----------------------------------------------------------------------
    # Status properties
    # -----------------------------------------------------------------------

    @property
    def mode(self) -> MonitorMode:
        """Current operational mode (ACTIVE, READ_ONLY, or UNAVAILABLE)."""
        return self._mode

    @property
    def is_initialized(self) -> bool:
        """Whether initialize() has been called."""
        return self._initialized

    def is_active(self) -> bool:
        """Convenience: True if mode == ACTIVE."""
        return self._mode == MonitorMode.ACTIVE

    def is_available(self) -> bool:
        """Convenience: True if mode != UNAVAILABLE."""
        return self._mode != MonitorMode.UNAVAILABLE

    @property
    def device_name(self) -> str:
        """Human-readable device identifier."""
        return self._device_name

    # -----------------------------------------------------------------------
    # Context manager support (optional)
    # -----------------------------------------------------------------------

    def __enter__(self) -> BaseEnergyMonitor:
        """Enter context: initialize hardware."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context: shutdown hardware (restore power limits, etc.)."""
        self.shutdown()
