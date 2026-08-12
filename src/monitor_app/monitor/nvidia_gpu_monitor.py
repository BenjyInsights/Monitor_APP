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
nvidia_gpu_monitor.py — NVIDIA GPU energy monitor via NVML.

Concrete implementation of BaseEnergyMonitor for NVIDIA GPUs using the
NVIDIA Management Library (NVML). Supports:
  - Reading power, temperature, utilization, memory, frequency.
  - Setting/getting power limits (requires NVIDIA driver with permission).
  - Graceful degradation when limitations prevent certain operations.

Graceful Degradation
---------------------
- **Mode.ACTIVE**: Full read/write access (typical with sudo or setuid driver).
- **Mode.READ_ONLY**: Can read metrics but not modify power limits (no root).
- **Mode.UNAVAILABLE**: NVIDIA driver/hardware not detected.

Examples
--------
>>> from monitor_app.monitor.nvidia_gpu_monitor import NvidiaGpuMonitor
>>> mon = NvidiaGpuMonitor(gpu_index=0)
>>> mon.initialize()
>>> if mon.is_active():
...     mon.set_power_limit(250)
>>> reading = mon.read_metrics()
>>> if reading:
...     print(f"Power: {reading.power_w} W")
>>> mon.shutdown()
"""

from __future__ import annotations

import logging
import signal
import time
from typing import Optional

from .base_energy_monitor import (
    BaseEnergyMonitor,
    EnergyReading,
    MonitorMode,
    MonitorError,
    PermissionError,
    HardwareNotFoundError,
)

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


logger = logging.getLogger(__name__)


class NvidiaGpuMonitor(BaseEnergyMonitor):
    """NVIDIA GPU energy monitor using NVML.

    Parameters
    ----------
    gpu_index : int, optional
        NVIDIA NVML device index (0 = first GPU). Default is 0.

    Examples
    --------
    >>> mon = NvidiaGpuMonitor(gpu_index=0)
    >>> with mon:
    ...     reading = mon.read_metrics()
    ...     print(f"Temp: {reading.temperature_c} C")
    """

    def __init__(self, gpu_index: int = 0) -> None:
        """Initialize NVIDIA GPU monitor.

        Parameters
        ----------
        gpu_index : int, optional
            NVML device index. Default is 0.
        """
        super().__init__(device_name=f"GPU-{gpu_index}")
        self._gpu_index = gpu_index
        self._handle: Optional[any] = None
        self._original_power_limit_w: Optional[int] = None
        self._signal_handler_installed = False

    # -----------------------------------------------------------------------
    # BaseEnergyMonitor implementation
    # -----------------------------------------------------------------------

    def _initialize_hardware(self) -> None:
        """Initialize NVML and probe GPU capabilities.

        Sets mode to ACTIVE, READ_ONLY, or UNAVAILABLE based on what
        operations are supported.

        Raises
        ------
        MonitorError
            If NVML library initialization fails. Graceful degradation
            (e.g., missing root) uses mode=READ_ONLY without raising.
        """
        if not _PYNVML_AVAILABLE:
            logger.warning(f"[{self._device_name}] pynvml not installed")
            self._mode = MonitorMode.UNAVAILABLE
            return

        try:
            pynvml.nvmlInit()
            logger.debug(f"[{self._device_name}] NVML initialized")
        except Exception as e:
            logger.warning(f"[{self._device_name}] NVML init failed: {e}")
            self._mode = MonitorMode.UNAVAILABLE
            return

        try:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            logger.debug(f"[{self._device_name}] Device handle acquired")
        except Exception as e:
            logger.warning(f"[{self._device_name}] Device not found: {e}")
            self._mode = MonitorMode.UNAVAILABLE
            return

        # Probe read access (power, thermal)
        can_read = self._probe_read_capability()
        if not can_read:
            logger.warning(f"[{self._device_name}] Cannot read metrics")
            self._mode = MonitorMode.UNAVAILABLE
            return

        # Probe write access (power limit)
        can_write = self._probe_write_capability()
        self._mode = MonitorMode.ACTIVE if can_write else MonitorMode.READ_ONLY

        logger.info(
            f"[{self._device_name}] Initialized in {self._mode.value} mode"
        )

        # Register signal handlers for graceful restore on SIGINT/SIGTERM
        if can_write:
            self._register_signal_handlers()

    def _read_metrics(self) -> EnergyReading:
        """Read current GPU metrics.

        Returns
        -------
        EnergyReading
            Populated with power_w, temperature_c, utilization_pct, and
            metadata (frequency, memory utilization).
        """
        if not self._handle:
            raise MonitorError(f"[{self._device_name}] Hardware not initialized")

        power_w: Optional[float] = None
        temp_c: Optional[float] = None
        util_pct: Optional[float] = None
        freq_mhz: Optional[float] = None
        mem_used_mb: Optional[float] = None
        mem_total_mb: Optional[float] = None
        sm_util_pct: Optional[float] = None

        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            power_w = power_mw / 1000.0
        except Exception as e:
            logger.debug(f"[{self._device_name}] Cannot read power: {e}")

        try:
            temp_c = float(pynvml.nvmlDeviceGetTemperature(self._handle, 0))
        except Exception as e:
            logger.debug(f"[{self._device_name}] Cannot read temperature: {e}")

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            util_pct = float(util.gpu)
        except Exception as e:
            logger.debug(f"[{self._device_name}] Cannot read utilization: {e}")

        try:
            freq_mhz = float(pynvml.nvmlDeviceGetClockInfo(self._handle, 0))
        except Exception as e:
            logger.debug(f"[{self._device_name}] Cannot read frequency: {e}")

        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            mem_used_mb = mem_info.used / (1024 ** 2)
            mem_total_mb = mem_info.total / (1024 ** 2)
        except Exception as e:
            logger.debug(f"[{self._device_name}] Cannot read memory: {e}")

        # Attempt to read SM (Streaming Multiprocessor) utilization if supported
        try:
            # NVIDIA Kepler and later support SM clock
            sm_clock = pynvml.nvmlDeviceGetClockInfo(self._handle, 1)  # 1 = SM clock
            max_sm_clock = pynvml.nvmlDeviceGetMaxClockInfo(self._handle, 1)
            if max_sm_clock > 0:
                sm_util_pct = (sm_clock / max_sm_clock) * 100.0
        except Exception as e:
            logger.debug(f"[{self._device_name}] Cannot read SM utilization: {e}")

        metadata = {
            "frequency_mhz": freq_mhz,
            "memory_used_mb": mem_used_mb,
            "memory_total_mb": mem_total_mb,
        }
        if sm_util_pct is not None:
            metadata["sm_utilization_pct"] = sm_util_pct

        return EnergyReading(
            timestamp=time.time(),
            power_w=power_w,
            temperature_c=temp_c,
            utilization_pct=util_pct,
            metadata=metadata,
        )

    def _shutdown(self) -> None:
        """Shutdown NVML and restore original power limit if modified."""
        if self._original_power_limit_w is not None:
            self.restore_defaults()

        try:
            pynvml.nvmlShutdown()
            logger.debug(f"[{self._device_name}] NVML shutdown")
        except Exception as e:
            logger.warning(f"[{self._device_name}] NVML shutdown error: {e}")

        if self._signal_handler_installed:
            try:
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                self._signal_handler_installed = False
            except Exception:
                pass  # Signals unsafe in threads

    # -----------------------------------------------------------------------
    # Power management (optional in base class)
    # -----------------------------------------------------------------------

    def set_power_limit(self, limit_w: int) -> bool:
        """Set GPU power limit.

        Only works when mode=ACTIVE. Stores the original limit so it can be
        restored later.

        Parameters
        ----------
        limit_w : int
            Power limit in watts. Usually between min_cap_w and max_cap_w.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if self._mode != MonitorMode.ACTIVE or not self._handle:
            logger.warning(
                f"[{self._device_name}] Cannot set power limit in {self._mode.value} mode"
            )
            return False

        try:
            if self._original_power_limit_w is None:
                # Capture original on first write
                try:
                    self._original_power_limit_w = (
                        pynvml.nvmlDeviceGetPowerManagementLimit(self._handle) // 1000
                    )
                except Exception:
                    logger.debug(
                        f"[{self._device_name}] Could not read original power limit"
                    )

            limit_mw = limit_w * 1000
            pynvml.nvmlDeviceSetPowerManagementLimit(self._handle, limit_mw)
            logger.info(f"[{self._device_name}] Set power limit to {limit_w} W")
            return True
        except Exception as e:
            logger.error(
                f"[{self._device_name}] Failed to set power limit: {e}"
            )
            return False

    def get_power_limit(self) -> int | None:
        """Get current GPU power limit.

        Returns
        -------
        int or None
            Power limit in watts, or None if unavailable.
        """
        if not self._handle:
            return None

        try:
            limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(self._handle)
            return limit_mw // 1000
        except Exception as e:
            logger.debug(f"[{self._device_name}] Cannot read power limit: {e}")
            return None

    def restore_defaults(self) -> bool:
        """Restore original GPU power limit if one was captured.

        Returns
        -------
        bool
            True if successful or no limit was modified, False on error.
        """
        if self._original_power_limit_w is None:
            logger.debug(f"[{self._device_name}] No original limit to restore")
            return True

        return self.set_power_limit(self._original_power_limit_w)

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def _probe_read_capability(self) -> bool:
        """Test whether we can read power and temperature.

        Returns
        -------
        bool
            True if at least one metric is readable.
        """
        if not self._handle:
            return False

        can_read_power = False
        can_read_temp = False

        try:
            pynvml.nvmlDeviceGetPowerUsage(self._handle)
            can_read_power = True
        except Exception:
            pass

        try:
            pynvml.nvmlDeviceGetTemperature(self._handle, 0)
            can_read_temp = True
        except Exception:
            pass

        return can_read_power or can_read_temp

    def _probe_write_capability(self) -> bool:
        """Test whether we can write power limits (requires root/elevated).

        Returns
        -------
        bool
            True if power limits can be modified.
        """
        if not self._handle:
            return False

        try:
            # Try a no-op: read the limit, then set it to the same value
            current_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(
                self._handle
            )
            pynvml.nvmlDeviceSetPowerManagementLimit(self._handle, current_limit_mw)
            return True
        except Exception:
            return False

    def _register_signal_handlers(self) -> None:
        """Register SIGINT/SIGTERM handlers to restore power limits on shutdown.

        Called only when mode=ACTIVE. Handlers call restore_defaults() before
        exiting, ensuring the GPU returns to factory settings even on abrupt
        termination.
        """
        def _signal_handler(signum, frame):  # noqa: ANN001
            logger.info(f"[{self._device_name}] Signal {signum} received; restoring")
            self.restore_defaults()
            raise SystemExit(0)

        try:
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
            self._signal_handler_installed = True
            logger.debug(f"[{self._device_name}] Signal handlers registered")
        except Exception as e:
            # Signals are unsafe in multi-threaded contexts
            logger.warning(f"[{self._device_name}] Cannot register signals: {e}")
