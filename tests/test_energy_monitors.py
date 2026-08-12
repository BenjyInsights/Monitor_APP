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
tests/test_energy_monitors.py — Unit tests for energy monitor abstractions.

Tests BaseEnergyMonitor interface, NvidiaGpuMonitor, and IntelCpuMonitor
using unittest.mock to simulate hardware access without requiring actual
GPU/CPU hardware or elevated privileges.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, mock_open
import time

from monitor_app.monitor.base_energy_monitor import (
    BaseEnergyMonitor,
    EnergyReading,
    MonitorMode,
    MonitorError,
)
from monitor_app.monitor.nvidia_gpu_monitor import NvidiaGpuMonitor
from monitor_app.monitor.intel_cpu_monitor import IntelCpuMonitor


class TestBaseEnergyMonitor(unittest.TestCase):
    """Test the BaseEnergyMonitor abstract base class."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """BaseEnergyMonitor should not be instantiable directly."""
        with self.assertRaises(TypeError):
            BaseEnergyMonitor()  # type: ignore

    def test_subclass_must_implement_abstract_methods(self) -> None:
        """Subclass must implement all abstract methods."""
        class IncompleteMonitor(BaseEnergyMonitor):
            def _initialize_hardware(self) -> None:
                pass
            # Missing _read_metrics and _shutdown

        with self.assertRaises(TypeError):
            IncompleteMonitor()  # type: ignore

    def test_concrete_subclass_can_be_instantiated(self) -> None:
        """Concrete subclass implementing all abstract methods should work."""
        class ConcreteMonitor(BaseEnergyMonitor):
            def _initialize_hardware(self) -> None:
                self._mode = MonitorMode.ACTIVE
            def _read_metrics(self) -> EnergyReading:
                return EnergyReading(timestamp=time.time(), power_w=100.0)
            def _shutdown(self) -> None:
                pass

        monitor = ConcreteMonitor()
        self.assertIsNotNone(monitor)
        self.assertEqual(monitor.mode, MonitorMode.UNAVAILABLE)
        self.assertFalse(monitor.is_initialized)

    def test_context_manager_lifecycle(self) -> None:
        """Test __enter__ and __exit__ for context manager support."""
        class SimpleMonitor(BaseEnergyMonitor):
            def __init__(self) -> None:
                super().__init__()
                self.init_called = False
                self.shutdown_called = False

            def _initialize_hardware(self) -> None:
                self.init_called = True
                self._mode = MonitorMode.READ_ONLY

            def _read_metrics(self) -> EnergyReading:
                return EnergyReading(timestamp=time.time(), power_w=50.0)

            def _shutdown(self) -> None:
                self.shutdown_called = True

        monitor = SimpleMonitor()
        self.assertFalse(monitor.is_initialized)

        with monitor as m:
            self.assertTrue(m.is_initialized)
            self.assertTrue(m.init_called)

        self.assertTrue(monitor.shutdown_called)


class TestNvidiaGpuMonitor(unittest.TestCase):
    """Test NvidiaGpuMonitor with mocked NVML."""

    @patch("monitor_app.monitor.nvidia_gpu_monitor.pynvml")
    @patch("monitor_app.monitor.nvidia_gpu_monitor._PYNVML_AVAILABLE", True)
    def test_initialization_active_mode(self, mock_pynvml: Mock) -> None:
        """Test successful GPU monitor initialization in ACTIVE mode."""
        # Setup mocks
        mock_handle = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 250000  # mW
        mock_pynvml.nvmlDeviceSetPowerManagementLimit.return_value = None

        monitor = NvidiaGpuMonitor(gpu_index=0)
        self.assertEqual(monitor.mode, MonitorMode.UNAVAILABLE)  # Before init

        monitor.initialize()

        self.assertTrue(monitor.is_initialized)
        self.assertEqual(monitor.mode, MonitorMode.ACTIVE)
        mock_pynvml.nvmlInit.assert_called_once()

    @patch("monitor_app.monitor.nvidia_gpu_monitor.pynvml")
    @patch("monitor_app.monitor.nvidia_gpu_monitor._PYNVML_AVAILABLE", True)
    def test_initialization_read_only_mode(self, mock_pynvml: Mock) -> None:
        """Test initialization in READ_ONLY mode (no write permission)."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 250000
        # Simulate permission error on write
        mock_pynvml.nvmlDeviceSetPowerManagementLimit.side_effect = \
            RuntimeError("NVMLError_NoPermission")

        monitor = NvidiaGpuMonitor(gpu_index=0)
        monitor.initialize()

        self.assertTrue(monitor.is_initialized)
        self.assertEqual(monitor.mode, MonitorMode.READ_ONLY)

    @patch("monitor_app.monitor.nvidia_gpu_monitor.pynvml")
    @patch("monitor_app.monitor.nvidia_gpu_monitor._PYNVML_AVAILABLE", True)
    def test_read_metrics(self, mock_pynvml: Mock) -> None:
        """Test reading GPU metrics."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 250000
        mock_pynvml.nvmlDeviceSetPowerManagementLimit.return_value = None

        # Mock metric readings
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150000  # mW
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 65  # °C
        util_obj = MagicMock()
        util_obj.gpu = 75  # %
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = util_obj
        mock_pynvml.nvmlDeviceGetClockInfo.return_value = 1980  # MHz
        mem_obj = MagicMock()
        mem_obj.used = 8 * 1024 ** 3  # 8 GB
        mem_obj.total = 24 * 1024 ** 3  # 24 GB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_obj

        monitor = NvidiaGpuMonitor(gpu_index=0)
        monitor.initialize()
        reading = monitor.read_metrics()

        self.assertIsNotNone(reading)
        self.assertAlmostEqual(reading.power_w, 150.0, places=1)
        self.assertEqual(reading.temperature_c, 65)
        self.assertEqual(reading.utilization_pct, 75)
        self.assertIn("frequency_mhz", reading.metadata)
        self.assertEqual(reading.metadata["frequency_mhz"], 1980)

    @patch("monitor_app.monitor.nvidia_gpu_monitor.pynvml")
    @patch("monitor_app.monitor.nvidia_gpu_monitor._PYNVML_AVAILABLE", True)
    def test_set_power_limit_active_mode(self, mock_pynvml: Mock) -> None:
        """Test setting power limit in ACTIVE mode."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 250000
        mock_pynvml.nvmlDeviceSetPowerManagementLimit.return_value = None

        monitor = NvidiaGpuMonitor(gpu_index=0)
        monitor.initialize()

        success = monitor.set_power_limit(200)
        self.assertTrue(success)
        mock_pynvml.nvmlDeviceSetPowerManagementLimit.assert_called()

    def test_set_power_limit_unavailable_mode(self) -> None:
        """Test set_power_limit returns False when unavailable."""
        monitor = NvidiaGpuMonitor(gpu_index=0)
        monitor._mode = MonitorMode.UNAVAILABLE

        success = monitor.set_power_limit(200)
        self.assertFalse(success)


class TestIntelCpuMonitor(unittest.TestCase):
    """Test IntelCpuMonitor with mocked RAPL sysfs."""

    @patch("os.path.isdir")
    @patch("os.path.isfile")
    @patch("builtins.open", new_callable=mock_open, read_data="50000000\n")
    def test_initialization_success(
        self,
        mock_file: Mock,
        mock_isfile: Mock,
        mock_isdir: Mock,
    ) -> None:
        """Test successful CPU monitor initialization."""
        mock_isdir.return_value = True
        mock_isfile.return_value = True

        monitor = IntelCpuMonitor(package_index=0)
        monitor.initialize()

        self.assertTrue(monitor.is_initialized)
        self.assertEqual(monitor.mode, MonitorMode.READ_ONLY)

    @patch("os.path.isdir")
    def test_initialization_rapl_not_found(self, mock_isdir: Mock) -> None:
        """Test initialization when RAPL sysfs not available."""
        mock_isdir.return_value = False

        monitor = IntelCpuMonitor(package_index=0)
        monitor.initialize()

        self.assertTrue(monitor.is_initialized)
        self.assertEqual(monitor.mode, MonitorMode.UNAVAILABLE)

    @patch("os.path.isdir")
    @patch("os.path.isfile")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_read_metrics(
        self,
        mock_file: Mock,
        mock_makedirs: Mock,
        mock_isfile: Mock,
        mock_isdir: Mock,
    ) -> None:
        """Test reading CPU energy metrics."""
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        # Energy values in microjoules. initialize() ya consume una lectura para
        # fijar la línea base, de ahí que hagan falta tres valores y no dos.
        mock_file.return_value.__enter__.return_value.read.side_effect = [
            "45000000\n",  # Lectura de initialize() (línea base)
            "50000000\n",  # Primera read_metrics()
            "55000000\n",  # Segunda read_metrics() (delta = 5 J sobre ~0,01 s)
        ]

        monitor = IntelCpuMonitor(package_index=0)
        monitor.initialize()

        # First read to set baseline
        reading1 = monitor.read_metrics()
        time.sleep(0.01)  # Small delay

        # Second read to compute power
        reading2 = monitor.read_metrics()

        self.assertIsNotNone(reading2)
        self.assertIsNotNone(reading2.power_w)
        self.assertGreater(reading2.power_w, 0)
        self.assertIsNotNone(reading2.energy_j)

    @patch("os.path.isdir")
    def test_read_metrics_unavailable(self, mock_isdir: Mock) -> None:
        """Test read_metrics when monitor unavailable.

        El monitor debe estar inicializado: read_metrics() distingue entre "no
        inicializado" (RuntimeError) y "hardware no disponible" (None). Sin RAPL,
        initialize() deja el modo en UNAVAILABLE y la lectura devuelve None.
        """
        mock_isdir.return_value = False

        monitor = IntelCpuMonitor(package_index=0)
        monitor.initialize()
        self.assertEqual(monitor.mode, MonitorMode.UNAVAILABLE)

        reading = monitor.read_metrics()
        self.assertIsNone(reading)


class TestEnergyReading(unittest.TestCase):
    """Test EnergyReading dataclass."""

    def test_energy_reading_creation(self) -> None:
        """Test creating an EnergyReading."""
        ts = time.time()
        reading = EnergyReading(
            timestamp=ts,
            power_w=100.0,
            temperature_c=65.0,
            utilization_pct=75.0,
            metadata={"frequency_mhz": 1980},
        )

        self.assertEqual(reading.timestamp, ts)
        self.assertEqual(reading.power_w, 100.0)
        self.assertEqual(reading.temperature_c, 65.0)
        self.assertEqual(reading.utilization_pct, 75.0)
        self.assertEqual(reading.metadata["frequency_mhz"], 1980)

    def test_energy_reading_partial(self) -> None:
        """Test EnergyReading with optional fields as None."""
        reading = EnergyReading(timestamp=time.time(), power_w=50.0)

        self.assertIsNotNone(reading.timestamp)
        self.assertEqual(reading.power_w, 50.0)
        self.assertIsNone(reading.temperature_c)
        self.assertIsNone(reading.utilization_pct)
        self.assertIsNone(reading.energy_j)
        self.assertIsNone(reading.metadata)


if __name__ == "__main__":
    unittest.main()
