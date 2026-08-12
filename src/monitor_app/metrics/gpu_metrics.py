# metrics/GPU_Metrics/gpu_metrics.py #
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

import psutil
import pynvml

from .base import BaseMetric
from ..config.metrics_units import MetricsUnits
from ..utils.carbon_emissions import CarbonEmissions

class NVIDIAGPUMetrics(BaseMetric):
    """
    A class to gather and manage metrics for NVIDIA GPUs.

    This class utilizes the NVIDIA Management Library (NVML) to collect various
    metrics from NVIDIA GPUs, such as temperature, fan speed, power usage, and more.
    It also estimates carbon emissions based on power usage.
    """
    def __init__(self, metrics_units:MetricsUnits, process:psutil.Process, monitor_interval_ms:int):
        """
        Initializes the NVIDIAGPUMetrics with the given metrics units configuration, process,
        and monitoring interval.

        Args:
            metrics_units (MetricsUnits): The instance of MetricsUnits holding the units configuration.
            process (psutil.Process): The process to monitor.
            monitor_interval_ms (int): The monitoring interval in milliseconds.

        Raises:
            ValueError: If metrics_units is None.
        """
        if metrics_units is None:
            raise ValueError("A valid MetricsUnits instance is requiered")
       
        super().__init__(metrics_units, process, monitor_interval_ms)
        self._temperature_unit, self._temperature_factor = metrics_units.temperature
        self._frequency_unit, self._frequency_factor = metrics_units.frequency
        self._memory_unit, self._memory_factor = metrics_units.memory
        self._ce_emissions = CarbonEmissions().get_ce_data()
        self._total_emissions_by_gpu = {}
        self._total_energy_by_gpu = {}

        # PID of the process being monitored (used to isolate per-process metrics)
        self._monitored_pid = process.pid
        # Per-GPU accumulated process energy (Wh)
        self._proc_total_energy_by_gpu: dict = {}
        # Per-GPU last-seen timestamp for nvmlDeviceGetProcessUtilization (µs)
        self._proc_util_last_ts: dict = {}

        try:
            pynvml.nvmlInit()
            self._nvidia_gpu_detected = True
        except Exception:
            self._nvidia_gpu_detected = False

        # Probe whether nvmlDeviceGetProcessUtilization is supported (NVML 7.0+).
        # If not, we fall back to the VRAM-fraction heuristic.
        self._proc_util_supported = False
        if self._nvidia_gpu_detected:
            try:
                _h = pynvml.nvmlDeviceGetHandleByIndex(0)
                pynvml.nvmlDeviceGetProcessUtilization(_h, 0)
                self._proc_util_supported = True
            except Exception:
                self._proc_util_supported = False


    def get_header(self):
        """
        Gets the header for the metrics of the NVIDIA GPUs.

        Returns:
            str: The header for the metrics of the NVIDIA GPUs.
        """
        return "NVIDIA GPU Info"
    
    def get_value(self) -> dict:
        """
        Gets the metrics for the NVIDIA GPUs.
        
        Returns:
            dict: A dictionary where each key is the GPU index and the value is a dictionary containing the metrics for that GPU.
        """
        if not self._nvidia_gpu_detected:
            return "No compatible NVIDIA GPU detected."
        
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                return "No NVIDIA GPUs can be found."
            
            gpus_info={}
            for gpu_index in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                gpu_data = self.get_data(handle) 
                gpus_info[f"GPU {int(gpu_index)}"] = gpu_data

            return gpus_info
                    
        except Exception as e:
            return f"Error retrieving GPU info: {e}"


    def get_data(self, handle:pynvml.nvmlDeviceGetHandleByIndex) -> dict:
        """
        Gets the metrics for the specified NVIDIA GPU.

        Args:
            handle (pynvml.nvmlDeviceGetHandleByIndex): The handle of the GPU.

        Returns:
            dict: A dictionary containing the metrics for the specified GPU.
        """
        gpu_info = {}

        # GPU Info #
        self.get_name(handle, gpu_info)
        self.get_memory_stats(handle, gpu_info)
        self.get_temperature_stats(handle, gpu_info)
        self.get_fan_stats(handle, gpu_info)
        self.get_power_usage(handle, gpu_info)
        self.get_core_mem_usage(handle, gpu_info)
        self.get_core_mem_clocks(handle, gpu_info)

        # GPU Processes running #
        self.get_gpu_processes(handle, gpu_info)

        # Process-specific metrics (VRAM, SM utilization, proportional power) #
        self.get_process_metrics(handle, gpu_info)

        return gpu_info

    def get_name(self, handle:pynvml.nvmlDeviceGetHandleByIndex, gpu_data:dict) -> None:
        """
        Retrieves the name of the NVIDIA GPU and updates the provided dictionary with the GPU name.

        Args:
            handle (pynvml.nvmlDeviceGetHandleByIndex): The handle of the GPU.
            gpu_data (dict): The dictionary to update with the GPU name.

        Returns:
            None
        """
        name_bytes = pynvml.nvmlDeviceGetName(handle)
        name = name_bytes.decode('utf-8', errors='replace') if isinstance(name_bytes, bytes) else name_bytes
        gpu_data["Name"] = name
    
    def get_memory_stats(self, handle:pynvml.nvmlDeviceGetHandleByIndex, gpu_data:dict) -> None:
        """
        Retrieves the memory stats of the NVIDIA GPU and updates the provided dictionary with the memory stats.

        Args:
            handle (pynvml.nvmlDeviceGetHandleByIndex): The handle of the GPU.
            gpu_data (dict): The dictionary to update with the memory stats.

        Returns:
            None
        """
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle) # B
        gpu_data[f"Memory Total ({self._memory_unit})"] = float(round(mem_info.total * self._memory_factor, 2))
        gpu_data[f"Memory System Used ({self._memory_unit})"] = float(round(mem_info.used * self._memory_factor, 2))
        gpu_data[f"Memory Free ({self._memory_unit})"] = float(round(mem_info.free * self._memory_factor, 2))


    def get_temperature_stats(self, handle:pynvml.nvmlDeviceGetHandleByIndex, gpu_data:dict) -> None:
        """
        Retrieves the temperature stats of the NVIDIA GPU and updates the provided dictionary with the temperature stats.

        Args:
            handle (pynvml.nvmlDeviceGetHandleByIndex): The handle of the GPU.
            gpu_data (dict): The dictionary to update with the temperature stats.

        Returns:
            None
        """
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU) # ºC
        gpu_data[f"Temperature ({self._temperature_unit})"] = float(round(self._temperature_factor(temp), 1))

    def get_fan_stats(self, handle:pynvml.nvmlDeviceGetHandleByIndex, gpu_data:dict) -> None:
        """
        Retrieves the fan stats of the NVIDIA GPU and updates the provided dictionary with the fan stats.

        Args:
            handle (pynvml.nvmlDeviceGetHandleByIndex): The handle of the GPU.
            gpu_data (dict): The dictionary to update with the fan stats.

        Returns:
            None
        """
        fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle) # %
        fan_rpm = pynvml.nvmlDeviceGetFanSpeedRPM(handle) # RPM
        gpu_data["Fan Speed (%)"] = int(round(fan_speed, 1))
        gpu_data["Fan Speed (RPM)"] = int(round(fan_rpm, 1))

    
    def get_power_usage(self, handle:pynvml.nvmlDeviceGetHandleByIndex, gpu_data:dict) -> None:
        """
        Retrieves the power usage stats of the NVIDIA GPU and updates the provided dictionary with the power usage stats.

        Args:
            handle (pynvml.nvmlDeviceGetHandleByIndex): The handle of the GPU.
            gpu_data (dict): The dictionary to update with the power usage stats.

        Returns:
            None
        """
        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) # mW
        power_usage_w = power_usage / 1000
        power_usage_actual = power_usage_w * (self._monitor_interval_ms / 3600000)
        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) # mW 
        pstate = pynvml.nvmlDeviceGetPerformanceState(handle) # [0, 15]
        gpu_data["Power Usage (W)"] = float(round(power_usage_w, 2))
        gpu_data["Power Usage Interval (Wh)"] = float(round(power_usage_actual, 5))
        gpu_data["Power Limit (W)"] = int(round(power_limit / 1000, 1))
        # Accumulate Power Consumption #
        gpu_index = pynvml.nvmlDeviceGetIndex(handle)
        gpu_id = f"GPU {gpu_index}"
        self._total_energy_by_gpu[gpu_id] = self._total_energy_by_gpu.get(gpu_id, 0.0) + power_usage_actual
        gpu_data["Total Consumption (W)"] = float(round(self._total_energy_by_gpu[gpu_id], 2))
        
        gpu_data["P-State (P0: Max Performance, P15: Min Performance)"] = f"P{int(pstate)}"
        self.get_carbon_emissions(power_usage_actual, gpu_data)

        #  Accumulate Carbon Emissions #
        emissions = gpu_data.get("Carbon Emissions Estimated (g CO2)", {})
        if gpu_id not in self._total_emissions_by_gpu:
            self._total_emissions_by_gpu[gpu_id] = {}

        for continent, emission in emissions.items():
            if emission is not None:
                self._total_emissions_by_gpu[gpu_id][continent] = (
                    self._total_emissions_by_gpu[gpu_id].get(continent, 0.0) + emission
                )

        gpu_data["Total Carbon Emissions (g CO2)"] = {
            continent: float(round(value, 5))
            for continent, value in self._total_emissions_by_gpu[gpu_id].items()
        }
                
    def get_carbon_emissions(self, power_usage_actual:float, gpu_data:dict) -> None:
        """
        Calculates the estimated carbon emissions for the NVIDIA GPU based on its power usage
        and updates the provided dictionary with the emissions data.

        Args:
            power_usage (float): The power usage of the GPU in milliwatts.
            gpu_data (dict): The dictionary to update with the estimated carbon emissions data.

        Returns:
            None: This function updates the gpu_data dictionary in place.
        """
        emissions = {}
        gpu_energy_kwh = power_usage_actual / 1000 # convert to kw
        for continent, factor in self._ce_emissions.items():
            emissions[continent] = None if factor is None else float(round(gpu_energy_kwh * factor, 5))
        gpu_data["Carbon Emissions Estimated (g CO2)"] = emissions
    
    def get_core_mem_usage(self, handle:pynvml.nvmlDeviceGetHandleByIndex, gpu_data:dict) -> None:
        """
        Retrieves the utilization rates of the NVIDIA GPU and updates the provided dictionary with the utilization rates.

        Args:
            handle (pynvml.nvmlDeviceGetHandleByIndex): The handle of the GPU.
            gpu_data (dict): The dictionary to update with the utilization rates.

        Returns:
            None
        """
        gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_data["Core Usage (%)"] = float(round(gpu_usage.gpu, 2))
        gpu_data["Memory Usage (%)"] = float(round(gpu_usage.memory, 2))

    
    def get_core_mem_clocks(self, handle:pynvml.nvmlDeviceGetHandleByIndex, gpu_data:dict) -> None:
        """
        Retrieves the core and memory clock rates of the NVIDIA GPU and updates the provided dictionary with the clock rates.

        Args:
            handle (pynvml.nvmlDeviceGetHandleByIndex): The handle of the GPU.
            gpu_data (dict): The dictionary to update with the clock rates.

        Returns:
            None
        """
        core_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        gpu_data[f"Core Clock ({self._frequency_unit})"] = round(float(core_clock * self._frequency_factor), 2)
        gpu_data[f"Memory Clock ({self._frequency_unit})"] = round(float(mem_clock) * self._frequency_factor, 2)


    def get_gpu_processes(self, handle:pynvml.nvmlDeviceGetHandleByIndex, gpu_data:dict) -> None:
        """
        Retrieves the list of compute processes running on the NVIDIA GPU and updates the provided dictionary with the list of processes.

        Args:
            handle (pynvml.nvmlDeviceGetHandleByIndex): The handle of the GPU.
            gpu_data (dict): The dictionary to update with the list of processes.

        Returns:
            None
        """
        proc_list = []

        try:
            gpu_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except Exception:
            gpu_data["GPU Processes Running"] = "Cannot get GPU processes running."
            return
        
        if not gpu_processes:
             gpu_data["GPU Processes Running"] = "No compute processes in GPU running."
             return
        
        for p in gpu_processes:
            try:
                proc = psutil.Process(p.pid)
                proc_info = {
                    "PID": p.pid,
                    "Name": proc.name(),
                    "cmdline": proc.cmdline(),
                    f"Process GPU Memory Usage ({self._memory_unit})": float(round(p.usedGpuMemory * self._memory_factor, 2)) if p.usedGpuMemory else 0.0,
                }
            except Exception as ex:
                proc_info = {"PID": p.pid, "error": str(ex)}

            proc_list.append(proc_info)

        gpu_data["GPU Processes Running"] = proc_list


    def get_process_metrics(self, handle: pynvml.nvmlDeviceGetHandleByIndex, gpu_data: dict) -> None:
        """
        Adds process-specific GPU metrics to gpu_data for the monitored process:

          - "Process VRAM (<unit>)"                    — memory used by this process only
          - "Process SM Util (%)"                      — SM utilisation of this process
                                                         (only present when nvmlDeviceGetProcessUtilization
                                                         is supported; absent when using the VRAM fallback)
          - "Process Power Usage (W)"                  — total GPU power × attribution ratio
          - "Process Power Usage Interval (Wh)"        — same, over the monitoring interval
          - "Process Total Consumption (Wh)"           — accumulated process energy
          - "Process Carbon Emissions Estimated (g CO2)" — per-continent CO2 for process energy

        Attribution ratio is computed in order of preference:
          1. SM ratio  — process_smUtil / sum(all_smUtil)   [most accurate]
          2. VRAM ratio — process_vram  / sum(all_vram)     [fallback, less accurate]

        All existing total-GPU keys are left unchanged so that logs generated
        before this change remain fully compatible.

        Returns without adding any key if the monitored process is not found
        on this GPU (e.g. running on a different GPU).
        """
        try:
            compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except Exception:
            return

        my_proc = next(
            (p for p in compute_procs if p.pid == self._monitored_pid), None
        )
        if my_proc is None:
            return  # monitored process is not using this GPU

        # ── P2: Process VRAM ─────────────────────────────────────────────────
        proc_vram_bytes = my_proc.usedGpuMemory or 0
        gpu_data[f"Process VRAM ({self._memory_unit})"] = float(
            round(proc_vram_bytes * self._memory_factor, 2)
        )

        # ── P1 / P4: Attribution ratio (SM preferred, VRAM as fallback) ─────
        ratio: float | None = None

        if self._proc_util_supported:
            try:
                gpu_index = pynvml.nvmlDeviceGetIndex(handle)
                last_ts   = self._proc_util_last_ts.get(gpu_index, 0)
                samples   = pynvml.nvmlDeviceGetProcessUtilization(handle, last_ts)
                if samples:
                    self._proc_util_last_ts[gpu_index] = max(s.timeStamp for s in samples)
                    total_sm = sum(s.smUtil for s in samples)
                    my_sm    = next(
                        (s.smUtil for s in samples if s.pid == self._monitored_pid), None
                    )
                    if my_sm is not None:
                        gpu_data["Process SM Util (%)"] = int(my_sm)
                        ratio = (my_sm / total_sm) if total_sm > 0 else 0.0
            except Exception:
                # Permanently disable to avoid repeated failures
                self._proc_util_supported = False

        # P4: VRAM fallback when SM data is unavailable
        if ratio is None:
            total_vram = sum(
                (p.usedGpuMemory or 0) for p in compute_procs
            )
            ratio = (proc_vram_bytes / total_vram) if total_vram > 0 else 1.0

        # ── P1: Process power and energy ─────────────────────────────────────
        total_power_w   = gpu_data.get("Power Usage (W)", 0.0)
        total_energy_wh = gpu_data.get("Power Usage Interval (Wh)", 0.0)

        proc_power_w   = float(round(total_power_w   * ratio, 2))
        proc_energy_wh = float(round(total_energy_wh * ratio, 5))

        gpu_data["Process Power Usage (W)"]           = proc_power_w
        gpu_data["Process Power Usage Interval (Wh)"] = proc_energy_wh

        # Accumulate process-specific energy
        gpu_index = pynvml.nvmlDeviceGetIndex(handle)
        gpu_id    = f"GPU {gpu_index}"
        self._proc_total_energy_by_gpu[gpu_id] = (
            self._proc_total_energy_by_gpu.get(gpu_id, 0.0) + proc_energy_wh
        )
        gpu_data["Process Total Consumption (Wh)"] = float(
            round(self._proc_total_energy_by_gpu[gpu_id], 5)
        )

        # P1: Per-continent CO2 for the process fraction only
        gpu_proc_energy_kwh = proc_energy_wh / 1000.0
        proc_co2: dict = {}
        for continent, factor in self._ce_emissions.items():
            proc_co2[continent] = (
                None if factor is None
                else float(round(gpu_proc_energy_kwh * factor, 5))
            )
        gpu_data["Process Carbon Emissions Estimated (g CO2)"] = proc_co2
