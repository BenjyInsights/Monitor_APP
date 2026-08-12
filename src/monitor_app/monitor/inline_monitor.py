# src/monitor/inline_monitor.py #
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

import traceback
import psutil
import time
import json
import sys
import os

from functools import wraps
from datetime import datetime
from multiprocessing import Process, Event
from contextlib import AbstractContextManager

from ..config.metrics_units import MetricsUnits
from ..config.metrics_list import MetricsList
from ..config.parse_monitor import ParseMonitorArguments

class BaseMonitor:
    """
    Abstract base class for all monitor classes.

    This class provides the basic structure and methods for classes that are
    responsible for monitoring a process. It provides a start method to start
    the monitoring process and a stop method to stop the monitoring process.

    Attributes:
        context (str): A string representing the context of the monitoring.
        interval (float): The monitoring interval in seconds.
        log_file_path (str): The path to the log file for storing monitoring data.
        frequency_unit (str): The unit for frequency measurement.
        memory_unit (str): The unit for memory measurement.
        temperature_unit (str): The unit for temperature measurement.
        stop_event (Event): An event to stop the monitoring process.
        monitored_pid (int): The process ID of the monitored process.
        monitor_proc (Process): The process responsible for monitoring the process.
    """
    def __init__(self, context, interval, log_file_path, frequency_unit, memory_unit, temperature_unit):
        validated = ParseMonitorArguments(
            context=context,
            interval=interval,
            log_file_path=log_file_path,
            frequency_unit=frequency_unit,
            memory_unit=memory_unit,
            temperature_unit=temperature_unit
        )

        self.context = validated.context
        self.interval = validated.interval
        self.log_file_path = validated.log_file_path
        self.log_file_path = self.log_file_path + ".ndjson"
        self.frequency_unit = validated.frequency_unit
        self.memory_unit = validated.memory_unit
        self.temperature_unit = validated.temperature_unit

        self.stop_event = Event()
        self.monitored_pid = os.getpid()
        self.monitor_proc = None

    def start(self):
        """
        Start the monitoring process.

        Creates the necessary directories for the log file, initializes the
        monitoring process with the specified parameters, and starts the process
        to periodically log metrics for the monitored process.
        """
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        self.monitor_proc = Process(
            target=_monitor_worker,
            args=(
                self.context,
                self.interval,
                self.log_file_path,
                self.frequency_unit,
                self.memory_unit,
                self.temperature_unit,
                self.stop_event,
                self.monitored_pid
            )
        )
        self.monitor_proc.start()
        # Block until the worker has created and initialised the log file,
        # so any EpochTracker appends after start() cannot be wiped by the
        # worker's "w"-mode open (which would truncate the file).
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if os.path.isfile(self.log_file_path):
                break
            time.sleep(0.005)

    def stop(self):
        """
        Stop the monitoring process.

        If the monitoring process is running, this method sets the stop event
        and waits for the process to terminate.
        """
        if self.monitor_proc is not None:
            self.stop_event.set()
            self.monitor_proc.join()


def _monitor_worker(context:str, interval:int, json_file_path:str, frequency_unit:str, memory_unit:str, temperature_unit:str, stop_event, monitored_pid:int) -> None:
    """
    Worker function for monitoring a process and logging metrics.

    This function runs in a separate process and periodically logs the
    specified metrics for a given process ID to a JSON file. It continues
    logging until the provided stop event is set.

    Args:
        context (str): Description of the monitoring context.
        interval (int): The monitoring interval in seconds.
        json_file_path (str): Path to the JSON file where metrics are logged.
        frequency_unit (str): Unit for frequency metrics (e.g., "MHz").
        memory_unit (str): Unit for memory metrics (e.g., "MB").
        temperature_unit (str): Unit for temperature metrics (e.g., "ºC").
        stop_event (Event): Event to signal the worker to stop.
        monitored_pid (int): Process ID to be monitored.

    Raises:
        Exception: If there is an issue with retrieving metric values.
    """
    process = psutil.Process(monitored_pid)
    process.cpu_percent(interval=None) # Dummy call
    metrics_units = MetricsUnits(frequency_unit, memory_unit, temperature_unit)
    metrics_factory = MetricsList()
    metrics = metrics_factory.get_metrics(metrics_units, process, int(interval * 1000))

    with open(json_file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "context": context,
            "monitoring_pid": monitored_pid,
            "frequency": metrics_units.frequency[0],
            "memory": metrics_units.memory[0],
            "temperature": metrics_units.temperature[0],
        }, ensure_ascii=False) + "\n")

        f.write(json.dumps({"Process started at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) + "\n")
        f.flush()

        time.sleep(0.1)  # CPU warmup for cpu_percent first reading

        while not stop_event.is_set():
            entry = {"Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            for metric in metrics:
                try:
                    entry[metric.get_header()] = metric.get_value()
                except Exception:
                    entry[metric.get_header()] = None
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(interval)

        f.write(json.dumps({
            "Process finished at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }) + "\n")
        f.flush()


# DECORATOR #
def inline_monitor(context="", interval=1.0, log_file_path="logs/function_monitor.ndjson",
                   frequency_unit="MHz", memory_unit="MB", temperature_unit="ºC"):
    """
    Inline decorator for monitoring a function's execution.

    Args:
        context (str): Description of the monitoring context.
        interval (int): The monitoring interval in seconds.
        log_file_path (str): Path to the JSON file where metrics are logged.
        frequency_unit (str): Unit for frequency metrics ("GHz" or"MHz").
        memory_unit (str): Unit for memory metrics ("B", "KB", "MB" or "GB").
        temperature_unit (str): Unit for temperature metrics ("ºC" or "ºF").

    Returns:
        function: Decorated function that starts and stops a monitor instance
                 around its execution.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                monitor = BaseMonitor(context, interval, log_file_path,
                                      frequency_unit, memory_unit, temperature_unit)
                monitor.start()
                try:
                    return func(*args, **kwargs)
                finally:
                    monitor.stop()
            except ValueError as e:
                # Safe traceback extraction #
                tb = traceback.extract_stack()
                last_user_frame = next(
                    (frame for frame in reversed(tb) if "inline_monitor.py" not in frame.filename),
                    tb[-1]  # Fallback #
                )
                print(f"[Error] [inline_monitor] {e}")
                print(f"File: {last_user_frame.filename}, Line: {last_user_frame.lineno}\n")
                sys.exit(1)
        return wrapper
    return decorator


# CONTEXT MANAGER #
class MonitorContext(AbstractContextManager):
    """
    A context manager for monitoring a process.

    This class provides a context manager that can be used to monitor a process
    for a specific interval and log metrics such as frequency, memory, and temperature
    to a specified log file.

    Attributes:
        monitor (BaseMonitor): An instance of BaseMonitor responsible for handling
            the monitoring process.
    """

    def __init__(self, context="", interval=1.0, log_file_path="logs/context_monitor.ndjson",
                 frequency_unit="MHz", memory_unit="MB", temperature_unit="ºC"):
        """
        Initializes the MonitorContext with the specified parameters.

        Args:
            context (str): A string representing the context of the monitoring.
            interval (float): The monitoring interval in seconds.
            log_file_path (str): The path to the log file for storing monitoring data.
            frequency_unit (str): The unit for frequency measurement.
            memory_unit (str): The unit for memory measurement.
            temperature_unit (str): The unit for temperature measurement.
        """
        try:
            self.monitor = BaseMonitor(context, interval, log_file_path,
                                       frequency_unit, memory_unit, temperature_unit)
        except ValueError as e:
             # Safe traceback extraction #
            tb = traceback.extract_stack()
            last_user_frame = next(
                (frame for frame in reversed(tb) if "inline_monitor.py" not in frame.filename),
                tb[-1] # Fallback #
            )
            print(f"\n[Error] [MonitorContext] {e}")
            print(f"File: {last_user_frame.filename}, Line: {last_user_frame.lineno}\n")
            sys.exit(1)


    def __enter__(self):
        """
        Enter the runtime context related to this object.

        This method starts the monitoring process and returns the context manager instance,
        allowing it to be used with a 'with' statement.

        Returns:
            MonitorContext: The context manager instance.
        """
        self.monitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context related to this object.

        This method stops the monitoring process. It is called when the
        'with' statement is exited, regardless of whether an exception
        occurred or not.

        Args:
            exc_type (type): The exception type, if an exception was raised.
            exc_val (Exception): The exception instance, if an exception was raised.
            exc_tb (traceback): The traceback object, if an exception was raised.
        """
        self.monitor.stop()