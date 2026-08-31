# moniaenergy/monitor/process_monitor.py #
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
import pynvml
import json

from datetime import datetime
from PyQt5.QtCore import QCoreApplication, QTimer, QProcess

from ..config.metrics_units import MetricsUnits
from ..config.metrics_list import MetricsList

class ProcessMonitor():
    def __init__(self, interval: int, json_file_path: str, program_path: str, metrics_units: MetricsUnits, metrics_list: MetricsList):
        """
        Initialize a ProcessMonitor instance.

        Args:
            interval (int): Monitoring interval in seconds.
            json_file_path (str): Path for writing the JSON log file.
            program_path (str): Path to the executable to monitor.
            metrics_units (MetricsUnits): Configuration for metrics units.
            metrics_list (MetricsList): Factory for creating metric instances.

        Attributes:
            _interval_ms (int): Monitoring interval in milliseconds.
            _json_file_path (str): JSON file path.
            _program_path (str): Executable path.
            _metrics_units (MetricsUnits): Metrics units configuration.
            _metrics_list_factory (MetricsList): Factory for creating metric instances.
            _process (QProcess): QProcess instance for monitoring.
            _process_pid (int): Process ID of the monitored process.
            _json_file (file): File object for JSON logging.
            _psutil_process (psutil.Process): psutil process for monitoring.
            _qttimer (QTimer): QTimer for scheduling monitoring tasks.
        """
        self._interval_ms = int(interval * 1000)
        self._json_file_path = json_file_path
        self._program_path = program_path
        self._metrics_units = metrics_units
        self._metrics_list_factory = metrics_list
        self._process = None
        self._process_pid = None
        self._json_file = None
        self._psutil_process = None
        self._qttimer = None

    def run(self):
        """
        Start the process and configure the QTimer for monitoring.

        Steps:
            1. Instantiate QProcess.
            2. Connect stdout and stderr signals.
            3. Start the process.
            4. Create psutil process.
            5. Open JSON log file.
            6. Create QTimer.
            7. Configure QTimer.
            8. Instantiate metrics.
            9. Start QTimer.
        """
        self._process = QProcess()

        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_read_stdout)

        self._process.start("python", ["-u", self._program_path])
        self._process_pid = self._process.processId()
        self._psutil_process = psutil.Process(self._process_pid)
        self._psutil_process.cpu_percent(interval=None) # Dummy call

        self._json_file = open(self._json_file_path, "w", encoding="utf-8")
        self._json_file.write(json.dumps({"frequency": self._metrics_units.frequency[0], 
                                          "memory": self._metrics_units.memory[0], 
                                          "temperature": self._metrics_units.temperature[0]},
                                          ensure_ascii=False) + "\n")
        
        self._json_file.write(json.dumps({"Process started at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) + "\n")
        self._json_file.flush()

        self._qttimer = QTimer()
        self._qttimer.setInterval(self._interval_ms)
        self._qttimer.timeout.connect(self.monitor_step)

        self._metrics_list = self._metrics_list_factory.get_metrics(self._metrics_units, self._psutil_process, self._interval_ms)
        self._qttimer.start()

    def monitor_step(self):
        """
        Periodically monitor the process.

        Checks if the process is running. If not, stops monitoring and finalizes the log.
        Otherwise, logs the current timestamp and metrics values in the JSON file.
        """
        if not psutil.pid_exists(self._process_pid):
            print("Process stopped. Report generated.")
            self._json_file.write(json.dumps({"Process finished at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) + "\n")
            self._json_file.close()
            QCoreApplication.quit()
            return
        
        metrics_step_dict = {}
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics_step_dict["Timestamp"] = current_timestamp

        for metric in self._metrics_list:
            try:
                metrics_step_dict[metric.get_header()] = metric.get_value()
            except psutil.NoSuchProcess:
                metrics_step_dict[metric.get_header()] = 0

        self._json_file.write(json.dumps(metrics_step_dict, ensure_ascii=False) + "\n")
        self._json_file.flush()

    def stop_monitoring(self):
        """
        Stop monitoring the process.

        Stops the QTimer, terminates the process, and closes the JSON log file.
        """
        self._stop_timer()
        self._terminate_process()
        self._terminate_pynvml()
        self._close_json_file()

    def _stop_timer(self):
        """Stop the QTimer."""
        if self._qttimer is not None:
            self._qttimer.stop()

    def _terminate_process(self):
        """Terminate the running process and wait for it to finish."""
        if self._process is not None:
            self._process.readyReadStandardOutput.disconnect(self._on_read_stdout)
            self._process.terminate()
            self._process.waitForFinished(self._interval_ms * 5)

    def _terminate_pynvml(self):
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVML_ERROR_UNINITIALIZED:
            pass

    def _close_json_file(self):
        """Close the JSON log file."""
        if self._json_file is not None:
            self._json_file.close()

    def _on_read_stdout(self):
        """
        Read and process the standard output from the running process.

        Connected to the QProcess's readyReadStandardOutput signal and reads
        available data, decoding it to a UTF-8 string and printing it to the console.
        """
        try:
            data = self._process.readAllStandardOutput()
            text = data.data().decode("utf-8", errors="replace")
            print(text, end="")
        except KeyboardInterrupt:
            pass

