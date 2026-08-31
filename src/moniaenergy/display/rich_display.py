# moniaenergy/display/rich_display.py #
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
rich_display.py  —  Live terminal dashboard for moniaenergy training runs.

Provides TrainingDisplay: a context manager that renders a real-time panel
showing GPU/CPU metrics, batch progress, training statistics, and energy data.
This module does NOT modify any existing moniaenergy functionality; it only
reads hardware state independently via pynvml and RAPL sysfs.
"""

import os
import time
from typing import Optional

import pynvml
from rich import box
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text


class TrainingDisplay:
    """
    Real-time Rich terminal dashboard for moniaenergy training sessions.

    Usage:
        with TrainingDisplay(model_name, device, fp16, total_epochs) as disp:
            for epoch in range(total_epochs):
                for batch_idx, (x, y) in enumerate(loader):
                    ...
                    disp.update_batch("TRAIN", batch_idx, len(loader), loss, acc)
                disp.update_epoch(epoch, train_loss, train_acc, test_acc, ...)
    """

    _RAPL_ENERGY = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
    _RAPL_MAX    = "/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj"

    def __init__(
        self,
        model_name: str,
        device: str,
        fp16: bool,
        total_epochs: int,
        gpu_index: int = 0,
    ):
        self._model_name   = model_name
        self._device       = device
        self._fp16         = fp16
        self._total_epochs = total_epochs

        # Training state
        self._epoch         = 0
        self._phase         = "INIT"
        self._batch         = 0
        self._total_batches = 1
        self._loss          = 0.0
        self._acc           = 0.0
        self._train_loss: Optional[float] = None
        self._train_acc:  Optional[float] = None
        self._test_acc:   Optional[float] = None

        # Energy state (populated after epoch 1)
        self._energy_j:   Optional[float] = None
        self._eps_j:      Optional[float] = None
        self._edp:        Optional[float] = None
        self._co2_eu:     Optional[float] = None
        self._ees_status: Optional[str]   = None
        self._energy_grade: Optional[str] = None
        self._power_cap_w:  Optional[int]  = None   # active GPU power cap from optimizer

        # CPU RAPL state for live power estimate
        self._rapl_last_uj   = self._read_rapl_uj()
        self._rapl_last_time = time.time()
        self._cpu_power_w    = 0.0

        # PID of the training process — used to filter per-process GPU metrics
        self._pid = os.getpid()

        # NVML — direct lightweight queries, independent of NVIDIAGPUMetrics
        self._nvml_ok    = False
        self._gpu_handle = None
        try:
            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self._nvml_ok    = True
        except Exception:
            pass

        # P4: probe nvmlDeviceGetProcessUtilization support (NVML 7.0+ / driver 346+)
        self._proc_util_ok       = False
        self._proc_util_last_ts  = 0   # last-seen timestamp in µs; 0 = first call
        if self._nvml_ok:
            try:
                pynvml.nvmlDeviceGetProcessUtilization(self._gpu_handle, 0)
                self._proc_util_ok = True
            except Exception:
                self._proc_util_ok = False

        # Batch progress bar (embedded inside the Training panel)
        self._progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("[dim]{task.completed}/{task.total}"),
            expand=True,
        )
        self._ptask = self._progress.add_task("INIT", total=1)

        self._live = Live(
            self._render(),
            refresh_per_second=4,
            screen=False,
            transient=False,
        )

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self):
        self._live.start()
        return self

    def __exit__(self, *_):
        self._live.stop()

    # ── Public API ─────────────────────────────────────────────────────────────

    def update_batch(
        self,
        phase: str,
        batch: int,
        total: int,
        loss: float,
        acc: float,
    ) -> None:
        """Call inside the training/test loop after each batch."""
        self._phase         = phase
        self._batch         = batch
        self._total_batches = total
        self._loss          = loss
        self._acc           = acc

        color = "cyan" if phase == "TRAIN" else "magenta"
        self._progress.update(
            self._ptask,
            completed=batch + 1,
            total=total,
            description=f"[{color}]{phase}[/{color}]",
        )
        self._cpu_power_w = self._poll_cpu_power()
        self._live.update(self._render())

    def update_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        test_acc: float,
        energy_j:        Optional[float] = None,
        eps_j:           Optional[float] = None,
        edp:             Optional[float] = None,
        co2_eu:          Optional[float] = None,
        ees_status:      Optional[str]   = None,
        energy_grade:    Optional[str]   = None,
        active_power_cap_w: Optional[int] = None,
    ) -> None:
        """Call at the end of each epoch with aggregated metrics."""
        self._epoch      = epoch
        self._phase      = "DONE"
        self._train_loss = train_loss
        self._train_acc  = train_acc
        self._test_acc   = test_acc
        self._energy_j   = energy_j
        self._eps_j      = eps_j
        self._edp        = edp
        self._co2_eu     = co2_eu
        self._ees_status = ees_status
        self._energy_grade = energy_grade
        if active_power_cap_w is not None:
            self._power_cap_w = active_power_cap_w
        self._live.update(self._render())

    # ── Rendering ──────────────────────────────────────────────────────────────

    def _render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body",   size=10),
            Layout(name="footer", size=3),
        )
        layout["header"].update(self._header())
        layout["body"].split_row(
            Layout(self._gpu_panel(),    name="gpu",    ratio=1),
            Layout(self._train_panel(),  name="train",  ratio=2),
            Layout(self._energy_panel(), name="energy", ratio=1),
        )
        layout["footer"].update(self._footer())
        return layout

    def _header(self) -> Panel:
        prec = "FP16" if self._fp16 else "FP32"
        markup = (
            f"[bold cyan]monIAenergy[/bold cyan]  "
            f"[bold white]{self._model_name}[/bold white]"
            f"  [dim]│[/dim]  "
            f"Epoch [yellow]{self._epoch}[/yellow] / [yellow]{self._total_epochs}[/yellow]"
            f"  [dim]│[/dim]  {prec}"
            f"  [dim]│[/dim]  [dim]{self._device}[/dim]"
        )
        return Panel(Text.from_markup(markup), box=box.ROUNDED, border_style="dim")

    def _gpu_panel(self) -> Panel:
        """
        Shows GPU metrics in two sections:
          · Upper rows — total GPU (all processes combined)
          · Lower rows — this process only (P3)

        When nvmlDeviceGetProcessUtilization is available (P4-primary) the
        process power is labelled "proc (SM)"; when the VRAM-ratio fallback
        is used it is labelled "proc (≈V)" so the user knows the accuracy tier.
        """
        info = self._query_gpu()
        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        table.add_column(style="dim", no_wrap=True)
        table.add_column(justify="right")
        table.add_column(style="dim", justify="left", no_wrap=True)

        if not info:
            table.add_row("GPU", "[dim]N/A[/dim]", "")
            return Panel(table, title="[bold]GPU[/bold]", border_style="blue")

        # ── Total GPU (all processes) ─────────────────────────────────────────
        t  = info["temp_c"]
        tc = "red" if t > 85 else "yellow" if t > 75 else "green"
        table.add_row("Temp", f"[{tc}]{t} °C[/{tc}]", "")

        total_pw = info["power_w"]
        tpc = "red" if total_pw > 350 else "yellow" if total_pw > 250 else "green"
        table.add_row("GPU pwr", f"[{tpc}]{total_pw:.1f} W[/{tpc}]", "[dim]total[/dim]")

        total_vram = info["vram_total_gb"]
        all_used   = info["vram_used_gb"]
        all_pct    = all_used / total_vram * 100 if total_vram > 0 else 0
        avc = "red" if all_pct > 90 else "yellow" if all_pct > 75 else "green"
        table.add_row("GPU VRAM", f"[{avc}]{all_used:.1f}/{total_vram:.0f} GB[/{avc}]", "[dim]total[/dim]")

        # ── Process-specific (P2 / P3) ────────────────────────────────────────
        if "proc_power_w" in info:
            pw  = info["proc_power_w"]
            ppc = "red" if pw > 350 else "yellow" if pw > 250 else "green"
            method_tag = "(SM)" if info.get("proc_ratio_method") == "sm" else "(\u2248V)"
            table.add_row("Proc pwr", f"[{ppc}]{pw:.1f} W[/{ppc}]", f"[dim]{method_tag}[/dim]")

        if "proc_vram_gb" in info:
            pv  = info["proc_vram_gb"]
            pvc = "red" if (pv / total_vram * 100 if total_vram > 0 else 0) > 90 \
                  else "yellow" if (pv / total_vram * 100 if total_vram > 0 else 0) > 60 \
                  else "green"
            table.add_row("Proc VRAM", f"[{pvc}]{pv:.1f}/{total_vram:.0f} GB[/{pvc}]", "[dim]proc[/dim]")

        if "proc_sm_util" in info:
            sm  = info["proc_sm_util"]
            smc = "green" if sm > 80 else "yellow" if sm > 40 else "dim"
            table.add_row("Proc SM", f"[{smc}]{sm}%[/{smc}]", "")

        return Panel(table, title="[bold]GPU[/bold]", border_style="blue")

    def _train_panel(self) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        table.add_column(style="dim", no_wrap=True)
        table.add_column(justify="right")

        pc = {"TRAIN": "cyan", "TEST": "magenta", "DONE": "green", "INIT": "dim"}.get(
            self._phase, "white"
        )
        table.add_row("Phase", f"[{pc}]{self._phase}[/{pc}]")
        table.add_row("Loss",  f"{self._loss:.4f}")
        table.add_row("Acc",   f"[bold]{self._acc * 100:.2f}%[/bold]")

        if self._train_acc is not None:
            table.add_row("Train Acc", f"{self._train_acc * 100:.2f}%")
        if self._test_acc is not None:
            table.add_row("Test Acc", f"[bold green]{self._test_acc * 100:.2f}%[/bold green]")

        return Panel(
            Group(table, self._progress),
            title="[bold]Training[/bold]",
            border_style="cyan",
        )

    def _energy_panel(self) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        table.add_column(style="dim", no_wrap=True)
        table.add_column(justify="right")

        # --- Green Grade (most prominent) ------------------------------------
        _grade_colour = {
            "A++": "bold bright_green",
            "A+":  "bright_green",
            "A":   "green",
            "B":   "cyan",
            "C":   "yellow",
            "D":   "yellow",
            "E":   "red",
            "F":   "bold red",
        }
        if self._energy_grade is not None:
            colour = _grade_colour.get(self._energy_grade, "white")
            table.add_row("Grade", f"[{colour}]{self._energy_grade}[/{colour}]")

        if self._energy_j is not None:
            kj = self._energy_j / 1000
            kc = "red" if kj > 500 else "yellow" if kj > 200 else "green"
            table.add_row("Energy", f"[{kc}]{kj:.1f} kJ[/{kc}]")
        if self._eps_j is not None:
            table.add_row("J/sample", f"{self._eps_j:.3f}")
        if self._edp is not None:
            table.add_row("EDP", f"{self._edp:.2e}")
        if self._co2_eu is not None:
            table.add_row("EU CO\u2082", f"{self._co2_eu:.1f} g")
        if self._energy_j is None:
            table.add_row("", "[dim]after epoch 1\u2026[/dim]")

        # --- Active GPU Power Cap (from GpuPowerOptimizer) --------------------
        if self._power_cap_w is not None:
            try:
                import pynvml as _pnvml
                _h = _pnvml.nvmlDeviceGetHandleByIndex(0)
                _max_mw = _pnvml.nvmlDeviceGetPowerManagementLimitConstraints(_h)[1]
                _max_w = int(_max_mw / 1000)
                _cap_pct = self._power_cap_w / _max_w if _max_w > 0 else 1.0
                _cap_col = "green" if _cap_pct < 0.85 else "yellow" if _cap_pct < 1.0 else "dim"
            except Exception:
                _cap_col = "cyan"
                _max_w = None
            _suffix = f" / {_max_w} W" if _max_w else ""
            table.add_row("PWR Cap", f"[{_cap_col}]{self._power_cap_w} W{_suffix}[/{_cap_col}]")

        return Panel(table, title="[bold]Energy[/bold]", border_style="green")

    def _footer(self) -> Panel:
        cpu_color = "yellow" if self._cpu_power_w > 10 else "dim"
        markup = f"CPU RAPL  [{cpu_color}]{self._cpu_power_w:.1f} W[/{cpu_color}]"

        if self._ees_status:
            markup += f"  [dim]│[/dim]  EES: {self._ees_status}"
        else:
            markup += "  [dim]│[/dim]  EES: [dim]disabled[/dim]"

        return Panel(Text.from_markup(markup), box=box.SIMPLE, border_style="dim")

    # ── Hardware queries ───────────────────────────────────────────────────────

    def _query_gpu(self) -> Optional[dict]:
        """
        Returns a dict with total GPU metrics plus process-specific fields when
        available.  Process fields (prefixed 'proc_') are derived from:
          1. nvmlDeviceGetProcessUtilization  → SM ratio  (most accurate)
          2. VRAM fraction heuristic           → fallback (P4)

        Keys always present (if NVML is available):
          power_w, temp_c, vram_used_gb, vram_total_gb, util_pct

        Keys present when the monitored process is found on this GPU (P2/P3):
          proc_power_w        – estimated process power (W)
          proc_vram_gb        – VRAM used by this process only (GB)
          proc_sm_util        – process SM utilisation (int %) or None if fallback
          proc_ratio_method   – "sm" or "vram" — indicates which method was used
        """
        if not self._nvml_ok:
            return None
        try:
            h             = self._gpu_handle
            total_power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            temp_c        = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            mem           = pynvml.nvmlDeviceGetMemoryInfo(h)
            util_pct      = pynvml.nvmlDeviceGetUtilizationRates(h).gpu

            result = {
                "power_w":       total_power_w,
                "temp_c":        temp_c,
                "vram_used_gb":  mem.used  / 1024 ** 3,
                "vram_total_gb": mem.total / 1024 ** 3,
                "util_pct":      util_pct,
            }

            # ── Process-specific metrics (P2 + P3 + P4) ──────────────────────
            try:
                compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            except Exception:
                return result

            my_proc = next(
                (p for p in compute_procs if p.pid == self._pid), None
            )
            if my_proc is None:
                return result   # process not using this GPU

            # P2: process VRAM
            proc_vram_bytes = my_proc.usedGpuMemory or 0
            result["proc_vram_gb"] = proc_vram_bytes / 1024 ** 3

            # P3/P4: attribution ratio — SM preferred, VRAM as fallback
            ratio: Optional[float] = None
            proc_sm_util: Optional[int] = None

            if self._proc_util_ok:
                try:
                    samples = pynvml.nvmlDeviceGetProcessUtilization(
                        h, self._proc_util_last_ts
                    )
                    if samples:
                        self._proc_util_last_ts = max(s.timeStamp for s in samples)
                        total_sm = sum(s.smUtil for s in samples)
                        my_sm    = next(
                            (s.smUtil for s in samples if s.pid == self._pid), None
                        )
                        if my_sm is not None:
                            proc_sm_util = int(my_sm)
                            ratio = (my_sm / total_sm) if total_sm > 0 else 0.0
                            result["proc_sm_util"]      = proc_sm_util
                            result["proc_ratio_method"] = "sm"
                except Exception:
                    self._proc_util_ok = False

            # P4: VRAM fallback
            if ratio is None:
                total_vram = sum((p.usedGpuMemory or 0) for p in compute_procs)
                ratio = (proc_vram_bytes / total_vram) if total_vram > 0 else 1.0
                result["proc_ratio_method"] = "vram"

            result["proc_power_w"] = round(total_power_w * ratio, 1)
            return result

        except Exception:
            return None

    def _read_rapl_uj(self) -> Optional[int]:
        try:
            with open(self._RAPL_ENERGY) as f:
                return int(f.read().strip())
        except Exception:
            return None

    def _poll_cpu_power(self) -> float:
        now_uj = self._read_rapl_uj()
        if now_uj is None or self._rapl_last_uj is None:
            return self._cpu_power_w
        now_t    = time.time()
        delta_uj = now_uj - self._rapl_last_uj
        if delta_uj < 0:
            try:
                with open(self._RAPL_MAX) as f:
                    max_uj = int(f.read().strip())
                delta_uj = (max_uj - self._rapl_last_uj) + now_uj
            except Exception:
                delta_uj = 0
        delta_t = now_t - self._rapl_last_time
        self._rapl_last_uj   = now_uj
        self._rapl_last_time = now_t
        if delta_t <= 0:
            return self._cpu_power_w
        self._cpu_power_w = round(delta_uj * 1e-6 / delta_t, 2)
        return self._cpu_power_w
