# API Reference — Complete Class Documentation

> **Auto-generated reference for monitor_app v0.5.1**  
> Last updated: 2026-04-16

---

## Table of Contents

1. [High-Level API: `monitor_train()`](#monitor_train)
2. [Monitoring Core](#monitoring-core)
3. [Energy Optimization](#energy-optimization)
4. [Metrics & Grading](#metrics--grading)
5. [Data Export](#data-export)

---

## High-Level API: `monitor_train()`

### `monitor_train()` — Context Manager

**Module:** `monitor_app.facade`

**Purpose:** Single-entry point for all monitoring, optimization, and grading features.

```python
from monitor_app import monitor_train

with monitor_train(
    model=my_model,
    experiment_name="resnet50_cifar10",
    country="Spain",
    batch_size=128,
    fp16=False,
    early_stopping=True,
    patience=3,
    power_optimize=True,
    gpu_index=0,
    time_budget_pct=0.10,
) as mon:
    for epoch in range(num_epochs):
        mon.epoch_start(epoch)
        loss, acc = train_one_epoch(...)
        if mon.epoch_end(epoch, samples=N, loss=loss, accuracy=acc):
            break  # Early stopping triggered
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | `torch.nn.Module` | **required** | Neural network to monitor |
| `experiment_name` | `str` | **required** | Unique run identifier |
| `country` | `str` | `"Spain"` | Country for CO₂ factor (ISO code: `ES`, `DE`, etc.) |
| `batch_size` | `int` | auto-detect | Batch size (used by optimizer advisor) |
| `fp16` | `bool` | `False` | Is training using mixed precision? |
| `early_stopping` | `bool` | `False` | Enable **Energy Early Stopping** |
| `patience` | `int` | `3` | EES patience (epochs without improvement) |
| `power_optimize` | `bool` | `False` | Enable **GPU Power Optimizer** |
| `gpu_index` | `int` | `0` | NVIDIA GPU device index |
| `time_budget_pct` | `float` | `0.10` | Tolerate up to N% longer epochs for energy savings |
| `verbose` | `bool` | `True` | Print live progress to terminal |

**Returns:** `MonitorSession` — Context object with epoch lifecycle methods.

**Output Files:**

```
logs/{experiment_name}/
├── run_YYYYMMDD_HHMMSS.ndjson              # Raw per-epoch metrics
├── run_YYYYMMDD_HHMMSS.events.ndjson       # Epoch boundary events
├── run_YYYYMMDD_HHMMSS_energy_metrics.csv  # Per-epoch energy summary
└── run_YYYYMMDD_HHMMSS_layer_profile.csv   # Per-layer energy breakdown
```

**Raises:** `RuntimeError` if model parameter count cannot be determined.

---

### `MonitorSession.epoch_start(epoch: int)` → `None`

Mark the start of a training epoch. Emits `epoch_start` event to the log.

```python
with monitor_train(...) as mon:
    for epoch in range(num_epochs):
        mon.epoch_start(epoch)  # Start timing
        ...
```

### `MonitorSession.epoch_end(epoch, samples, loss, accuracy, **kwargs)` → `bool`

Report epoch completion with metrics. Emits `epoch_end` event.

**Parameters:**

- `epoch` (`int`): Zero-based epoch index
- `samples` (`int`): Number of samples processed this epoch
- `loss` (`float`): Training loss
- `accuracy` (`float`): Accuracy (0–1 or 0–100, user convention)
- `val_loss` (optional, `float`): Validation loss
- `val_accuracy` (optional, `float`): Validation accuracy

**Returns:** `True` if **Energy Early Stopping** requests training termination; `False` otherwise.

```python
for epoch in range(num_epochs):
    mon.epoch_start(epoch)
    loss, acc = train_one_epoch(loader, model, optimizer)
    
    should_stop = mon.epoch_end(
        epoch=epoch,
        samples=len(dataset),
        loss=loss,
        accuracy=acc,
    )
    
    if should_stop:
        print("EES triggered — stopping training")
        break
```

---

## Monitoring Core

### `MonitorContext` — Low-Level Hardware Monitor

**Module:** `monitor_app.monitor.inline_monitor`

A context manager for raw system monitoring (CPU, GPU, memory, temperature).

```python
from monitor_app import MonitorContext

with MonitorContext(
    context="training phase",
    interval=1.0,
    log_file_path="logs/raw_monitor.ndjson",
) as mon_ctx:
    # Training loop — hardware metrics logged automatically
    for epoch in range(num_epochs):
        loss = train_one_epoch(...)
```

**Attributes:**

- `monitor` — Internal `BaseMonitor` instance handling hardware sampling

**Methods:**

- `__enter__()` → Starts hardware monitoring thread
- `__exit__()` → Stops monitoring and closes log file

---

### `EpochTracker` — Epoch Event Logging

**Module:** `monitor_app.monitor.pytorch_hooks`

Records epoch boundaries (start/end) to a sidecar NDJSON file for energy attribution.

```python
from monitor_app import EpochTracker

tracker = EpochTracker(log_file_path="logs/training.ndjson")

for epoch in range(num_epochs):
    tracker.on_epoch_start(epoch)
    loss, acc = train_one_epoch(...)
    tracker.on_epoch_end(epoch, samples=50000, loss=loss, accuracy=acc)
```

**Methods:**

#### `__init__(log_file_path: str)`

Initialize the tracker. Epoch events are written to `{stem}.events.ndjson`.

#### `on_epoch_start(epoch: int)`

Record the start of epoch N. Writes JSON line:

```json
{"event": "epoch_start", "epoch": 0, "Timestamp": "2026-04-16 10:20:30"}
```

#### `on_epoch_end(epoch: int, samples: int = 0, loss: float = None, accuracy: float = None)`

Record the end of epoch N with optional metrics. Writes JSON line:

```json
{
  "event": "epoch_end",
  "epoch": 0,
  "samples": 50000,
  "loss": 0.123,
  "accuracy": 0.95,
  "Timestamp": "2026-04-16 10:21:45"
}
```

---

### `LayerEnergyProfiler` — Per-Layer Timing

**Module:** `monitor_app.monitor.pytorch_hooks`

Measures forward-pass compute time per layer using PyTorch hooks.

```python
from monitor_app import LayerEnergyProfiler

profiler = LayerEnergyProfiler(model, device="cuda")

for epoch in range(num_epochs):
    for batch in train_loader:
        model(batch)  # Profiler records timing automatically
    
    # Retrieve per-layer summary
    df = profiler.get_summary()
    print(df[["module", "avg_time_ms", "std_time_ms"]])

profiler.remove()  # Unregister hooks
```

**Methods:**

#### `__init__(model: torch.nn.Module, device: str = "cuda")`

Register pre/post-hooks on all leaf modules.

- **Parameters:**
  - `model`: PyTorch neural network
  - `device`: `"cuda"` or `"cpu"` for timing method selection

#### `get_summary() → pd.DataFrame`

Return accumulated timings as a DataFrame.

| Column | Type | Description |
|--------|------|-------------|
| `module` | `str` | Layer name (e.g., `features.0`) |
| `calls` | `int` | Number of forward passes |
| `avg_time_ms` | `float` | Mean forward time (milliseconds) |
| `std_time_ms` | `float` | Std deviation |
| `total_time_ms` | `float` | Sum of all forward passes |

#### `remove()`

Unregister all hooks. Call at the end of profiling.

---

## Energy Optimization

### `EnergyEarlyStopping` — Automatic Training Termination

**Module:** `monitor_app.monitor.pytorch_hooks`

Stops training when per-sample energy efficiency falls below a threshold.

```python
from monitor_app import EnergyEarlyStopping

ees = EnergyEarlyStopping(
    log_file_path="logs/resnet50/run_XXX.ndjson",
    min_efficiency_ratio=0.05,  # Auto-calibrate from first epoch
    patience=3,                  # Stop after 3 bad epochs
)

for epoch in range(num_epochs):
    tracker.on_epoch_start(epoch)
    loss, acc = train_one_epoch(...)
    tracker.on_epoch_end(epoch, samples=N, loss=loss, accuracy=acc)
    
    if ees.step(epoch, accuracy=acc):
        print("EES triggered — insufficient progress per joule")
        break
```

**Attributes:**

- `_ratio` — Calibration ratio (default: 0.05)
- `_min_efficiency` — Absolute threshold ΔAccuracy/J (if set, disables auto-calibration)
- `_patience` — Consecutive bad epochs before stopping
- `_bad_epochs` — Counter of recent inefficient epochs
- `_prev_accuracy` — Previous epoch accuracy (for delta)

**Methods:**

#### `__init__(log_file_path, min_efficiency_ratio=0.05, min_efficiency=None, patience=3)`

Initialize EES controller.

- **Parameters:**
  - `log_file_path` (`str`): Path to NDJSON monitoring log
  - `min_efficiency_ratio` (`float`): Calibration ratio. If `min_efficiency` is None, threshold = ratio × first_epoch_efficiency
  - `min_efficiency` (`float`, optional): Absolute threshold in ΔAccuracy/J. When set, disables auto-calibration.
  - `patience` (`int`): Max consecutive inefficient epochs before stopping

#### `step(epoch: int, accuracy: float) → bool`

Evaluate EES criterion. Call after `EpochTracker.on_epoch_end()`.

- **Returns:** `True` if training should stop; `False` otherwise.
- **Prints:** Status line with efficiency vs. threshold comparison.

---

### `GpuPowerOptimizer` — Pareto-Optimal Power Limiting

**Module:** `monitor_app.monitor.gpu_power_optimizer`

Automatically adjusts GPU power cap (W) to explore the Pareto frontier of energy vs. accuracy.

**Algorithm:** Two-phase Explore → Exploit using NVIDIA RAPL.

```python
from monitor_app import GpuPowerOptimizer

optimizer = GpuPowerOptimizer(
    gpu_index=0,
    time_budget_pct=0.10,  # Allow 10% slowdown for energy gains
    min_frac=0.60,         # Min power = 60% of max
    n_steps=5,             # Probe 5 power levels
)

for epoch in range(30):
    # ... training ...
    
    optimizer.step(
        epoch=epoch,
        epoch_time_s=45.2,
        energy_j=500.0,
        samples=50000,
    )
```

**Attributes:**

- `mode` — `"active"` (can set power caps), `"advisor-only"` (no root), or `"unavailable"` (pynvml missing)
- `power_caps_w` — List of candidate power levels (W)
- `min_cap_w`, `max_cap_w` — Constraint bounds

**Methods:**

#### `__init__(gpu_index=0, time_budget_pct=0.10, min_frac=0.60, n_steps=5, verbose=True)`

Initialize the optimizer.

- **Parameters:**
  - `gpu_index` (`int`): NVIDIA GPU device index
  - `time_budget_pct` (`float`): Max tolerated slowdown (e.g., 0.10 = 10%)
  - `min_frac` (`float`): Minimum power as fraction of max (e.g., 0.60 = 60%)
  - `n_steps` (`int`): Number of power levels to probe
  - `verbose` (`bool`): Print decision table

#### `step(epoch: int, epoch_time_s: float, energy_j: float, samples: int) → None`

Process one epoch and adjust power cap if Exploration complete.

- **Parameters:**
  - `epoch` (`int`): Epoch number
  - `epoch_time_s` (`float`): Epoch duration (seconds)
  - `energy_j` (`float`): Energy consumed (Joules)
  - `samples` (`int`): Samples processed this epoch

#### `restore() → None`

Restore original GPU power limit. Called automatically on cleanup.

**Output Example:**

```
GPU Power Optimization (Epoch 3):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [Exploration Complete] Switching to Exploit phase.
  
  Available Pareto-Optimal Caps:
  ┌─────────┬──────────────┬─────────────┬───────┐
  │ Cap (W) │ Avg Time (s) │ Energy (J)  │ Grade │
  ├─────────┼──────────────┼─────────────┼───────┤
  │   150   │    46.5      │    480      │   A++ │ ← Selected
  │   180   │    45.2      │    515      │   A+  │
  │   240   │    43.8      │    580      │   A   │
  │   270   │    42.5      │    650      │   B   │
  └─────────┴──────────────┴─────────────┴───────┘
  
  Decision: Cap 150W selected.
  Justification: Lowest energy consumption (480J) with acceptable
                 slowdown (+3.0%, within budget of +10%).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### `OptimizerAdvisor` — Real-Time Suggestions

**Module:** `monitor_app.monitor.optimizer_advisor`

Emits per-epoch optimization suggestions based on energy trends.

```python
from monitor_app import OptimizerAdvisor
from monitor_app import compute_energy_metrics

advisor = OptimizerAdvisor(
    batch_size=128,
    fp16=False,
    early_stopping_active=False,
)

for epoch in range(num_epochs):
    tracker.on_epoch_start(epoch)
    loss, acc = train_one_epoch(...)
    tracker.on_epoch_end(epoch, samples=N, loss=loss, accuracy=acc)
    
    # Compute energy metrics and let advisor analyze
    energy_df = compute_energy_metrics(log_path)
    advisor.step(epoch, energy_df)
```

**Methods:**

#### `__init__(batch_size=128, fp16=False, early_stopping_active=False, current_grade=None, min_improvement=0.20)`

Initialize advisor.

#### `step(epoch: int, energy_df: pd.DataFrame) → None`

Analyze epoch and print suggestions if applicable.

**Suggestion Types:**

1. **Batch Size:** If GPU utilization is sub-optimal
2. **Mixed Precision:** Recommend FP16 if not already enabled
3. **Energy Early Stopping:** Flag declining efficiency trend

**Output Example:**

```
Epoch 5 (Grade: A):  💡 SUGERENCIA: FP32 activo — prueba --fp16 para 
reducir consumo ~25%.
```

---

## Metrics & Grading

### `compute_grade()` — Energy Efficiency Letter Grade

**Module:** `monitor_app.metrics.green_grader`

Compute the Universal Energy Grade (A++–F) for a configuration.

```python
from monitor_app import compute_grade

result = compute_grade(
    accuracy=0.95,           # 95% accuracy
    parameters=23500000,     # 23.5M model params
    total_energy_j=10000.0,  # 10 kJ consumed
    reference_score=0.35,    # Reference (B grade) score
)

print(f"Grade: {result.grade}")
print(f"Score: {result.eff_score:.4f}")
print(f"Percentage of Reference: {result.pct_of_reference:.1f}%")
print(f"Label: {result.label}")
```

**Parameters:**

- `accuracy` (`float`): Accuracy [0–1] or [0–100]
- `parameters` (`int`): Model parameter count
- `total_energy_j` (`float`): Total energy (Joules)
- `reference_score` (`float`, optional): Calibration reference (B grade = 100%)

**Returns:** `GradeResult` dataclass with fields:

| Field | Type | Description |
|-------|------|-------------|
| `grade` | `str` | Letter (A++, A+, A, B, C, D, E, F) |
| `eff_score` | `float` | Raw score = (acc × log₁₀(params)) / energy_j |
| `reference_score` | `float` | Reference threshold (B grade) |
| `pct_of_reference` | `float` | Efficiency as % of reference |
| `label` | `str` | Human-readable summary |

**Formula:**

$$\text{Grade Score} = \frac{\text{Accuracy} \times \log_{10}(\text{Parameters})}{\text{Total Energy (J)}}$$

---

### `calibrate_reference()` — Auto-Calibrate Thresholds

**Module:** `monitor_app.metrics.green_grader`

Auto-calibrate reference score (B grade threshold) from a log directory.

```python
from monitor_app import calibrate_reference

reference, percentiles = calibrate_reference(
    log_dir="logs/",
    percentile=50.0,  # Use median
)

print(f"Reference (B grade): {reference:.4f}")
print(f"Percentiles: {percentiles}")
```

**Returns:** Tuple of `(reference_score, percentiles_dict)`

---

### `compute_energy_metrics()` — Per-Epoch Energy Summary

**Module:** `monitor_app.monitor.pytorch_hooks`

Compute per-epoch energy, carbon, and efficiency metrics from monitoring logs.

```python
from monitor_app import compute_energy_metrics

energy_df = compute_energy_metrics(
    log_file_path="logs/resnet50/run_20260416_102030.ndjson",
    country="Spain",
    model_parameters=23500000,
)

print(energy_df[[
    "epoch",
    "energy_j",
    "energy_per_sample_j",
    "carbon_g",
    "accuracy",
    "grade",
]])
```

**Parameters:**

- `log_file_path` (`str`): Path to NDJSON monitoring log
- `country` (`str`, optional): Country for CO₂ factor lookup (default: "Spain")
- `model_parameters` (`int`, optional): Parameter count for grading

**Returns:** `pd.DataFrame` with columns:

| Column | Type | Description |
|--------|------|-------------|
| `epoch` | `int` | Epoch number |
| `energy_j` | `float` | Total energy (Joules) |
| `cpu_energy_j` | `float` | CPU portion |
| `gpu_energy_j` | `float` | GPU portion |
| `energy_per_sample_j` | `float` | Energy / samples (Intensity Factor) |
| `carbon_g` | `float` | CO₂ emissions (grams) |
| `accuracy` | `float` | Training accuracy |
| `grade` | `str` | Energy Grade (A++–F) |
| `gpu_power_avg_w` | `float` | Avg GPU power |
| `gpu_temp_avg_c` | `float` | Avg GPU temperature |
| `timestamp_start` | `str` | Epoch start ISO timestamp |
| `timestamp_end` | `str` | Epoch end ISO timestamp |

---

## Data Export

### NDJSON Log Format

Each line is a JSON object containing instantaneous metrics:

```json
{
  "timestamp": "2026-04-16T10:20:30.123456Z",
  "epoch": 0,
  "gpu_index": 0,
  "gpu_power_w": 250.5,
  "gpu_energy_j": 12615.0,
  "gpu_utilization_pct": 98.5,
  "gpu_vram_used_mb": 8192,
  "gpu_temp_c": 65.2,
  "cpu_power_w": 45.3,
  "cpu_energy_j": 2265.0,
  "cpu_utilization_pct": 75.2,
  "cpu_freq_mhz": 3400,
  "memory_rss_mb": 4096
}
```

### CSV Energy Metrics Format

Per-epoch summary exported as CSV:

```csv
epoch,timestamp_start,timestamp_end,duration_s,samples,energy_j,cpu_energy_j,gpu_energy_j,energy_per_sample_j,carbon_g,accuracy,loss,grade,gpu_power_avg_w,gpu_temp_avg_c
0,2026-04-16T10:20:30Z,2026-04-16T10:21:45Z,75.0,50000,10000.5,2265.0,12615.0,0.02,0.25,0.85,0.123,A++,250.5,65.2
```

---

## Frequently Used Imports

```python
# High-level API
from monitor_app import monitor_train

# Core classes
from monitor_app import (
    MonitorContext,
    EpochTracker,
    LayerEnergyProfiler,
    EnergyEarlyStopping,
    GpuPowerOptimizer,
    OptimizerAdvisor,
)

# Metrics
from monitor_app import (
    compute_grade,
    compute_energy_metrics,
    calibrate_reference,
    GradeResult,
)

# Rich terminal display (optional)
from monitor_app import TrainingDisplay
```

---

**See also:** [Concepts](docs/concepts/energy-grading.md) for theory, [Examples](docs/examples/training.md) for practical recipes.
