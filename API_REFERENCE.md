# API_REFERENCE.md — Technical Documentation v1.0

> **monitor_app v1.0.0 — High-Fidelity Energy Monitoring Framework for AI Training**  
> **Benjamín Sánchez Calza · v1.0.0 · 2026 · GPL-3.0**

---

## Overview

This document provides comprehensive technical reference for the four core classes of `monitor_app`:

1. **`monitor_train()`** — High-level context manager for complete monitoring
2. **`GpuPowerOptimizer`** — Pareto frontier exploration and GPU power limiting (Zeus-style)
3. **`EnergyEarlyStopping`** — Automatic training termination based on Energy Intensity Factor
4. **`LayerEnergyProfiler`** — Per-layer energy attribution and hotspot detection

---

## 1. `monitor_train()` — High-Level Facade

### Synopsis

```python
from monitor_app import monitor_train

with monitor_train(
    model,
    experiment_name,
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
        loss, acc = train_one_epoch(loader, model, optimizer)
        if mon.epoch_end(epoch, samples=len(dataset), loss=loss, accuracy=acc):
            break
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module` | **required** | PyTorch neural network to monitor. Auto-detects parameter count for Green AI Grade computation. |
| `experiment_name` | `str` | **required** | Unique identifier for this run (defines output directory in `logs/`). Use alphanumeric + underscores. |
| `country` | `str` | `"Spain"` | Country code (ISO-3166-1 alpha-2 or alpha-3) for carbon intensity lookup (Ember 2025 dataset). Examples: `"ES"`, `"DE"`, `"US"`. |
| `batch_size` | `int` | auto | Batch size used in training. Passed to `OptimizerAdvisor` for suggestions (e.g., "increase batch to 256 for better GPU utilization"). |
| `fp16` | `bool` | `False` | Whether mixed-precision (FP16/AMP) is enabled. Passed to advisor for recommendations. |
| `early_stopping` | `bool` | `False` | Enable **Energy Early Stopping (EES)** — automatic termination when Energy Intensity decays. |
| `patience` | `int` | `3` | Consecutive epochs below efficiency threshold before EES triggers. |
| `power_optimize` | `bool` | `False` | Enable **GPU Power Optimizer** — automatic Pareto frontier exploration via NVIDIA RAPL. Requires `nvidia-ml-py`. |
| `gpu_index` | `int` | `0` | Which NVIDIA GPU to monitor/optimize (0-indexed). |
| `time_budget_pct` | `float` | `0.10` | Tolerate up to N% slowdown when reducing power (e.g., 0.10 = allow 10% longer epochs for energy savings). |
| `verbose` | `bool` | `True` | Print real-time dashboard and energy grade updates to terminal. |

### Return Type

**`MonitorSession`** — Context object with two public methods:

#### `mon.epoch_start(epoch: int) → None`

Record the start of a training epoch. Emits `epoch_start` event to sidecar `.events.ndjson` file.

```python
mon.epoch_start(0)
loss, acc = train_one_epoch(...)
```

#### `mon.epoch_end(epoch: int, samples: int, loss: float, accuracy: float, **kwargs) → bool`

Report epoch completion with metrics. Emits `epoch_end` event. 

**Parameters:**
- `epoch` (`int`): Zero-based epoch index
- `samples` (`int`): Number of samples processed
- `loss` (`float`): Training loss value
- `accuracy` (`float`): Accuracy (0–1 or 0–100, user convention)
- `val_loss` (optional, `float`): Validation loss
- `val_accuracy` (optional, `float`): Validation accuracy

**Returns:** `bool` — `True` if **Energy Early Stopping** requests training halt; `False` otherwise.

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
        print(f"EES triggered at epoch {epoch}")
        break
```

### Advanced Usage Example

**Scenario:** Train ResNet50 on CIFAR-10 with full optimization suite (EES + GPU power limiting).

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from monitor_app import monitor_train

# Data loading
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
train_dataset = datasets.CIFAR10('.', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=256, num_workers=4)

# Model and optimizer
device = torch.device('cuda:0')
model = models.resnet50(pretrained=False, num_classes=10).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Enable full optimization
with monitor_train(
    model=model,
    experiment_name="resnet50_cifar10_optimized",
    country="Spain",
    batch_size=256,
    fp16=False,
    early_stopping=True,      # ← Enable EES
    patience=5,
    power_optimize=True,       # ← Enable GPU power optimization (Pareto frontier)
    gpu_index=0,
    time_budget_pct=0.15,      # ← Allow 15% slowdown for energy savings
) as mon:
    for epoch in range(100):
        mon.epoch_start(epoch)
        
        model.train()
        total_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(labels)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += len(labels)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        if mon.epoch_end(epoch, samples=total, loss=avg_loss, accuracy=accuracy):
            break  # EES or power limit reached
        
        scheduler.step()

# Output: Energy grade, J/sample, CO₂ printed automatically
```

**Expected Output:**

```
════════════════════════════════════════════════════════════════════════════
  monitor_train — Final Report: resnet50_cifar10_optimized
════════════════════════════════════════════════════════════════════════════
  Energy Grade:         A   (210% of reference)
  Intensity Factor:     0.0385 J/sample
  Total Energy:         9840 J
  CO₂ Estimated:        0.32 g (Spain, 91 gCO₂/kWh)
  Epochs Completed:     35 (stopped by EES at epoch 35)
  
  Optimizations Active:
    ✓ Energy Early Stopping (EES) — triggered at epoch 35
      Reason: Energy per +1% accuracy dropped below threshold
      Savings: ~42% vs. full training (predicted)
    
    ✓ GPU Power Cap:  180W (applied from epoch 10)
      Impact: +8.5% epoch time, 27.3% energy reduction
      Trade-off: Pareto-optimal (sweet spot)
  
  Recommendations:
    — Accuracy degradation: <1.5% (within acceptable bounds)
    — Consider FP16 training for additional ~20% savings
  
  Log: logs/resnet50_cifar10_optimized/run_20260416_143000.ndjson
════════════════════════════════════════════════════════════════════════════
```

---

## 2. `GpuPowerOptimizer` — Pareto Frontier Exploration

### Synopsis

The **GPU Power Optimizer** automatically seeks the **Pareto Frontier** (optimal trade-off between energy and time) by dynamically adjusting GPU power limits.

```python
from monitor_app import GpuPowerOptimizer

optimizer = GpuPowerOptimizer(
    gpu_index=0,
    time_budget_pct=0.10,
    min_frac=0.60,
    n_steps=5,
)

for epoch in range(num_epochs):
    # ... training ...
    optimizer.step(
        epoch=epoch,
        epoch_time_s=45.2,
        energy_j=500.0,
        samples=50000,
    )

optimizer.restore()  # Restore original power limit
```

### Algorithm

**Phase 1 — Exploration** (first `n_steps` epochs):
- Test power caps: [60%, 72%, 84%, 96%, 100%] of GPU maximum
- Record (cap_w, epoch_time_s, energy_j) for each
- Restore original limit after each test (no persistence)

**Phase 2 — Exploitation** (remaining epochs):
- Build Pareto frontier from exploration data
- Select lowest cap where: `epoch_time ≤ baseline_time × (1 + time_budget_pct)`
- Apply once and hold for rest of training
- Print decision table with human-readable justification

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu_index` | `int` | `0` | NVIDIA device index (0 = first GPU). |
| `time_budget_pct` | `float` | `0.10` | Max tolerated slowdown (0.10 = 10%). Selector chooses lowest energy cap respecting this. |
| `min_frac` | `float` | `0.60` | Minimum power as fraction of GPU max (0.60 = 60%). Prevents over-limiting. |
| `n_steps` | `int` | `5` | Number of power levels to probe during exploration. More steps = finer granularity, longer exploration. |
| `verbose` | `bool` | `True` | Print Pareto analysis table and decision justification. |

### Advanced Usage Example

**Scenario:** Optimize ResNet18 on CIFAR-10, exploring power frontier with detailed reporting.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from monitor_app import GpuPowerOptimizer, EpochTracker, compute_energy_metrics

# Setup
device = torch.device('cuda:0')
model = models.resnet18(pretrained=False, num_classes=10).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Load data (assume CIFAR10 downloaded)
transform = transforms.ToTensor()
train_dataset = datasets.CIFAR10('.', train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4)

# Initialize power optimizer
power_opt = GpuPowerOptimizer(
    gpu_index=0,
    time_budget_pct=0.15,  # Allow 15% slowdown for energy savings
    min_frac=0.50,         # Try down to 50% of max power
    n_steps=7,             # Probe 7 power levels (finer granularity)
    verbose=True,
)

# Initialize epoch tracker for energy metrics
log_path = "logs/resnet18_power_opt/run_20260416.ndjson"
epoch_tracker = EpochTracker(log_path)

import time

for epoch in range(50):
    epoch_tracker.on_epoch_start(epoch)
    epoch_start_time = time.time()
    
    model.train()
    total_loss = 0.0
    correct, total = 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += len(labels)
    
    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / total
    accuracy = correct / total
    
    # Report to epoch tracker
    epoch_tracker.on_epoch_end(
        epoch=epoch,
        samples=total,
        loss=avg_loss,
        accuracy=accuracy,
    )
    
    # Compute energy metrics and feed to optimizer
    energy_df = compute_energy_metrics(log_path, model_parameters=11173962)
    if not energy_df.empty:
        row = energy_df[energy_df['epoch'] == epoch]
        if not row.empty:
            energy_j = row['energy_j'].iloc[0]
            power_opt.step(
                epoch=epoch,
                epoch_time_s=epoch_time,
                energy_j=energy_j,
                samples=total,
            )
    
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={accuracy:.2%}, Time={epoch_time:.1f}s")

# Restore original GPU power limit
power_opt.restore()
```

**Expected Console Output (Exploration Phase):**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[GpuPowerOptimizer] Probing Power Caps — Exploration Phase
────────────────────────────────────────────────────────────────────────────
Epoch 0: Testing cap=270W  (100%) — Time=45.2s, Energy=500.0J  [Baseline]
Epoch 1: Testing cap=226W  (84%)  — Time=47.5s, Energy=415.0J  [ΔT=+5.1%, ΔE=-17.0%]
Epoch 2: Testing cap=183W  (68%)  — Time=50.3s, Energy=335.0J  [ΔT=+11.3%, ΔE=-33.0%]
Epoch 3: Testing cap=145W  (54%)  — Time=54.1s, Energy=280.0J  [ΔT=+19.7%, ΔE=-44.0%]
Epoch 4: Testing cap=108W  (40%)  — Time=62.8s, Energy=235.0J  [ΔT=+38.9%, ΔE=-53.0%] ✗ Over budget
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[GpuPowerOptimizer] Pareto Frontier Analysis — Switching to Exploit Phase
────────────────────────────────────────────────────────────────────────────
  Cap (W)    Time (s)    Energy (J)   ΔT vs. Best    Viable?
  ──────────────────────────────────────────────────────────
  108        62.8        235          +38.9%         ✗ Over budget (+15%)
  145        54.1        280          +19.7%         ✗ Over budget
  183        50.3        335          +11.3%         ✓ Within budget
  226        47.5        415          +5.1%          ✓ (more expensive)
  270        45.2        500          (baseline)     ✓ (no optimization)

  → SELECTION: Cap = 183W (68% of max)
     JUSTIFICATION: Lowest energy (335J) respecting time budget (+15%).
                    Saves 33% energy with 11.3% slowdown.
                    Applied from epoch 5 onwards.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 3. `EnergyEarlyStopping` — Efficiency-Based Termination

### Synopsis

Automatically halts training when per-sample **Energy Intensity Factor** (J/sample) efficiency decays below a threshold.

```python
from monitor_app import EnergyEarlyStopping

ees = EnergyEarlyStopping(
    log_file_path="logs/run.ndjson",
    min_efficiency_ratio=0.05,
    patience=3,
)

for epoch in range(num_epochs):
    tracker.on_epoch_start(epoch)
    loss, acc = train_one_epoch(...)
    tracker.on_epoch_end(epoch, samples=N, loss=loss, accuracy=acc)
    
    if ees.step(epoch, accuracy=acc):
        print("EES triggered — stopping training")
        break
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_file_path` | `str` | required | Path to NDJSON monitoring log (same as `EpochTracker`). EES reads energy metrics from this file. |
| `min_efficiency_ratio` | `float` | `0.05` | Calibration ratio. Threshold = ratio × first_epoch_efficiency. (Auto-calibrates from first epoch.) |
| `min_efficiency` | `float` | `None` | Absolute threshold in ΔAccuracy/J. If set, disables auto-calibration. |
| `patience` | `int` | `3` | Consecutive epochs below threshold before training stops. |

### Theory

**Efficiency metric:** `(ΔAccuracy) / (ΔEnergy_J)`

Threshold auto-calibrates from first productive epoch:
```
threshold = min_efficiency_ratio × first_epoch_efficiency
```

This makes the controller **hardware-agnostic**: faster GPUs have higher baseline efficiency, so threshold scales accordingly.

### Advanced Usage Example

**Scenario:** Train with EES on multiple GPUs, with mixed-precision and checkpointing.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, models, transforms
from monitor_app import (
    EpochTracker,
    EnergyEarlyStopping,
    compute_energy_metrics,
)

def train_with_ees(local_rank, world_size):
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    
    device = torch.device(f'cuda:{local_rank}')
    
    # Model setup with DDP
    model = models.resnet50(pretrained=False, num_classes=10)
    model.to(device)
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Data loading with DistributedSampler
    transform = transforms.ToTensor()
    train_dataset = datasets.CIFAR10('.', train=True, transform=transform)
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        sampler=sampler,
        num_workers=4,
    )
    
    # Energy monitoring
    log_path = f"logs/resnet50_ddp_rank{local_rank}.ndjson"
    tracker = EpochTracker(log_path)
    
    # EES with explicit threshold
    ees = EnergyEarlyStopping(
        log_file_path=log_path,
        min_efficiency=1e-6,  # Explicit: ΔAccuracy per joule
        patience=4,           # Stop after 4 bad epochs
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision
    
    num_epochs = 100
    checkpoint_dir = "checkpoints"
    
    for epoch in range(num_epochs):
        tracker.on_epoch_start(epoch)
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        
        model.train()
        total_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() * len(labels)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += len(labels)
        
        accuracy = correct / total
        avg_loss = total_loss / total
        
        # Report to monitoring
        tracker.on_epoch_end(
            epoch=epoch,
            samples=total,
            loss=avg_loss,
            accuracy=accuracy,
        )
        
        # Check EES criterion
        if ees.step(epoch, accuracy=accuracy):
            # Save checkpoint before stopping
            if local_rank == 0:
                torch.save({
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, f'{checkpoint_dir}/best_ees.pt')
            print(f"Rank {local_rank}: EES triggered at epoch {epoch}")
            break
        
        scheduler.step()
    
    torch.distributed.destroy_process_group()

# Launch: python -m torch.distributed.launch --nproc_per_node=2 script.py
```

---

## 4. `LayerEnergyProfiler` — Per-Layer Energy Attribution

### Synopsis

Measures forward-pass compute time per layer and enables energy attribution by layer.

```python
from monitor_app import LayerEnergyProfiler

profiler = LayerEnergyProfiler(model, device="cuda")

for epoch in range(num_epochs):
    for batch in train_loader:
        model(batch)  # Hooks record timing

df = profiler.get_summary()
print(df[["module", "avg_time_ms", "std_time_ms", "total_time_ms"]])

profiler.remove()
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module` | required | PyTorch model to profile. Registers hooks on all leaf modules. |
| `device` | `str` | `"cuda"` | `"cuda"` for GPU timing (torch.cuda.Event), `"cpu"` for perf_counter(). |

### Methods

#### `get_summary() → pd.DataFrame`

Return accumulated timing statistics.

| Column | Type | Description |
|--------|------|-------------|
| `module` | `str` | Layer name (e.g., `features.0.conv`) |
| `calls` | `int` | Number of forward passes |
| `avg_time_ms` | `float` | Mean time (milliseconds) |
| `std_time_ms` | `float` | Standard deviation |
| `total_time_ms` | `float` | Sum of all forward passes |

#### `remove() → None`

Unregister all hooks. Call at end of profiling to avoid memory leaks.

### Advanced Usage Example

**Scenario:** Identify energy bottlenecks in ResNet50 on CIFAR-10, correlate layer time with energy consumption.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from monitor_app import (
    LayerEnergyProfiler,
    MonitorContext,
    EpochTracker,
    compute_energy_metrics,
)
import pandas as pd

# Setup
device = torch.device('cuda:0')
model = models.resnet50(pretrained=False, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()

transform = transforms.ToTensor()
train_dataset = datasets.CIFAR10('.', train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, num_workers=4)

# Initialize monitoring
log_path = "logs/resnet50_layer_profile"
tracker = EpochTracker(log_path + ".ndjson")
profiler = LayerEnergyProfiler(model, device="cuda")

with MonitorContext(
    context="resnet50_profiling",
    interval=1.0,
    log_file_path=log_path,
) as mon:
    # Single epoch profiling
    model.eval()
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            tracker.on_epoch_start(0)
            outputs = model(images)
            tracker.on_epoch_end(0, samples=len(labels), accuracy=0.9)
    
    # Get per-layer timings
    layer_timings = profiler.get_summary()
    print(layer_timings)

# Get overall energy metrics
energy_metrics = compute_energy_metrics(log_path + ".ndjson")

# Correlate layer timing with energy
print("\n=== Layer Energy Attribution ===")
total_time_ms = layer_timings['total_time_ms'].sum()
total_energy_j = energy_metrics['energy_j'].iloc[0] if not energy_metrics.empty else 0

layer_timings['energy_fraction'] = (layer_timings['total_time_ms'] / total_time_ms) * total_energy_j
layer_timings = layer_timings.sort_values('energy_fraction', ascending=False)

print(layer_timings[['module', 'avg_time_ms', 'total_time_ms', 'energy_fraction']].head(10))

# Identify bottlenecks
bottleneck_threshold = 0.1  # Top 10%
bottlenecks = layer_timings[layer_timings['energy_fraction'] > bottleneck_threshold * total_energy_j]
print(f"\n=== Energy Bottlenecks (>{bottleneck_threshold*100:.0f}% of total) ===")
for idx, row in bottlenecks.iterrows():
    print(f"  {row['module']:25s} — {row['energy_fraction']:.2f}J "
          f"({row['energy_fraction']/total_energy_j*100:.1f}%)")

profiler.remove()
```

**Expected Output:**

```
                   module  avg_time_ms  total_time_ms
0          layer4.2.conv2         12.5         625.0
1          layer3.5.conv2         10.2         510.0
2          layer4.1.conv1          9.8         490.0
3          layer3.4.conv1          8.5         425.0
4          layer4.0.conv1          7.2         360.0
5          layer2.3.conv2          6.1         305.0
...

=== Energy Bottlenecks (>10% of total) ===
  layer4.2.conv2              103.50J (25.9%)
  layer3.5.conv2               72.30J (18.1%)
  layer4.1.conv1               68.40J (17.1%)
  layer3.4.conv1               45.60J (11.4%)
```

**Insights:** Layer4 (ResNet's final residual block) consumes 56% of total energy — main optimization target.

---

## Terminology Glossary

All documentation uses consistent terminology across code, docs and results:

- **Energy Grade** — Calificación Energética (A++ to F)
- **Energy Intensity Factor** — Factor de Intensidad (J/sample)
- **Pareto Frontier** — Frontera de Pareto (energy-accuracy trade-off)
- **Energy Early Stopping (EES)** — Parada temprana por eficiencia
- **GPU Power Optimizer** — Optimizador de potencia (Zeus-style)
- **Green AI Grade** — Calificación sostenibilidad

---

## Troubleshooting

### Common Issues

**Q: `ModuleNotFoundError: No module named 'nvidia_ml_py'`**
```bash
pip install nvidia-ml-py
pip install pynvml  # May be needed alternatively
```

**Q: `RuntimeError: CUDA out of memory`**
- Reduce `batch_size` in monitoring
- Disable `LayerEnergyProfiler` if memory is critical

**Q: GpuPowerOptimizer fails with "No permission"**
- Power optimization needs sudo; runs in "advisor-only" mode without
- Recommended: `sudo python train.py` or configure passwordless sudo

**Q: EES triggers too early / too late**
- Adjust `min_efficiency_ratio` (lower = more aggressive)
- Or set explicit `min_efficiency` threshold

---

## References

- **GitHub:** https://github.com/BenjyInsights/Monitor_APP
- **Documentation:** https://monitor-app.readthedocs.io
- **Trabajo original:** Sánchez Calza, B. «monitor_app: Framework para Monitorización Energética de IA» (2026)
- **Zeus Framework:** ML.Energy Initiative, University of Michigan
- **Carbon Intensity Data:** Ember 2025 Dataset

---

**Version:** 1.0.0  
**Last Updated:** 2026-04-16  
**License:** GPL-3.0
