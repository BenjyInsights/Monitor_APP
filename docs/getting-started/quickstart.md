# Quick Start

Get your first energy monitoring run in 5 minutes.

## Minimal Example

```python
from monIAenergy import monitor_train
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Setup: dummy dataset and simple model
X_train = torch.randn(1000, 10)
y_train = torch.randint(0, 2, (1000,))
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=32)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

# ===== Energy Monitoring =====
with monitor_train(
    model=model,
    experiment_name="first_run",
    country="Spain",
    batch_size=32,
) as mon:
    for epoch in range(5):
        mon.epoch_start(epoch)
        
        samples = 0
        total_loss = 0
        for X, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            samples += len(y)
        
        avg_loss = total_loss / samples
        # Dummy accuracy for demo
        accuracy = 0.95 - epoch * 0.01
        
        mon.epoch_end(
            epoch,
            samples=samples,
            loss=avg_loss,
            accuracy=accuracy
        )
```

**Output:**

```
══════════════════════════════════════════════════════════════════
  monitor_train — Final Report: first_run
══════════════════════════════════════════════════════════════════
  Energy Grade:         A   (240% of reference)
  J/sample (avg):       0.0125
  Total energy (avg/ep):0.40 kJ
  CO₂ estimate (Spain): 0.002 g
  Log:                  logs/first_run/run_20260416_102030.ndjson
  
  Recommendation:
  ✓ Energy efficiency is good. Continue current configuration.
══════════════════════════════════════════════════════════════════
```

## Key Parameters

### `monitor_train()`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | `torch.nn.Module` | ✓ | — | The neural network to monitor |
| `experiment_name` | `str` | ✓ | — | Unique identifier for this run |
| `country` | `str` | | `"Spain"` | Country for CO₂ factor estimation |
| `batch_size` | `int` | | Auto-detect | Batch size (for advisor hints) |
| `fp16` | `bool` | | `False` | Using mixed precision? (for advisor) |
| `early_stopping` | `bool` | | `False` | Enable Energy Early Stopping |
| `patience` | `int` | | `3` | EES patience (epochs without improvement) |
| `power_optimize` | `bool` | | `False` | Enable GPU power optimization |
| `gpu_index` | `int` | | `0` | Which GPU to optimize/monitor |
| `time_budget_pct` | `float` | | `0.10` | Tolerate up to 10% more time to save energy |

### `epoch_end()`

```python
mon.epoch_end(
    epoch=0,                    # Epoch number (0-indexed)
    samples=1000,              # Number of samples processed
    loss=0.123,                # Training loss
    accuracy=0.95,             # Accuracy (0-1 or 0-100)
    val_loss=0.150,            # (Optional) Validation loss
    val_accuracy=0.92,         # (Optional) Validation accuracy
)
```

Returns: `bool` — `True` if training should stop (EES triggered), `False` otherwise.

## Output Files

After each run, moniaenergy generates:

```
logs/first_run/
├── run_20260416_102030.ndjson          # Raw per-epoch metrics (NDJSON)
├── run_20260416_102030.events.ndjson   # Epoch boundary events
├── run_20260416_102030_energy_metrics.csv    # Per-epoch energy summary
└── run_20260416_102030_layer_profile.csv     # Per-layer energy breakdown
```

### Reading the NDJSON Log

```python
import pandas as pd
import json

# Load per-epoch metrics
df = pd.read_json('logs/first_run/run_20260416_102030.ndjson', lines=True)
print(df[['timestamp', 'epoch', 'gpu_power_w', 'energy_j', 'carbon_g']])
```

## Next Steps

- Read the [Reproducibility Guide](reproducibility.md) to replicate the published benchmarks
- Explore [Advanced Monitoring](../user-guide/advanced.md) for layer profiling and optimization
- Learn about [Energy Grading](../concepts/energy-grading.md) and how scores are computed

---

**Docs:** [moniaenergy Documentation](../index.md)
