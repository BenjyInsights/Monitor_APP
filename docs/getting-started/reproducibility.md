# Reproducibility Guide

> **Goal:** Replicate exactly the published benchmark results using `moniaenergy`.

## Prerequisites

- **Python:** 3.11
- **GPU:** NVIDIA RTX Ada (recommended for comparison)
- **Dataset:** CIFAR-10 (auto-downloaded by PyTorch)
- **Time:** ~2 hours per configuration

## Step-by-Step Replication

### 1. Clone and Install

```bash
git clone https://github.com/BenjyInsights/monIAenergy.git
cd monIAenergy

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install with GPU support
pip install -e .[gpu]

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 2. Download & Prepare Data

```bash
mkdir -p data/cifar10

# CIFAR-10 will auto-download on first run, or:
python -c "
from torchvision import datasets
datasets.CIFAR10('data/cifar10', train=True, download=True)
"
```

### 3. Run a Single Benchmark Configuration

**ResNet18 with batch size 128 (Control mode, no optimization):**

```python
# benchmark_single.py
import torch
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn

from monIAenergy import monitor_train

# Load data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    ),
])

train_dataset = datasets.CIFAR10('data/cifar10', train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

# Load model
device = torch.device('cuda:0')
model = models.resnet18(pretrained=False, num_classes=10).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Schedule
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

# ===== Energy Monitoring =====
with monitor_train(
    model=model,
    experiment_name="resnet18_cifar10_bs128_control",
    country="Spain",
    batch_size=128,
    fp16=False,
    early_stopping=False,        # Control: no optimization
    power_optimize=False,
) as mon:
    for epoch in range(50):  # Full benchmark training length
        mon.epoch_start(epoch)
        
        model.train()
        correct = 0
        total = 0
        total_loss = 0
        
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
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / total
        
        # Report to monitor
        should_stop = mon.epoch_end(
            epoch=epoch,
            samples=total,
            loss=avg_loss,
            accuracy=accuracy / 100,  # Convert to 0-1 range
        )
        
        scheduler.step()
        
        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

print("✓ Benchmark complete. Check logs/ for results.")
```

Run it:

```bash
python benchmark_single.py
```

### 4. Analyze Results

```python
# analyze.py
import pandas as pd
import json

log_path = "logs/resnet18_cifar10_bs128_control/run_*.ndjson"

# Load NDJSON
df = pd.read_json(log_path, lines=True)

# Extract key metrics
print(f"Total energy: {df['energy_j'].sum():.2f} J")
print(f"Avg J/sample: {df['energy_j_per_sample'].mean():.6f}")
print(f"Energy grade: {df['grade'].iloc[-1]}")  # Final grade
print(f"RGB CO₂ (Spain): {df['carbon_g'].sum():.3f} g")
```

### 5. Replicate All 84 Configurations

The benchmark spans:

- **6 models:** ResNet18, ResNet50, VGG19, EfficientNetB0, MobileNetV2, DenseNet121
- **4 batch sizes:** 32, 64, 128, 256
- **3 repetitions:** Randomness in initialization
- **2 modes:** Control (no optimization), Optimized (EES + power limiting)

Script to automate:

```python
# run_all_benchmarks.py
from itertools import product
import subprocess

models = ['resnet18', 'resnet50', 'vgg19', 'efficientnet_b0', 'mobilenet_v2', 'densenet121']
batch_sizes = [32, 64, 128, 256]
modes = ['control', 'optimized']
repetitions = 3

for model, bs, mode, rep in product(models, batch_sizes, modes, repetitions):
    experiment_name = f"{model}_cifar10_bs{bs}_{mode}_rep{rep}"
    
    print(f"Starting: {experiment_name}")
    
    # (Launch training with monitor_train)
    # ...
    
    print(f"Completed: {experiment_name}")
```

## Expected Output Structure

```
logs/
├── resnet18_cifar10_bs128_control_rep0/
│   ├── run_20260416_102030.ndjson          # Raw metrics
│   ├── run_20260416_102030.events.ndjson   # Epoch events
│   ├── run_20260416_102030_energy_metrics.csv
│   └── run_20260416_102030_layer_profile.csv
├── resnet18_cifar10_bs128_control_rep1/
│   └── ...
├── ...
└── [71 more configurations]
```

## Validation

After all runs, verify:

1. **No missing runs:** Should have 72 × 3 = 216 runs total
2. **Energy ranges:** Check that median J/sample falls in expected range (0.01–0.05)
3. **Grades:** All should be A–C (within reference calibration)
4. **CO₂ sums:** Compare Spain values to thesis document

## Generating the Executive Summary

After all benchmarks:

```bash
python tools/generate_benchmark_analysis.py --input logs/ --output results/
```

This generates:
- `experiment_summary.md` — Markdown executive summary
- `energy_comparison.csv` — Aggregated metrics
- `pareto_frontier.png` — Publication-ready plots (300 DPI)

---

**Notes:**

- Times may vary depending on GPU (reference hardware: RTX 6000 Ada)
- CO₂ values will differ if using a different country
- Randomness in training may cause ±2% variance in final accuracy

See: [Advanced Monitoring](../user-guide/advanced.md) for layer profiling and optimization tuning.
