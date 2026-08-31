# Data Organization — monIAenergy v1.0

This directory structure organizes raw monitoring logs and processed metrics for reproducibility.

## Directory Structure

```
data/
├── raw_logs/
│   ├── CIFAR10_ResNet18_cuda0_bs128_fp16/
│   │   ├── run_20260416_140000.ndjson          # Raw NDJSON monitoring log
│   │   ├── run_20260416_140000.events.ndjson   # Epoch boundary events (sidecar)
│   │   ├── run_20260416_140000_energy_metrics.csv  # Pre-computed energy aggregation
│   │   └── run_20260416_140000_layer_profile.csv   # Per-layer timing (if profiling enabled)
│   └── ...
│
└── processed_metrics/
    ├── aggregated_metrics.csv     # Master table: all runs aggregated
    ├── pareto_frontier.csv        # Pareto-optimality analysis
    ├── grade_distribution.json    # Energy Grade histogram
    ├── by_model/                  # Model-specific summaries
    │   ├── ResNet18_summary.csv
    │   ├── ResNet50_summary.csv
    │   └── ...
    └── by_batch_size/             # Batch-size impact analysis
        ├── bs32_summary.csv
        ├── bs64_summary.csv
        └── ...
```

## File Formats

### raw_logs/*.ndjson

**Newline-Delimited JSON** — One energy/hardware event per line.

```json
{"timestamp": 1713271200.123, "epoch": 0, "cpu_energy_j": 45.2, "gpu_energy_j": 342.1, "power_w": 387.3, "temp_c": 62.5}
{"timestamp": 1713271210.456, "epoch": 0, "cpu_energy_j": 52.1, "gpu_energy_j": 387.5, "power_w": 390.1, "temp_c": 63.2}
...
```

### raw_logs/*.events.ndjson (Sidecar)

**Epoch boundary markers** — Separate file for event tracking.

```json
{"event": "epoch_start", "epoch": 0, "timestamp": 1713271200.000}
{"event": "epoch_end", "epoch": 0, "timestamp": 1713271289.000, "samples": 50000, "loss": 2.302, "accuracy": 0.108}
{"event": "epoch_start", "epoch": 1, "timestamp": 1713271290.000}
...
```

### processed_metrics/aggregated_metrics.csv

Master table summarizing all training runs:

```
model,batch_size,num_params,epochs_completed,total_energy_j,co2_grams,avg_accuracy,energy_grade,intensity_j_sample
ResNet18,32,11173962,200,8240.5,0.269,0.925,A,0.0329
ResNet18,64,11173962,200,8156.2,0.267,0.927,A,0.0326
ResNet18,128,11173962,200,8023.1,0.262,0.930,A+,0.0321
ResNet18,256,11173962,200,8401.3,0.275,0.928,A,0.0336
ResNet50,32,23512066,150,12450.8,0.407,0.945,B,0.0499
...
```

### processed_metrics/pareto_frontier.csv

Identifies Pareto-optimal configurations (best energy-accuracy trade-off):

```
model,batch_size,energy_j,accuracy,on_frontier,dominance_ratio
ResNet18,128,8023.1,0.930,true,1.00
EfficientNetB0,256,5432.1,0.915,true,0.98
MobileNetV2,64,3891.2,0.908,true,0.96
ResNet50,256,12801.5,0.948,false,1.04
...
```

## Usage Examples

### 1. Export logs from monitoring directory to raw_logs/

```bash
python export_to_zip.py --logs_dir logs/ --output_dir data/raw_logs/ --format ndjson
```

### 2. Generate aggregated metrics

```python
from tools.analysis_utils import aggregate_metrics

aggregate_metrics(
    logs_dir="logs/",
    output_file="data/processed_metrics/aggregated_metrics.csv"
)
```

### 3. Analyze Pareto frontier

```python
import pandas as pd

metrics = pd.read_csv("data/processed_metrics/aggregated_metrics.csv")
pareto = metrics[metrics['on_frontier'] == True]
print(pareto[['model', 'batch_size', 'energy_j', 'accuracy']])
```

## Data Retention Policy

- **raw_logs/**: Keep for ~30 days (debugging), then archive
- **processed_metrics/**: Keep for thesis submission + publication (~5 years)

## Contact

For data organization questions: sanchezcalzabenjamin@gmail.com

---

**Version:** 1.0.0  
**Autor:** Benjamín Sánchez Calza · 2026
