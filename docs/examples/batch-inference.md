# Example: Batch Inference Benchmark

While `monitor_train` is optimized for training loops, you can use the lower-level **`MonitorContext`** to benchmark inference latency and energy consumption per request batch.

## Complete Inference Script

Save this script as `benchmark_inference.py`:

```python
import torch
import torchvision.models as models
from monitor_app.monitor.inline_monitor import MonitorContext

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(device)
model.eval()

# Prepare dummy batch (size 32)
dummy_input = torch.randn(32, 3, 224, 224).to(device)

print("Starting inference benchmark...")

# Initialize monitor context
with MonitorContext(label="resnet50_inference_bs32") as ctx:
    # Warmup runs
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Reset telemetry counters for active tracking
    ctx.reset()
    
    # Active benchmarking loop
    num_batches = 100
    for i in range(num_batches):
        with torch.no_grad():
            outputs = model(dummy_input)
            
# Print collected statistics
reading = ctx.get_reading()
if reading:
    total_energy = reading.total_energy_j
    avg_power = reading.average_power_w
    duration = reading.duration_seconds
    
    energy_per_sample = total_energy / (num_batches * 32)
    latency_ms = (duration / num_batches) * 1000
    
    print("\n" + "="*40)
    print("INFERENCE SUMMARY:")
    print(f"Total Batches:      {num_batches}")
    print(f"Duration:           {duration:.3f} s")
    print(f"Average Latency:    {latency_ms:.2f} ms/batch")
    print(f"Average Power Draw: {avg_power:.1f} W")
    print(f"Total Energy:       {total_energy:.2f} Joules")
    print(f"Energy per Sample:  {energy_per_sample:.6f} J/sample")
    print("="*40)
```

## Running the Benchmark

Execute the script:
```bash
python benchmark_inference.py
```

This is ideal for evaluating the deployment footprint of different models (e.g., comparing FP32 vs FP16 or INT8 quantized variants) under target hardware constraints.
