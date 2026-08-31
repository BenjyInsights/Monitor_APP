# Example: Multi-GPU Setups

When working in environments with multiple GPUs (e.g., SLURM clusters or multi-socket workstations), you need to ensure `moniaenergy` tracks and optimizes the correct hardware device.

## Selecting a Specific GPU

By default, `moniaenergy` queries NVML for GPU `0`. In multi-GPU rigs, pass the target GPU index using the `gpu_index` parameter:

```python
from monIAenergy import monitor_train

with monitor_train(
    model=model,
    experiment_name="multi_gpu_run",
    gpu_index=1,             # Track and cap GPU 1 (CUDA:1)
    power_optimize=True,     # Optimize GPU 1
) as mon:
    # Training loop...
```

## Data Parallelism (DP/DDP)

If you are using PyTorch's distributed frameworks:

### 1. `DataParallel` (DP)
`DataParallel` runs on a single process but splits the batch across multiple GPUs.
- **Monitoring**: Instantiate `monitor_train` on the host GPU (usually index `0`). The energy reading will capture the host GPU's power draw, but note that secondary GPUs will draw power that NVML tracks separately.
- **Recommendation**: Wrap multiple `NvidiaGpuMonitor` objects manually if you want to sum power across all active devices.

### 2. `DistributedDataParallel` (DDP)
DDP spawns a separate Python process per GPU.
- **Monitoring**: Ensure only the main rank process (rank `0`) initializes `monitor_train` or controls power limits.
- **Example Pattern**:
```python
import os
from moniaenergy import monitor_train

local_rank = int(os.environ.get("LOCAL_RANK", 0))

# Only run energy monitoring on rank 0 to prevent conflicts
if local_rank == 0:
    mon_context = monitor_train(
        model=model,
        experiment_name="ddp_run",
        gpu_index=local_rank,  # Matches local rank
        power_optimize=True,
    )
else:
    # Null context manager for worker ranks
    from contextlib import nullcontext
    mon_context = nullcontext()

with mon_context as mon:
    for epoch in range(10):
        if local_rank == 0 and mon:
            mon.epoch_start(epoch)
            
        # Distributed DDP train step...
        
        if local_rank == 0 and mon:
            mon.epoch_end(epoch, samples=samples, loss=loss, accuracy=acc)
```

By restricting power limit commands to rank `0`, you avoid race conditions where multiple processes attempt to cap the same GPU hardware limit concurrently.
