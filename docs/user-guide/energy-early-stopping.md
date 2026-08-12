# Energy Early Stopping (EES)

Traditional early stopping monitors a validation metric (like validation loss) and halts training when it stops improving. However, it ignores the *energy rate* at which that improvement is happening. 

**Energy Early Stopping (EES)** introduces an energy-aware criterion: it halts training when the marginal gain in accuracy per unit of energy consumed falls below a threshold.

## How EES Works

The stopping criterion checks the efficiency ratio at epoch $t$:

$$
\text{Efficiency Ratio} = \frac{\Delta \text{Accuracy}}{\Delta \text{Energy (J)}}
$$

Where:
- $\Delta \text{Accuracy} = \text{Accuracy}_t - \text{Accuracy}_{t-\text{patience}}$
- $\Delta \text{Energy}$ is the total energy consumed (in Joules) over those epochs.

If this ratio falls below `min_efficiency_ratio` (default: $1.0 \times 10^{-6}$ %/J), it signals that training has reached the **diminishing returns phase**. The energy required to squeeze out another 0.1% accuracy is no longer environmentally or financially viable, and training is terminated.

## Configuration

To enable EES in `monitor_train`:

```python
with monitor_train(
    model=model,
    experiment_name="ees_run",
    early_stopping=True,     # Enable EES
    patience=3,              # Wait 3 epochs without significant efficiency improvement
    min_efficiency_ratio=1e-6 # Set minimum efficiency threshold
) as mon:
    for epoch in range(50):
        mon.epoch_start(epoch)
        # Train...
        # If mon.epoch_end returns True, halt training
        if mon.epoch_end(epoch, samples=samples, loss=loss, accuracy=accuracy):
            print("Energy Early Stopping triggered!")
            break
```

## Empirical Impact

Our experiments validated the efficacy of EES across 24 configurations:
- **Average Energy Savings**: **64.55%** reduction in total training energy.
- **Accuracy Trade-off**: An average loss of only **4.33 percentage points**.
- **Hypothesis Testing**: The Mann-Whitney U test confirmed that the difference in energy consumption between the Control and Full Optimized modes is **highly significant** ($p = 2 \times 10^{-6}$), while the difference in validation accuracy is **not statistically significant** ($p = 0.085$).

### Warning on Pre-convergence
If a model has slow initial convergence (such as EfficientNetB0 with batch size 32), EES may trigger too early (e.g., at epoch 9), leading to a high accuracy loss (e.g., -12.03%). 
To prevent this, increase the `patience` parameter or add a warmup phase where EES is disabled for the first $N$ epochs.

---
**Next:** Read about JIT [GPU Power Optimization](gpu-optimization.md) (Zeus-style).
