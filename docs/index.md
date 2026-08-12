# Home

Welcome to **monitor_app** ⚡🌍 — a high-fidelity energy monitoring framework for Python & PyTorch.

## What is monitor_app?

`monitor_app` measures with precision the energy consumption (CPU + GPU), carbon footprint by country, and training efficiency of AI models in Python. It's designed to:

- **Measure accurately** — *"How much energy does my model really consume?"*
- **Grade automatically** — Classify efficiency as **A++, A+, A, B, C, D, E, F** (like appliance energy labels)
- **Optimize actively** — Seek the **Pareto Frontier** between energy and time (Zeus-style GPU power limiting)
- **Suggest intelligently** — Emit contextual optimization hints when misconfigurations are detected
- **Integrate easily** — Single-line integration with `monitor_train()`

## Key Features

✨ **Energy Monitoring**
- Per-epoch energy tracking (CPU + GPU power)
- Real-time GPU power, VRAM, and temperature metrics
- Carbon emissions estimation by country (Ember 2025 dataset)

🎯 **Green AI Grading**
- Intuitive A++–F letter grades based on efficiency/accuracy trade-off
- Hardware-agnostic reference normalization
- Pareto frontier analysis

⚡ **Optimization Tools**
- **Energy Early Stopping (EES)**: Halt training when energy per accuracy improvement stagnates (30–50% savings)
- **GPU Power Optimizer**: Dynamic power limiting on Pareto frontier (Zeus-style, 20–40% savings)
- **Layer-wise Profiling**: Identify energy hotspots in your neural network

📊 **Rich Output**
- Live terminal dashboard (Rich library)
- Markdown executive summaries
- Export to CSV, NDJSON for further analysis
- Professional publication-ready plots (300 DPI, print colors)

## Quick Example

```python
from monitor_app import monitor_train

with monitor_train(
    model=my_model,
    experiment_name="resnet50_cifar10",
    country="Spain",
    batch_size=128,
    early_stopping=True,
    power_optimize=True,
    gpu_index=0,
) as mon:
    for epoch in range(num_epochs):
        mon.epoch_start(epoch)
        loss, acc = train_one_epoch(loader, model, optimizer)
        if mon.epoch_end(epoch, samples=len(dataset), loss=loss, accuracy=acc):
            break  # Energy Early Stopping triggered
```

**Output:**

```
══════════════════════════════════════════════════════════════════
  monitor_train — Final Report: resnet50_cifar10
══════════════════════════════════════════════════════════════════
  Energy Grade:         A+   (450% of reference)
  J/sample (avg):       0.0342
  Total energy (avg/ep):1.94 kJ
  CO₂ estimate (Spain): 0.08 g
  Log:                  logs/resnet50_cifar10/run_20260416_102030.ndjson
══════════════════════════════════════════════════════════════════
```

## Terminology

> **Terminology is aligned across the codebase, the documentation and the published results.**

- **Energy Grade**: A++–F classification capturing efficiency/accuracy trade-offs
- **Energy Early Stopping (EES)**: Training termination when energy marginal benefit stagnates
- **Pareto Frontier**: Trade-off curve between energy and accuracy; no single config dominates
- **Intensity Factor**: Energy consumption per unit of work (J/sample, typically)
- **GPU Power Optimizer**: Auto-scaling GPU power limits to explore Pareto frontier (Zeus-inspired)
- **LayerEnergyProfiler**: Per-layer energy attribution for hotspot detection

## Getting Started

- **[Installation](getting-started/installation.md)** — Set up monitor_app in 5 minutes
- **[Quick Start](getting-started/quickstart.md)** — First energy monitoring run
- **[Reproducibility Guide](getting-started/reproducibility.md)** — Replicate the published benchmarks step-by-step

## Documentation

- **[API Reference](api/monitor_app.md)** — Full class and function documentation
- **[Concepts](concepts/energy-grading.md)** — Deep dive into energy grading and optimization theories
- **[Examples](examples/training.md)** — Real-world usage patterns

## Citation

If you use `monitor_app` in your research, please cite:

```bibtex
@thesis{SanchezCalza2026,
  title={monitor\\_app: Framework para Monitorización Energética de IA},
  author={Sánchez Calza, Benjamín},
  year={2026},
  school={UNED}
}
```

## License

`monitor_app` is licensed under the **GNU General Public License v3.0**. See [LICENSE](LICENSE.md) for details.

---

**Last updated:** 2026-04-16  
**Version:** 1.0.0
