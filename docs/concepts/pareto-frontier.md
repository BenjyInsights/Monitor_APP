# Pareto Frontier

## Definition

A **Pareto Frontier** (or Pareto Optimal Set) is the set of configurations where no single configuration strictly dominates all others in both dimensions simultaneously.

In the context of AI training:

> **Energy-Accuracy Pareto Frontier:** The curve/cloud of model configurations where improving accuracy requires accepting higher energy consumption, and vice versa.

## Visual Example

```
       Accuracy
          ↑
       100%│
          │     * (ResNet50, 95%, 10kJ)
           │    ╱
        95%│   * (VGG19, 93%, 8kJ)
          │  ╱  ╲
        90%│ * (MobileNetV2, 88%, 2kJ)
          │            * (EfficientNetB0, 87%, 3kJ)
        85%│
          │
          └─────────────────────→ Energy (kJ)
            2kJ    5kJ    10kJ

Points on the diagonal are on the Pareto Frontier.
Points below/right are dominated (worse in both dimensions).
```

## Why Pareto?

No single architecture or configuration configuration optimizes both energy *and* accuracy simultaneously. Instead:

- **VGG19** achieves low energy but moderate accuracy
- **ResNet50** achieves high accuracy but consumes more energy
- **MobileNetV2** falls below the frontier (poor both dimensions)

Workers must choose a trade-off point depending on priorities:

| Priority | Choice |
|----------|--------|
| Max accuracy | ResNet50 (A+, 95%, 10kJ) |
| Min energy | VGG19 (A, 93%, 8kJ) |
| Balanced | ResNet18 (A, 91%, 6kJ) |

## Frontera de Pareto (Terminology)

In the thesis context, *"Frontera de Pareto"* refers specifically to the energy-accuracy frontier and the mechanism (**GPU Power Optimizer**) that automatically explores it:

> *"la limitación dinámica de potencia GPU ... reduce el consumo entre un 20 y un 40% explorando la frontera de Pareto energía-exactitud"*

Translation: "GPU power limiting ... reduces consumption by 20–40% by exploring the energy-accuracy Pareto frontier."

## Practical Use in moniaenergy

When `power_optimize=True` is enabled:

```python
with monitor_train(
    model=model,
    power_optimize=True,  # Auto-explore Pareto frontier
    time_budget_pct=0.10, # Tolerate 10% more time for energy savings
):
    # Framework adjusts GPU power limits each epoch
    # to find configurations on the frontier
```

The optimizer reports which point it selected and **why**, with natural-language justification:

```
GPU Power Adjustment (Epoch 3):
  Explored 5 power levels: 150W, 180W, 210W, 240W, 270W
  Selected 180W — Sweet spot on Pareto frontier
  Justification: Accuracy impact negligible (< 0.5%)
                 Energy savings: 28% vs. full power
```

---

See also: [Energy Grading](energy-grading.md), [GPU Power Optimization](../user-guide/gpu-optimization.md)
