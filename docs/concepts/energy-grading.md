# Energy Grading System

## Overview

The **Energy Grade** (A++–F) is a universal, hardware-agnostic metric that captures the quality of a model configuration from a sustainability perspective.

## Grading Formula

```
Efficiency Score = (Accuracy × log₁₀(Parameters)) / Total Energy (J)
```

The score is then compared against tier thresholds, expressed as percentages of a reference baseline:

| Grade | Efficiency Score | Interpretation |
|-------|------------------|-----------------|
| **A++** | ≥ 800% of reference | Top-of-class energy efficiency |
| **A+** | ≥ 400% | Excellent efficiency |
| **A** | ≥ 200% | Very good |
| **B** | ≥ 100% (reference) | Good; industry baseline |
| **C** | ≥ 50% | Acceptable |
| **D** | ≥ 20% | Poor |
| **E** | ≥ 5% | Very poor |
| **F** | < 5% | Unacceptable |

## Why This Matters

A high-accuracy model consuming enormous energy receives a low grade (e.g., C or D), signaling to researchers that while results are good, sustainability should be improved. Conversely, a lightweight model achieving excellent results earns an A++ grade.

This avoids:
- ❌ Hardcoding CIFAR-10-specific thresholds
- ❌ Hardware-dependent comparisons
- ❌ Ignoring energy cost in favor of pure accuracy

## Practical Example

### Configuration A: ResNet50 on CIFAR-10
- **Accuracy:** 95%
- **Parameters:** 23.5M
- **Energy:** 10 kJ
- **Score:** (0.95 × log₁₀(23.5M)) / 10 = (0.95 × 7.37) / 10 = **0.70**
- **Grade:** A (if reference is 0.35)

### Configuration B: MobileNetV2 on CIFAR-10
- **Accuracy:** 92%
- **Parameters:** 2.2M
- **Energy:** 2 kJ
- **Score:** (0.92 × log₁₀(2.2M)) / 2 = (0.92 × 6.33) / 2 = **2.91**
- **Grade:** A++ (if reference is 0.35)

**Interpretation:** MobileNetV2 achieves slightly lower accuracy but consumes far less energy, earning a premium grade for its energy efficiency.

## Reference Calibration

The reference score (B grade threshold) is auto-calibrated from your benchmark dataset. monIAenergy:

1. Computes scores for all runs in your logs/
2. Selects the median as the reference
3. Derives all tier thresholds from that reference

This ensures grades adapt to your hardware and dataset, not to hardcoded global benchmarks.

## When to Use

- **Model Selection:** Prefer high-grade configurations for production deployments
- **Optimization Tracking:** Monitor if changes improve or degrade the grade
- **Reporting:** Communicate energy efficiency to stakeholders intuitively (A-F is universally understood)

---

See also: [Pareto Frontier](pareto-frontier.md), [Intensity Factor](intensity-factor.md)
