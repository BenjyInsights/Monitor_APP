# Installation

## System Requirements

- **Python:** 3.11 or later
- **Operating System:** Linux (primary), macOS (limited GPU support), Windows (WSL2 recommended)
- **For GPU monitoring:** NVIDIA GPU with CUDA support + nvidia-ml-py

## Basic Installation

### 1. Clone and Navigate

```bash
git clone https://github.com/BenjyInsights/monitor_app.git
cd monitor_app
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python3.11 -m venv venv
source venv/bin/activate

# Or using conda
conda create -n monitor_app python=3.11
conda activate monitor_app
```

### 3. Install Core Package

```bash
# Development mode (editable install with core dependencies)
pip install -e .

# Production mode
pip install .
```

### 4. (Optional) CPU Power Monitoring (Intel RAPL)

To enable per-process CPU power measurement on Intel systems:

```bash
# One-time setup — grants read permission to Intel RAPL
sudo chmod a+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
```

> Without this step, CPU power metrics will be estimated via TDP scaling.

## GPU Monitoring (NVIDIA)

### Install PyTorch + CUDA Dependencies

**Option A: Automatic (recommended)**

```bash
pip install -e [gpu]
```

**Option B: Manual with specific CUDA version**

```bash
# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Then install monitor_app
pip install -e .
```

**Option C: From requirements file**

```bash
pip install -r requirements-gpu.txt
```

### Verify NVIDIA Setup

```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

## Development Installation

For contributors and developers:

```bash
pip install -e .[dev]
```

This installs:
- pytest, pytest-cov — testing framework
- black, ruff, isort — code formatting & linting
- mypy — static type checking

#### Run Tests

```bash
pytest tests/ -v
```

## Documentation Setup (Optional)

To build and serve documentation locally:

```bash
pip install -e .[docs]

# Build HTML
mkdocs build

# Serve locally at http://127.0.0.1:8000
mkdocs serve
```

## Verify Installation

```bash
# Check import
python -c "from monitor_app import monitor_train; print('✓ monitor_app imported successfully')"

# Check CLI
monitor_app --help
```

## Troubleshooting

### ImportError: No module named 'pynvml'

```bash
pip install nvidia-ml-py
```

### RAPL permission denied

```bash
# Re-run the permission setup
sudo chmod a+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj

# Verify
cat /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
```

### CUDA not detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# If False, reinstall PyTorch with correct CUDA version
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

**Next:** See [Quick Start](quickstart.md) for your first energy monitoring run.
