# Contributing Guide

Thank you for your interest in contributing to **monitor_app**! As an open-source project, we welcome contributions that improve hardware support, accuracy calibration, active optimization, or documentation.

## Development Setup

To set up a local development environment:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/BenjyInsights/monitor_app.git
   cd monitor_app
   ```

2. **Create a virtual environment**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. **Install in development mode with all optional dependencies**:
   ```bash
   pip install -e .[dev,gpu,docs]
   ```

## Code Quality Standards

We enforce strict linting and style rules to keep the codebase clean and maintainable:
- **Formatting**: We use `black` for formatting and `isort` for imports sorting.
- **Linting**: We use `ruff` to check for syntax issues and code quality warnings.
- **Static Types**: We use `mypy` for static type checking.

Before submitting any changes, please run:
```bash
black src/ tests/
ruff check src/ tests/
mypy src/
```

## Running the Test Suite

We use `pytest` for unit testing. To execute tests:

```bash
pytest tests/ -v
```

### Note on Mocking Hardware
The test suite utilizes mocks (`unittest.mock`) to simulate CPU RAPL files and GPU NVML interfaces so that tests can run successfully on any machine without root privileges or specific NVIDIA hardware.

---
**License:** By contributing, you agree that your contributions will be licensed under the GPL-3.0 License.
