# Frequently Asked Questions (FAQ)

## Hardware Telemetry & Permissions

### Q: Why do I get a permission error when trying to read CPU power?
**A:** Reading Intel RAPL telemetry via sysfs (`/sys/class/powercap/intel-rapl/...`) requires read permissions on Linux. You can grant read permissions to all users with a single command:
```bash
sudo chmod a+r /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
```
If you do not have root access, `monitor_app` will fallback gracefully to estimating CPU consumption using TDP thermal limits.

### Q: Why is the GPU power optimizer in "ADVISOR-ONLY" mode?
**A:** Changing the GPU power limit (`nvmlDeviceSetPowerManagementLimit`) requires elevated privileges (`sudo` / root). If you run your script without root, `monitor_app` degrades gracefully to advisor mode: it constructs the Pareto frontier and outputs suggestions to the terminal without altering hardware limits. To run in ACTIVE mode, execute your training script with `sudo`:
```bash
sudo env PATH=$PATH python train.py
```

### Q: I get `ImportError: No module named 'pynvml'` or `nvidia-ml-py` errors
**A:** Make sure you installed the GPU-specific dependencies. Run:
```bash
pip install nvidia-ml-py pynvml
```

---

## Unit Testing & Mocking Issues

### Q: Why do I see failures in `TestCarbonEmissionsExternalAPI` when running pytest?
**A:** The tests fail with `AttributeError: module 'monitor_app.utils.carbon_emissions' has no attribute 'urllib'`. This is a mocking bug in the test suite. 

The test tries to patch `monitor_app.utils.carbon_emissions.urllib.request.urlopen`. However, in `carbon_emissions.py`, `urllib` is imported inside the local function body rather than at the module level:
```python
# In carbon_emissions.py:
def _query_electricity_maps_api(cls, country: str):
    ...
    import urllib.request
    import urllib.error
```
**Fix:** To resolve this, you can edit `tests/test_carbon_emissions.py` to patch the global `urllib.request.urlopen` instead, or add `import urllib.request` at the top of `src/monitor_app/utils/carbon_emissions.py`.

### Q: Why does `TestIntelCpuMonitor::test_read_metrics` fail?
**A:** The test fails with `AssertionError: unexpectedly None` because the mock file reader `side_effect` is exhausted. 
In `test_read_metrics`, the mock is set up with two values:
```python
mock_file.return_value.__enter__.return_value.read.side_effect = [
    "50000000\n",
    "55000000\n",
]
```
However, the test calls `initialize()`, which performs the first read to establish a baseline. Then it calls `read_metrics()` twice, resulting in three total calls to `_read_energy_uj()`.
**Fix:** To fix this, provide three values in the side effect array within `tests/test_energy_monitors.py`:
```python
mock_file.return_value.__enter__.return_value.read.side_effect = [
    "50000000\n",
    "55000000\n",
    "60000000\n",
]
```

### Q: Why does `TestIntelCpuMonitor::test_read_metrics_unavailable` fail?
**A:** The test fails with `RuntimeError: [CPU-Package-0] Must call initialize() first` because the monitor is queried without being initialized.
**Fix:** In `tests/test_energy_monitors.py`, call `monitor.initialize()` (or set `monitor._initialized = True`) before calling `read_metrics()` in `test_read_metrics_unavailable`:
```python
    def test_read_metrics_unavailable(self) -> None:
        monitor = IntelCpuMonitor(package_index=0)
        monitor.initialize()  # Initialize first
        monitor._mode = MonitorMode.UNAVAILABLE
        reading = monitor.read_metrics()
        self.assertIsNone(reading)
```
