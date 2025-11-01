# Troubleshooting Guide

## Channel Mismatch Error After Updates

### Symptom
```
RuntimeError: Given groups=1, weight of size [10, 1, 5, 5], expected input[128, 3, 32, 32] to have 1 channels, but got 3 channels instead
```

### Cause
Python caches compiled bytecode (`.pyc` files) in `__pycache__` directories. When you update the source code (e.g., adding `_match_channels` to model forward methods), Python may still load the old cached version.

### Solution

#### Option 1: Run the cleaning script (Recommended)
```bash
# From the project root
./clean_cache.sh

# Then restart the app
streamlit run csfl_simulator/app/main.py
```

#### Option 2: Manual cleanup
```bash
# Remove all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove all .pyc files
find . -type f -name "*.pyc" -delete

# Then restart the app
streamlit run csfl_simulator/app/main.py
```

#### Option 3: Force Python to ignore cache (temporary)
```bash
# Run with -B flag to bypass cache
python -B -m streamlit run csfl_simulator/app/main.py
```

### Prevention
After pulling updates from git or modifying core modules (`models.py`, `simulator.py`, etc.), always clean the cache before running.

## Other Common Issues

### "No module named 'csfl_simulator.app.state'"
**Solution:** Use the robust import helper or clean cache as above.

### Visualization shows no data after loading snapshot
**Cause:** Methods/metrics selections don't match snapshot content.
**Solution:** Already fixed - UI auto-syncs to loaded data. If still seeing issues, reload the page.

### Dataset/Model mismatch
The simulator now auto-adapts models to datasets:
- MNIST (1-channel) works with any model (CNN-MNIST, LightCNN, ResNet18)
- CIFAR (3-channel) works with any model
- Channel conversion happens automatically in model forward pass

If you still see errors, clean cache and restart.

