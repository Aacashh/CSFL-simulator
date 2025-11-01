# Mac Setup Fix - Channel Mismatch Error

## Problem
You're seeing this error on your Mac:
```
RuntimeError: Given groups=1, weight of size [10, 1, 5, 5], expected input[128, 3, 32, 32] to have 1 channels, but got 3 channels instead
```

## Root Cause
Python cached the OLD bytecode (`.pyc` files) from before the channel-matching fix was added. Your Mac is running the old cached version even though the source code is updated.

## Solution - Follow These Steps IN ORDER

### Step 1: Clean Python Cache (REQUIRED)

From your project directory on Mac:
```bash
cd /Users/advaitpathak/Desktop/CSFL-simulator

# Run the cleaning script
./clean_cache.sh
```

If the script doesn't work, manually run:
```bash
# Remove all cached bytecode
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

echo "Cache cleaned!"
```

### Step 2: Verify the Source Code is Updated

Check that `csfl_simulator/core/models.py` has the channel matching:
```bash
grep -A 3 "def forward" csfl_simulator/core/models.py | head -10
```

You should see:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = _match_channels(x, self.conv1.in_channels)
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
```

If you see the OLD version without `_match_channels`, git pull the latest changes:
```bash
git pull origin main
```

### Step 3: Force Reload and Restart

The updated `main.py` now includes automatic cache-busting. Restart Streamlit:

```bash
# Kill any existing Streamlit processes
pkill -9 streamlit

# Start fresh
streamlit run csfl_simulator/app/main.py
```

### Step 4: Test

1. In the UI, select:
   - Dataset: CIFAR-10
   - Model: CNN-MNIST
   - Clients: 3, Rounds: 2 (small test)
2. Click "Initialize Simulator"
3. Switch to "Run" tab
4. Click "Run"

It should work without the channel error.

## If Still Failing

### Nuclear Option: Force Python to Ignore All Cache
```bash
python -B -m streamlit run csfl_simulator/app/main.py
```

The `-B` flag tells Python to not write or read `.pyc` files.

### Check Python Environment
Ensure you're in the correct conda environment:
```bash
conda activate csfl-env
which python
# Should show: /Users/advaitpathak/anaconda3/envs/csfl-env/bin/python
```

### Reinstall the Package (if using editable install)
```bash
pip uninstall csfl-simulator -y
pip install -e .
```

## Why This Happened

1. You ran the old code → Python compiled `models.py` to bytecode → saved in `__pycache__/models.cpython-310.pyc`
2. We updated `models.py` source to add `_match_channels` 
3. You ran again → Python saw the cached `.pyc` file timestamp and loaded it INSTEAD of recompiling from the updated source
4. Result: Old forward method (without channel matching) was executed

## Prevention

After every git pull or major code update:
```bash
./clean_cache.sh && streamlit run csfl_simulator/app/main.py
```

## Verification

After fixing, verify the code is using the NEW version by checking the traceback.
If you still get an error, the line number should be **>42** (because `_match_channels` is at line 42).
If the traceback shows line 17 or line 43 in the OLD numbering, cache is still stale.

## Contact
If none of this works, check:
1. Are you editing the right file? (`/Users/advaitpathak/Desktop/CSFL-simulator/`)
2. Is there a second copy of the repo somewhere?
3. Is there a system-wide install of csfl-simulator conflicting with the local one?

Run:
```bash
python -c "import csfl_simulator.core.models; print(csfl_simulator.core.models.__file__)"
```

This shows which file Python is actually loading. It should point to your Desktop copy.

