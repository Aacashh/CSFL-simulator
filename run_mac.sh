#!/bin/bash
# Safe startup script for Mac users
# This ensures cache is always clean before running

echo "================================================"
echo "  CSFL Simulator - Mac Startup Script"
echo "================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "csfl_simulator/app/main.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Current directory: $(pwd)"
    echo ""
    echo "   Expected structure:"
    echo "   CSFL-simulator/"
    echo "   ├── csfl_simulator/"
    echo "   │   ├── app/"
    echo "   │   │   └── main.py"
    echo "   │   └── core/"
    echo "   └── run_mac.sh (this script)"
    exit 1
fi

# Check conda environment
if ! command -v conda &> /dev/null; then
    echo "⚠️  Warning: conda not found. Make sure you're in the correct environment."
else
    if [[ "$CONDA_DEFAULT_ENV" != "csfl-env" ]]; then
        echo "⚠️  Warning: Not in csfl-env environment"
        echo "   Current: $CONDA_DEFAULT_ENV"
        echo ""
        echo "   To activate: conda activate csfl-env"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

echo "✓ Directory check passed"
echo ""

# Clean cache
echo "🧹 Cleaning Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
echo "✓ Cache cleaned"
echo ""

# Verify models.py has the fix
echo "🔍 Verifying code is up-to-date..."
if grep -q "_match_channels" csfl_simulator/core/models.py; then
    echo "✓ models.py contains channel-matching fix"
else
    echo "❌ models.py is missing _match_channels!"
    echo "   Please git pull the latest changes"
    exit 1
fi
echo ""

# Start Streamlit
echo "🚀 Starting Streamlit..."
echo "================================================"
echo ""

streamlit run csfl_simulator/app/main.py

