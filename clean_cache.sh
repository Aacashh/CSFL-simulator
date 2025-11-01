#!/bin/bash
# Clean all Python cache files to force module reload

echo "Cleaning Python cache files..."

# Remove __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Remove .pyc files
find . -type f -name "*.pyc" -delete 2>/dev/null

# Remove .pyo files
find . -type f -name "*.pyo" -delete 2>/dev/null

# Remove egg-info if you want a full clean (optional)
# find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null

echo "Cache cleaned successfully!"
echo ""
echo "Now run: streamlit run csfl_simulator/app/main.py"

