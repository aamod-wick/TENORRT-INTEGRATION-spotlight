#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Create and activate virtual environment
echo "[INFO] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Step 2: Upgrade pip and install requirements
echo "[INFO] Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt
echo "[INFO] Setup complete. Virtual environment is ready."
