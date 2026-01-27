#!/bin/bash
# OhanaAI - Model Training Script
#
# This script runs the unified training pipeline.
# Use retrain.py directly for more control.
#
# Usage:
#   ./run_training.sh                    # Train with existing data
#   ./run_training.sh data.ged           # Prepare data and train
#   ./run_training.sh --epochs 200       # Custom training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    exit 1
fi

# Check for MLX
if ! python3 -c "import mlx" 2>/dev/null; then
    echo "MLX not found. Installing dependencies..."
    pip3 install -r "${SCRIPT_DIR}/requirements.txt"
fi

# Run the unified training script
python3 "${SCRIPT_DIR}/retrain.py" "$@"
