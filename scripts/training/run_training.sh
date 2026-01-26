#!/bin/bash
# OhanaAI - Training Pipeline Runner
# Runs the complete training pipeline for missing relative prediction

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
DATA_DIR="${PROJECT_ROOT}/training_data"
OUTPUT_DIR="${PROJECT_ROOT}/models/family_tree_gnn"
EPOCHS=100
LEARNING_RATE=0.001
HIDDEN_DIM=128
NUM_LAYERS=3
NUM_HEADS=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --hidden-dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --export-only)
            EXPORT_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --data-dir DIR      Training data directory (default: training_data)"
            echo "  --output-dir DIR    Output directory for model (default: models/family_tree_gnn)"
            echo "  --epochs N          Number of training epochs (default: 100)"
            echo "  --lr RATE           Learning rate (default: 0.001)"
            echo "  --hidden-dim N      Hidden dimension (default: 128)"
            echo "  --export-only       Only export ONNX model, skip training"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "OhanaAI - Missing Relative Prediction Training"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Data directory:  $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Epochs:          $EPOCHS"
echo "  Learning rate:   $LEARNING_RATE"
echo "  Hidden dim:      $HIDDEN_DIM"
echo ""

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

# Create output directory
mkdir -p "$OUTPUT_DIR"

if [ "$EXPORT_ONLY" = true ]; then
    echo "Exporting ONNX model only..."
    python3 "${SCRIPT_DIR}/export_onnx.py" \
        --simple \
        --output "${OUTPUT_DIR}/model.onnx" \
        --hidden-dim "$HIDDEN_DIM" \
        --version "1.0.0"
else
    # Run training
    echo "Starting training..."
    python3 "${SCRIPT_DIR}/train.py" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --epochs "$EPOCHS" \
        --lr "$LEARNING_RATE" \
        --hidden-dim "$HIDDEN_DIM" \
        --num-layers "$NUM_LAYERS" \
        --num-heads "$NUM_HEADS"

    # Export to ONNX
    echo ""
    echo "Exporting to ONNX format..."
    if [ -f "${OUTPUT_DIR}/best_model.npz" ]; then
        python3 "${SCRIPT_DIR}/export_onnx.py" \
            --checkpoint "${OUTPUT_DIR}/best_model.npz" \
            --output "${OUTPUT_DIR}/model.onnx" \
            --hidden-dim "$HIDDEN_DIM" \
            --version "1.0.0"
    else
        echo "No checkpoint found, exporting simple model..."
        python3 "${SCRIPT_DIR}/export_onnx.py" \
            --simple \
            --output "${OUTPUT_DIR}/model.onnx" \
            --hidden-dim "$HIDDEN_DIM"
    fi
fi

echo ""
echo "=============================================="
echo "Training complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "=============================================="
