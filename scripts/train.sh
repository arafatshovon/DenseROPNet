#!/bin/bash
# =============================================================================
# Training Script for ROP Detection Model
# =============================================================================
# This script trains the DenseNet121-SE-RANB-DualPool model for ROP detection.
#
# Usage:
#   ./scripts/train.sh                              # Use default config
#   ./scripts/train.sh /path/to/data                # Specify data directory
#   ./scripts/train.sh /path/to/data 100 16 0.0001  # Full customization
#
# Arguments:
#   $1 - Data directory (default: from config)
#   $2 - Number of epochs (default: 50)
#   $3 - Batch size (default: 32)
#   $4 - Learning rate (default: 0.0001)
#   $5 - GPU ID (default: 0)
# =============================================================================

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

# Parse arguments
DATA_DIR="${1:-}"
EPOCHS="${2:-50}"
BATCH_SIZE="${3:-32}"
LEARNING_RATE="${4:-0.0001}"
GPU_ID="${5:-0}"

# Create timestamp for experiment
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME="rop_training_${TIMESTAMP}"

echo "=============================================================="
echo "ROP Detection Model Training"
echo "=============================================================="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Timestamp: $(date)"
echo "=============================================================="

# Build command
CMD="python train.py --config configs/config.yaml"
CMD="${CMD} --epochs ${EPOCHS}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --learning_rate ${LEARNING_RATE}"
CMD="${CMD} --gpu ${GPU_ID}"

if [ -n "${DATA_DIR}" ]; then
    CMD="${CMD} --data_dir ${DATA_DIR}"
fi

# Create output directory
OUTPUT_DIR="experiments/${EXPERIMENT_NAME}"
mkdir -p "${OUTPUT_DIR}"
CMD="${CMD} --output_dir ${OUTPUT_DIR}"

echo ""
echo "Command: ${CMD}"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo "=============================================================="
echo ""

# Run training
${CMD} 2>&1 | tee "${OUTPUT_DIR}/training.log"

echo ""
echo "=============================================================="
echo "Training completed!"
echo "Model saved to: ${OUTPUT_DIR}/checkpoints/"
echo "Logs saved to: ${OUTPUT_DIR}/logs/"
echo "=============================================================="

