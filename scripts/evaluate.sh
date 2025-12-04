#!/bin/bash
# =============================================================================
# Evaluation Script for ROP Detection Model
# =============================================================================
# This script evaluates a trained ROP detection model.
#
# Usage:
#   ./scripts/evaluate.sh path/to/model.keras
#   ./scripts/evaluate.sh path/to/model.keras /path/to/test/data
#   ./scripts/evaluate.sh path/to/model.keras /path/to/test/data --explain
#
# Arguments:
#   $1 - Path to trained model (required)
#   $2 - Path to test data directory (optional)
#   $3 - Additional flags (e.g., --explain)
# =============================================================================

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

# Check arguments
if [ -z "$1" ]; then
    echo "Error: Please provide model path"
    echo "Usage: ./scripts/evaluate.sh path/to/model.keras [data_dir] [flags]"
    exit 1
fi

MODEL_PATH="$1"
DATA_DIR="${2:-}"
EXTRA_FLAGS="${3:-}"

# Create timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="evaluation_${TIMESTAMP}"

echo "=============================================================="
echo "ROP Detection Model Evaluation"
echo "=============================================================="
echo "Model: ${MODEL_PATH}"
echo "Timestamp: $(date)"
echo "=============================================================="

# Build command
CMD="python evaluate.py --model ${MODEL_PATH}"
CMD="${CMD} --output_dir ${OUTPUT_DIR}"

if [ -n "${DATA_DIR}" ]; then
    CMD="${CMD} --data_dir ${DATA_DIR}"
fi

if [ -n "${EXTRA_FLAGS}" ]; then
    CMD="${CMD} ${EXTRA_FLAGS}"
fi

echo ""
echo "Command: ${CMD}"
echo ""

# Run evaluation
${CMD}

echo ""
echo "=============================================================="
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}/"
echo "=============================================================="

