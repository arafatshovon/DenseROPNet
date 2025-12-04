#!/bin/bash
# =============================================================================
# Inference Script for ROP Detection Model
# =============================================================================
# This script performs inference on new images.
#
# Usage:
#   ./scripts/predict.sh path/to/model.keras path/to/image.jpg
#   ./scripts/predict.sh path/to/model.keras path/to/images/ --explain
#
# Arguments:
#   $1 - Path to trained model (required)
#   $2 - Path to image or image directory (required)
#   $3 - Additional flags (e.g., --explain)
# =============================================================================

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

# Check arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Please provide model path and image path"
    echo "Usage: ./scripts/predict.sh path/to/model.keras path/to/image.jpg"
    exit 1
fi

MODEL_PATH="$1"
IMAGE_PATH="$2"
EXTRA_FLAGS="${3:-}"

echo "=============================================================="
echo "ROP Detection Inference"
echo "=============================================================="
echo "Model: ${MODEL_PATH}"
echo "Input: ${IMAGE_PATH}"
echo "=============================================================="

# Build command
CMD="python predict.py --model ${MODEL_PATH}"

if [ -d "${IMAGE_PATH}" ]; then
    CMD="${CMD} --image_dir ${IMAGE_PATH}"
else
    CMD="${CMD} --image ${IMAGE_PATH}"
fi

if [ -n "${EXTRA_FLAGS}" ]; then
    CMD="${CMD} ${EXTRA_FLAGS}"
fi

echo ""
echo "Command: ${CMD}"
echo ""

# Run inference
${CMD}

