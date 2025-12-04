#!/bin/bash
# =============================================================================
# Setup Script for ROP Detection Project
# =============================================================================
# This script sets up the development environment for the ROP detection project.
#
# Usage:
#   ./scripts/setup.sh
# =============================================================================

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

echo "=============================================================="
echo "Setting up ROP Detection Project"
echo "=============================================================="

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p results
mkdir -p experiments
mkdir -p data

echo ""
echo "=============================================================="
echo "Setup completed!"
echo "=============================================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start training, run:"
echo "  ./scripts/train.sh /path/to/your/data"
echo ""

