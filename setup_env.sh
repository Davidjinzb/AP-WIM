#!/bin/bash

# WAM Environment Setup Script
# This script creates a conda environment and installs all dependencies

set -e  # Exit on any error

echo "🚀 Setting up WAM environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "wam"; then
    echo "⚠️  Environment 'wam' already exists."
    read -p "Do you want to remove it and create a fresh one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n wam
    else
        echo "✅ Using existing environment. Activate it with: conda activate wam"
        exit 0
    fi
fi

# Create new environment
echo "📦 Creating conda environment 'wam' with Python 3.9..."
conda create -n wam python=3.9 -y

# Activate environment
echo "🔧 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate wam

# Install PyTorch
echo "🔥 Installing PyTorch..."
# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 CUDA detected, installing PyTorch with CUDA support..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    echo "💻 Installing PyTorch for CPU..."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

# Install remaining dependencies
echo "📚 Installing other dependencies..."
pip install -r requirements.txt

# Verify installation
echo "✅ Verifying installation..."
python -c "import torch; import ptwt; import cv2; print('🎉 All dependencies installed successfully!')"

echo ""
echo "🎊 Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate wam"
echo ""
echo "To start using WAM:"
echo "  jupyter notebook example.ipynb"
echo ""
echo "Happy coding! 🚀" 