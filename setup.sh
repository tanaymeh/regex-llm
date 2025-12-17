#!/bin/bash
set -e  # Exit immediately if any command fails

echo "🚀 Starting Environment Setup..."

uv sync

# 1. Install System Dependencies (Crucial for C++ compilation)
# - libpcre2-dev: Required for your regex logic
# - python3-dev: Required for PyBind11 headers
echo "📦 Installing system libraries..."
apt-get update
apt-get install -y libpcre2-dev python3-dev build-essential

# 2. Install Python Build Tools
# (Ensure pybind11 is there before we try to compile)
echo "🐍 Installing Python build dependencies..."
uv pip install pybind11 setuptools wheel

# 3. Install Your Custom C++ Extension
# Points to the folder containing setup.py (adjust path if needed)
echo "⚙️  Compiling and installing fast_regex..."
uv pip install ./src/fast_regex_csrc --no-build-isolation

# 5. Launch Training
# echo "🔥 Starting GRPO Training..."
# export VLLM_GPU_MEMORY_UTILIZATION=0.4 # Uncomment if using vLLM
export WANDB_PROJECT="regex-r1"
# uv run src/train.py