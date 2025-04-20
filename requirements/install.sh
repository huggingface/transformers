#!/usr/bin/env bash
set -e  # exit on error
set -u  # error on undefined vars
set -o pipefail

# === Config ===
PYTHON_VERSION_REQUIRED=">=3.9,<3.13"
VENV_DIR=".my-env"
REQUIREMENTS_URL="https://raw.githubusercontent.com/huggingface/transformers/main/examples/pytorch/_tests_requirements.txt"

# === Validate Python Version ===
echo "Checking Python version..."
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
if ! python3 -c 'import sys; assert (3, 9) <= sys.version_info[:2] < (3, 13), "Python >=3.9 and <3.13 required."' ; then
  echo "Unsupported Python version: $(python3 --version)"
  exit 1
else
  echo "✅ Python $PYTHON_VERSION is OK."
fi

# === Create Virtual Env ===
echo "Creating virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# === Upgrade pip ===
pip install --upgrade pip setuptools

# === Install Required Packages ===
echo "Installing required packages..."
pip install "jax>=0.4.1,<=0.4.13"
pip install "optax>=0.0.8,<=0.1.4"
pip install "orbax-checkpoint==0.2.3"
pip install "torch>=2.1"
pip install "tensorflow>2.9,<2.16"
pip install "flax>=0.4.1,<=0.7.0"
pip install "accelerate>=0.26.0"
pip install "transformers"
pip install -r <(curl -s "$REQUIREMENTS_URL")  # optional extras for tests/examples

# === Validate Install ===
echo "Validating installation..."
python -c 'from transformers import pipeline; pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B"); print(pipeline("the secret to baking a really good cake is "))'

echo "Installation complete!"