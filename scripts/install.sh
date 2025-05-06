#!/usr/bin/env bash
# ----------------------------------------------
# Verified install script for Hugging Face Transformers
# Supports: pip install from PyPI or source, with optional extras
# Usage:
#   ./scripts/install.sh "[torch]"  # Install from source
# ----------------------------------------------

set -e

EXTRA=${1:-""}      # e.g., "[all]", "[torch]", or leave blank

echo "=== 1. Checking Python version"
if ! python - <<EOF
import sys
if not (3,9) <= sys.version_info < (3,13):
    sys.exit(1)
EOF
then
  echo "ERROR: Transformers requires Python >=3.9 and <3.13. Current: $(python --version)"
  exit 1
fi
echo "OK: $(python3 --version) is supported."

# -----------------------------------------------------------------------------
echo "=== 2. Ensuring virtual environment"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
  read -p "No virtual env detected. Create one at .venv/? [Y/n] " yn
  yn=${yn:-Y}
  if [[ $yn =~ ^[Yy]$ ]]; then
    python3 -m venv .venv
    echo "Created .venv/, activating..."
    # shellcheck disable=SC1091
    source .venv/bin/activate
  else
    echo "Continuing without virtual env—make sure you really want that."
  fi
else
  echo "Virtual env detected: ${VIRTUAL_ENV:-$CONDA_DEFAULT_ENV}"
fi

# -----------------------------------------------------------------------------
echo "=== 3. Upgrading pip, setuptools, wheel"
pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
echo "=== 4. Installing Transformers from local source"
pip install .${EXTRA}

# -----------------------------------------------------------------------------
echo "=== 5. Verifying installation"
CMD='from transformers import pipeline
pipe = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
print(pipe("the secret to baking a really good cake is "))'
echo "Running Python commands:"
echo "${CMD}"
python -c "${CMD}"

# -----------------------------------------------------------------------------
echo "=== Done: Transformers is installed and working."