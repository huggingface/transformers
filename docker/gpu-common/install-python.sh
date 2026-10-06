#!/bin/sh
# Installs Python $1 (the repo's `.python-version`, passed as the `PYTHON_VERSION` build arg) with uv, in a venv at
# `$VIRTUAL_ENV`. The Dockerfile sets `VIRTUAL_ENV`, `UV_PYTHON_INSTALL_DIR` and puts the venv first on `PATH`, so
# `python`, `python3` and `pip` all resolve to it.
set -e
: "${1:?usage: install-python.sh PYTHON_VERSION (is the PYTHON_VERSION build arg set?)}"
: "${VIRTUAL_ENV:?VIRTUAL_ENV must be set by the Dockerfile}"
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python "$1" --seed "$VIRTUAL_ENV"
python3 --version
