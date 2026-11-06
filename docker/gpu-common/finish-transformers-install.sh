#!/bin/sh
# Last steps of installing the `transformers` clone at `/<workdir>/transformers`, shared by the GPU images.
set -e
# `kernels` may give different outputs (within 1e-5 range) even with the same model (weights) and the same inputs
python3 -m pip uninstall -y kernels
# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
cd transformers && python3 setup.py develop
