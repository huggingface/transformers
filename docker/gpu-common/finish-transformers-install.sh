#!/bin/sh
# Last steps of installing the `transformers` clone at `/<workdir>/transformers`, shared by the GPU images.
set -e
# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
cd transformers && python3 setup.py develop
