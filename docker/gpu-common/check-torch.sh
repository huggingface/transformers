#!/bin/sh
# Smoke test: fail the image build immediately if a later `pip install` replaced or broke the pinned CUDA torch stack
# (e.g. a dep that drags in a different torch/CUDA/NCCL -- which surfaces as `undefined symbol: ncclCommResume` at
# `import torch`; see run 28991812987). We import torch (which eagerly loads `libtorch_cuda.so`) AND assert the version
# still matches the pin ($1, the `PYTORCH` build arg), since transformers only requires `torch>=2.4` -- a silent swap
# to a newer torch otherwise passes every version check and reaches the GPU test jobs. No GPU is present at build
# time, so we only exercise the import + CUDA runtime load, not device availability.
# Pass an empty pin or `pre` (nightly) to skip the version check.
set -e
python3 -c "import torch; print('torch version:', torch.__version__); torch.cuda.is_available()"
if [ -n "$1" ] && [ "$1" != "pre" ]; then
    python3 -c "import torch, sys; v = torch.__version__.split('+')[0]; sys.exit(0 if v.startswith('$1') else 'ERROR: torch is ' + torch.__version__ + ', expected $1.* - the pinned CUDA build was clobbered')"
fi
