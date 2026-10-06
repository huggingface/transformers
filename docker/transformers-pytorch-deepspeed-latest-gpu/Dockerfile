# Plain CUDA base, like `transformers-all-latest-gpu`, but deliberately held on CUDA 12.x rather
# than following that image to 13.0.
#
# DeepSpeed pre-compiles its CUDA ops here, and `op_builder/builder.py` picks the architectures to
# cross-compile for in `get_default_compute_capabilities()`, which only has branches for CUDA 11 and
# CUDA 12. On CUDA 13 neither matches, so the list stays at the `DEFAULT_COMPUTE_CAPABILITIES` of
# `6.0;6.1;7.0` -- every one of which CUDA 13 dropped -- and the build dies on
# `nvcc fatal: Unsupported gpu architecture 'compute_60'`. This is still the case on DeepSpeed
# master and in the latest release (0.19.6).
#
# So: bump this to a CUDA 13 base and `cu130` torch once DeepSpeed knows about CUDA 13, and keep the
# toolkit here matching `$CUDA` in the meantime -- `assert_no_cuda_mismatch()` compares them exactly
# and refuses to compile when they differ.
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

# Torch version is kept in sync with `transformers-all-latest-gpu`; `CUDA` is not, see above.
ARG PYTORCH='2.13.0'
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu126'

RUN apt update
# On top of what the general test image installs: `libaio-dev` for DeepSpeed's async I/O.
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng ffmpeg git-lfs libaio-dev curl

# Python comes from uv, at the version in the repo's `.python-version` (passed as the `PYTHON_VERSION` build arg by the
# build workflow), instead of Ubuntu 22.04's 3.10.
# The venv also provides `python`: DeepSpeed's `zero_to_fp32.py` is executed directly, with `#!/usr/bin/env python`.
ARG PYTHON_VERSION
ENV VIRTUAL_ENV="/opt/venv" UV_PYTHON_INSTALL_DIR="/opt/uv/python"
ENV PATH="$VIRTUAL_ENV/bin:/root/.local/bin:$PATH"
COPY --from=gpu-common install-python.sh /opt/ci/
RUN sh /opt/ci/install-python.sh "$PYTHON_VERSION"

RUN git lfs install
RUN python3 -m pip install --no-cache-dir --upgrade pip

# The DeepSpeed CI job passes `working-directory-prefix: /workspace` and so expects the clone at
# `/workspace/transformers`. The previous `nvcr.io` base image set this as its `WORKDIR`, the plain
# CUDA one does not.
WORKDIR /workspace

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

# `sklearn` on top of `deepspeed-testing`: the model zoo tests drive the example scripts, which
# compute metrics with scikit-learn (and scipy, which it pulls in). `transformers-all-latest-gpu`
# gets these through `[dev]`, and the previous `nvcr.io` base image happened to preinstall them.
RUN python3 -m pip install --no-cache-dir './transformers[deepspeed-testing,sklearn]'

# Install latest release PyTorch
# (PyTorch must be installed before pre-compiling any DeepSpeed c++/cuda ops.)
# (https://www.deepspeed.ai/tutorials/advanced-install/#pre-install-deepspeed-ops)
RUN python3 -m pip uninstall -y torch torchvision torchaudio torchcodec && python3 -m pip install --no-cache-dir -U torch==$PYTORCH torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/accelerate@main#egg=accelerate

# Narrow DeepSpeed's cross-compile list to the architectures the CI runners actually use, instead of
# the `6.0;6.1;7.0;8.0;8.6;9.0` default -- the old entries are dead weight in the op build, which is
# repeated inside every GPU VM. `TORCH_CUDA_ARCH_LIST` takes priority over the default list.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Pre-build **latest** DeepSpeed, so it would be ready for testing (otherwise, the 1st deepspeed test will timeout)
RUN python3 -m pip uninstall -y deepspeed
# This has to be run (again) inside the GPU VMs running the tests.
# The installation works here, but some tests fail, if we don't pre-build deepspeed again in the VMs running the tests.
# TODO: Find out why test fail.
RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 python3 -m pip install deepspeed --no-build-isolation --config-settings="--build-option=build_ext" --config-settings="--build-option=-j8" --no-cache -v --disable-pip-version-check 2>&1

COPY --from=gpu-common finish-transformers-install.sh /opt/ci/
RUN sh /opt/ci/finish-transformers-install.sh

COPY --from=gpu-common check-torch.sh /opt/ci/
RUN sh /opt/ci/check-torch.sh "$PYTORCH"

RUN python3 -c "from deepspeed.launcher.runner import main"
