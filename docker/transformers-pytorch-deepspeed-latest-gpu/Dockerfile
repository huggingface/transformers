# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-04.html#rel_22-04
FROM nvcr.io/nvidia/pytorch:22.04-py3
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

ARG PYTORCH='1.13.0'
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu116'

RUN apt -y update
RUN apt install -y libaio-dev
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

# Install latest release PyTorch
# (PyTorch must be installed before pre-compiling any DeepSpeed c++/cuda ops.)
# (https://www.deepspeed.ai/tutorials/advanced-install/#pre-install-deepspeed-ops)
RUN python3 -m pip install --no-cache-dir -U torch==$PYTORCH torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN python3 -m pip install --no-cache-dir ./transformers[deepspeed-testing]

# The base image ships with `torch_tensorrt` which currently causes the DeepSpeed CI fail from test collection
RUN python3 -m pip uninstall -y torch_tensorrt

# Pre-build **latest** DeepSpeed, so it would be ready for testing (otherwise, the 1st deepspeed test will timeout)
RUN python3 -m pip uninstall -y deepspeed
# This has to be run (again) inside the GPU VMs running the tests.
# The installation works here, but some tests fail, if we don't pre-build deepspeed again in the VMs running the tests.
# TODO: Find out why test fail.
RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 python3 -m pip install deepspeed --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

# The base image ships with `pydantic==1.8.2` which is not working - i.e. the next command fails
RUN python3 -m pip install -U --no-cache-dir pydantic
RUN python3 -c "from deepspeed.launcher.runner import main"
