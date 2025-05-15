ARG BASE_DOCKER_IMAGE
FROM $BASE_DOCKER_IMAGE
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

# Use login shell to read variables from `~/.profile` (to pass dynamic created variables between RUN commands)
SHELL ["sh", "-lc"]

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg git-lfs libaio-dev
RUN git lfs install
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF
RUN python3 -m pip install --no-cache-dir -e ./transformers[dev,onnxruntime]

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

ARG FRAMEWORK
ARG VERSION

# Control `setuptools` version to avoid some issues
RUN [ "$VERSION" != "1.10" ] && python3 -m pip install -U setuptools || python3 -m pip install -U "setuptools<=59.5"

# Remove all frameworks
RUN python3 -m pip uninstall -y torch torchvision torchaudio tensorflow jax flax

# Get the libraries and their versions to install, and write installation command to `~/.profile`.
RUN python3 ./transformers/utils/past_ci_versions.py --framework $FRAMEWORK --version $VERSION

# Install the target framework
RUN echo "INSTALL_CMD = $INSTALL_CMD"
RUN $INSTALL_CMD

RUN [ "$FRAMEWORK" != "pytorch" ] && echo "`deepspeed-testing` installation is skipped" || python3 -m pip install --no-cache-dir ./transformers[deepspeed-testing]

# Remove `accelerate`: it requires `torch`, and this causes import issues for TF-only testing
# We will install `accelerate@main` in Past CI workflow file
RUN python3 -m pip uninstall -y accelerate

# Uninstall `torch-tensorrt` and `apex` shipped with the base image
RUN python3 -m pip uninstall -y torch-tensorrt apex

# Pre-build **nightly** release of DeepSpeed, so it would be ready for testing (otherwise, the 1st deepspeed test will timeout)
RUN python3 -m pip uninstall -y deepspeed
# This has to be run inside the GPU VMs running the tests. (So far, it fails here due to GPU checks during compilation.)
# Issue: https://github.com/deepspeedai/DeepSpeed/issues/2010
# RUN git clone https://github.com/deepspeedai/DeepSpeed && cd DeepSpeed && rm -rf build && \
#    DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_UTILS=1 python3 -m pip install . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1

RUN python3 -m pip install -U "itsdangerous<2.1.0"

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop
