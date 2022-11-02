ARG BASE_DOCKER_IMAGE="nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04"
FROM $BASE_DOCKER_IMAGE
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

# Use login shell to read variables from `~/.profile` (to pass dynamic created variables between RUN commands)
SHELL ["sh", "-lc"]

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg git-lfs
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

# Remove all frameworks
# (`accelerate` requires `torch`, and this causes import issues for TF-only testing)
RUN python3 -m pip uninstall -y torch torchvision torchaudio accelerate tensorflow jax flax

# Get the libraries and their versions to install, and write installation command to `~/.profile`.
RUN python3 ./transformers/utils/past_ci_versions.py --framework $FRAMEWORK --version $VERSION

# Install the target framework
RUN echo "INSTALL_CMD = $INSTALL_CMD"
RUN $INSTALL_CMD

# Having installation problems for torch-scatter with torch <= 1.6. Disable so we have the same set of tests.
# (This part will be removed once the logic of using `past_ci_versions.py` is used in other Dockerfile files.)
# # Use installed torch version for `torch-scatter`.
# # (The env. variable $CUDA is defined in `past_ci_versions.py`)
# RUN [ "$FRAMEWORK" = "pytorch" ] && python3 -m pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-$(python3 -c "from torch import version; print(version.__version__.split('+')[0])")+$CUDA.html || echo "torch-scatter not to be installed"

RUN python3 -m pip install -U "itsdangerous<2.1.0"
