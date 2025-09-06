FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

# Use login shell to read variables from `~/.profile` (to pass dynamic created variables between RUN commands)
SHELL ["sh", "-lc"]

# The following `ARG` are mainly used to specify the versions explicitly & directly in this docker file, and not meant
# to be used as arguments for docker build (so far).

ARG PYTORCH='2.8.0'
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu126'
# Disable kernel mapping for now until all tests pass
ENV DISABLE_KERNEL_MAPPING=1

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg git-lfs
RUN git lfs install
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

# 1. Put several commands in a single `RUN` to avoid image/layer exporting issue. Could be revised in the future.
# 2. Regarding `torch` part, We might need to specify proper versions for `torchvision` and `torchaudio`.
#    Currently, let's not bother to specify their versions explicitly (so installed with their latest release versions).
RUN python3 -m pip install --no-cache-dir -e ./transformers[dev,onnxruntime] && [ ${#PYTORCH} -gt 0 -a "$PYTORCH" != "pre" ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; echo "export VERSION='$VERSION'" >> ~/.profile && echo torch=$VERSION && [ "$PYTORCH" != "pre" ] && python3 -m pip install --no-cache-dir -U $VERSION torchvision torchaudio torchcodec --extra-index-url https://download.pytorch.org/whl/$CUDA || python3 -m pip install --no-cache-dir -U --pre torch torchvision torchaudio torchcodec --extra-index-url https://download.pytorch.org/whl/nightly/$CUDA && python3 -m pip uninstall -y tensorflow tensorflow_text tensorflow_probability

RUN python3 -m pip uninstall -y flax jax

RUN python3 -m pip install --no-cache-dir -U timm

RUN [ "$PYTORCH" != "pre" ] && python3 -m pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git || echo "Don't install detectron2 with nightly torch"

RUN python3 -m pip install --no-cache-dir pytesseract

RUN python3 -m pip install -U "itsdangerous<2.1.0"

RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/accelerate@main#egg=accelerate

RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/peft@main#egg=peft

# For bettertransformer
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/optimum@main#egg=optimum
# For kernels
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/kernels@main#egg=kernels

# For video model testing
RUN python3 -m pip install --no-cache-dir av

# Some slow tests require bnb
RUN python3 -m pip install --no-cache-dir bitsandbytes

# Some tests require quanto
RUN python3 -m pip install --no-cache-dir quanto

# After using A10 as CI runner, let's run FA2 tests
RUN [ "$PYTORCH" != "pre" ] && python3 -m pip uninstall -y ninja && python3 -m pip install --no-cache-dir ninja && python3 -m pip install flash-attn --no-cache-dir --no-build-isolation || echo "Don't install FA2 with nightly torch"

# TODO (ydshieh): check this again
# `quanto` will install `ninja` which leads to many `CUDA error: an illegal memory access ...` in some model tests
# (`deformable_detr`, `rwkv`, `mra`)
RUN python3 -m pip uninstall -y ninja

# For `nougat` tokenizer
RUN python3 -m pip install --no-cache-dir python-Levenshtein

# For `FastSpeech2ConformerTokenizer` tokenizer
RUN python3 -m pip install --no-cache-dir g2p-en

# For Some bitsandbytes tests
RUN python3 -m pip install --no-cache-dir einops

# For Some tests with `@require_liger_kernel`
RUN python3 -m pip install --no-cache-dir liger-kernel

# `kernels` may give different outputs (within 1e-5 range) even with the same model (weights) and the same inputs
RUN python3 -m pip uninstall -y kernels

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop
