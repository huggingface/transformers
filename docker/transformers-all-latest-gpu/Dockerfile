FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

# Use login shell to read variables from `~/.profile` (to pass dynamic created variables between RUN commands)
SHELL ["sh", "-lc"]

# The following `ARG` are mainly used to specify the versions explicitly & directly in this docker file, and not meant
# to be used as arguments for docker build (so far).

ARG PYTORCH='2.12.0'
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu130'

# This needs to be compatible with the above `PYTORCH`.
ARG TORCHCODEC='0.11.0'

ARG FLASH_ATTN='false'

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg git-lfs
RUN git lfs install
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

RUN python3 -m pip install --no-cache-dir -e ./transformers[dev]

# 1. Put several commands in a single `RUN` to avoid image/layer exporting issue. Could be revised in the future.
# 2. For `torchcodec`, use `cpu` as we don't have `libnvcuvid.so` on the host runner. See https://github.com/meta-pytorch/torchcodec/issues/912
#    **Important**: We need to specify `torchcodec` version if the torch version is not the latest stable one.
# 3. `set -e` means "exit immediately if any command fails".
RUN python3 -m pip install --no-cache-dir torchcodec==0.12.0 --index-url https://download.pytorch.org/whl/test/cpu

RUN python3 -m pip install --no-cache-dir torch==2.12.0 torchvision==0.27.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/test/cu126

RUN python3 -m pip install --no-cache-dir -U timm

RUN [ "$PYTORCH" != "pre" ] && python3 -m pip install --no-cache-dir --no-build-isolation git+https://github.com/facebookresearch/detectron2.git || echo "Don't install detectron2 with nightly torch"

RUN python3 -m pip install --no-cache-dir pytesseract

RUN python3 -m pip install -U "itsdangerous<2.1.0"

RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/accelerate@main#egg=accelerate

RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/peft@main#egg=peft

# For bettertransformer
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/optimum@main#egg=optimum
# For kernels
RUN python3 -m pip install --no-cache-dir kernels

# For video model testing
RUN python3 -m pip install --no-cache-dir av

# Some slow tests require bnb
RUN python3 -m pip install --no-cache-dir bitsandbytes

# Some tests require quanto
RUN python3 -m pip install --no-cache-dir quanto

# After using A10 as CI runner, let's run FA2 tests
RUN [ "$FLASH_ATTN" != "false" ] && python3 -m pip uninstall -y ninja && python3 -m pip install --no-cache-dir ninja && python3 -m pip install flash-attn --no-cache-dir --no-build-isolation || echo "Don't install FA2 with nightly torch"

# TODO (ydshieh): check this again
# `quanto` will install `ninja` which leads to many `CUDA error: an illegal memory access ...` in some model tests
# (`deformable_detr`, `rwkv`, `mra`)
RUN python3 -m pip uninstall -y ninja

# For `nougat` tokenizer
RUN python3 -m pip install --no-cache-dir python-Levenshtein

# For `FastSpeech2ConformerTokenizer` tokenizer
RUN python3 -m pip install --no-cache-dir g2p-en

# For serving tests (audio pipelines)
RUN python3 -m pip install --no-cache-dir librosa python-multipart

# For Some bitsandbytes tests
RUN python3 -m pip install --no-cache-dir einops

# For `VibeVoice` (added in PR #40546)
RUN python3 -m pip install --no-cache-dir diffusers

# `kernels` may give different outputs (within 1e-5 range) even with the same model (weights) and the same inputs
RUN python3 -m pip uninstall -y kernels

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop
