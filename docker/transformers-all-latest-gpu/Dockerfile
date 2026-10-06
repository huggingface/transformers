FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

# Use login shell to read variables from `~/.profile` (to pass dynamic created variables between RUN commands)
SHELL ["sh", "-lc"]

# The following `ARG` are mainly used to specify the versions explicitly & directly in this docker file, and not meant
# to be used as arguments for docker build (so far).

ARG PYTORCH='2.14.0'
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu130'

# This needs to be compatible with the above `PYTORCH`.
ARG TORCHCODEC='0.16.0'

ARG FLASH_ATTN='false'

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng ffmpeg git-lfs curl

# Python comes from uv, at the version in the repo's `.python-version` (passed as the `PYTHON_VERSION` build arg by the
# build workflow), instead of Ubuntu 22.04's 3.10.
ARG PYTHON_VERSION
ENV VIRTUAL_ENV="/opt/venv" UV_PYTHON_INSTALL_DIR="/opt/uv/python"
ENV PATH="$VIRTUAL_ENV/bin:/root/.local/bin:$PATH"
COPY --from=gpu-common install-python.sh /opt/ci/
RUN sh /opt/ci/install-python.sh "$PYTHON_VERSION"

RUN git lfs install
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

RUN python3 -m pip install --no-cache-dir -e ./transformers[dev]

# 1. Put several commands in a single `RUN` to avoid image/layer exporting issue. Could be revised in the future.
# 2. For `torchcodec`, use `cpu` as we don't have `libnvcuvid.so` on the host runner. See https://github.com/meta-pytorch/torchcodec/issues/912
#    **Important**: We need to specify `torchcodec` version if the torch version is not the latest stable one.
# 3. `set -e` means "exit immediately if any command fails".
# 4. Warning: torchaudio only publishes wheels for the CUDA version it was released with (e.g. 2.11.0
#    only has a cu130 build). If torch is installed with a different CUDA suffix (e.g. cu132), pip
#    will keep the torchaudio installed in the previous step and torchaudio will raise a RuntimeError on import due to the
#    CUDA version mismatch. When upgrading CUDA, make sure torchaudio has a wheel for the new version.
RUN set -e; \
    # Determine torch version
    if [ ${#PYTORCH} -gt 0 ] && [ "$PYTORCH" != "pre" ]; then \
        VERSION="torch==${PYTORCH}.*"; \
        TORCHCODEC_VERSION="torchcodec==${TORCHCODEC}.*"; \
        # torchaudio is in maintenance mode: 2.11 is the final release, compatible with torch 2.11+
        # (no need to upgrade it with torch; it does not pin torch to a specific version)
        if printf '%s\n' "2.11.0" "${PYTORCH}" | sort -V | head -1 | grep -qx "2.11.0"; then \
            TORCHAUDIO_VERSION="torchaudio==2.11.0"; \
        else \
            TORCHAUDIO_VERSION="torchaudio==${PYTORCH}.*"; \
        fi; \
    else \
        VERSION="torch"; \
        TORCHAUDIO_VERSION="torchaudio"; \
        TORCHCODEC_VERSION="torchcodec"; \
    fi; \
    \
    # Log the version being installed
    echo "Installing torch version: $VERSION"; \
    \
    # Install PyTorch packages
    if [ "$PYTORCH" != "pre" ]; then \
        python3 -m pip install --no-cache-dir -U \
            $VERSION \
            torchvision \
            $TORCHAUDIO_VERSION \
            --extra-index-url https://download.pytorch.org/whl/$CUDA; \
        # We need to specify the version if the torch version is not the latest stable one.
        python3 -m pip install --no-cache-dir -U \
            $TORCHCODEC_VERSION --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
        python3 -m pip install --no-cache-dir -U --pre \
            torch \
            torchvision \
            torchaudio \
            --extra-index-url https://download.pytorch.org/whl/nightly/$CUDA; \
        python3 -m pip install --no-cache-dir -U --pre \
            torchcodec --extra-index-url https://download.pytorch.org/whl/nightly/cpu; \
    fi

RUN python3 -m pip install --no-cache-dir -U timm

RUN [ "$PYTORCH" != "pre" ] && python3 -m pip install --no-cache-dir --no-build-isolation git+https://github.com/facebookresearch/detectron2.git || echo "Don't install detectron2 with nightly torch"

RUN python3 -m pip install --no-cache-dir pytesseract

RUN python3 -m pip install -U "itsdangerous<2.1.0"

RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/accelerate@main#egg=accelerate

RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/peft@main#egg=peft

# For bettertransformer
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/optimum@main#egg=optimum

# For ONNX export tests (onnxscript pulls in onnx + onnx_ir; onnxruntime-gpu validates the exported
# graph on GPU for faster export testing)
RUN python3 -m pip install --no-cache-dir onnxscript onnxruntime-gpu

# For video model testing
RUN python3 -m pip install --no-cache-dir av

# Some slow tests require bnb
RUN python3 -m pip install --no-cache-dir bitsandbytes

# Some tests require quanto
RUN python3 -m pip install --no-cache-dir quanto

# After using A10 as CI runner, let's run FA2 tests
# Notes on the install command below (see https://github.com/huggingface/transformers/pull/47251):
# - wheel is upgraded before flash-attn: torch 2.13.0 added an unconditional setuptools>=77.0.3
#   requirement, which pip resolves to setuptools 83.0.0 (which removed pkg_resources). The Ubuntu
#   22.04 system wheel still imports pkg_resources, so --no-build-isolation builds fail. wheel>=0.43
#   (Feb 2024) uses importlib.metadata instead and is not affected.
# - if/else instead of `... || echo`: the old pattern silently swallowed real install failures;
#   now only FLASH_ATTN=false skips gracefully, and any actual install error will correctly
#   fail the Docker build.
RUN if [ "$FLASH_ATTN" != "false" ]; then \
        python3 -m pip uninstall -y ninja && \
        python3 -m pip install --no-cache-dir ninja && \
        python3 -m pip install --no-cache-dir -U wheel && \
        python3 -m pip install flash-attn --no-cache-dir --no-build-isolation; \
    else \
        echo "Skipping FA2 install (FLASH_ATTN=${FLASH_ATTN})"; \
    fi

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

COPY --from=gpu-common finish-transformers-install.sh /opt/ci/
RUN sh /opt/ci/finish-transformers-install.sh

# Smoke test of the pinned torch stack, see the script.
COPY --from=gpu-common check-torch.sh /opt/ci/
RUN sh /opt/ci/check-torch.sh "$PYTORCH"
