# check=skip=InvalidDefaultArgInFrom
# `PYTHON_VERSION` comes from the repo's `.python-version` (passed as a build arg by the build workflow): the base
# image ships torch in its own Python, so it has to be picked for that version.
ARG PYTHON_VERSION
FROM rocm/pytorch:rocm7.2_ubuntu22.04_py${PYTHON_VERSION}_pytorch_release_2.10.0
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y --no-install-recommends git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-dev python3-pip python3-dev ffmpeg git-lfs libjpeg-turbo8-dev libpng-dev zlib1g-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install

RUN python3 -m pip install --no-cache-dir --upgrade pip numpy importlib-metadata setuptools wheel ninja pytesseract "itsdangerous<2.1.0"

# Rebuild torchvision so decode_image has libjpeg and ROCm image ops stay on GPU.
RUN python3 -m pip install --no-cache-dir "setuptools<81" pybind11
RUN TV_VERSION=$(python3 -c "import torchvision; print(torchvision.__version__.split('+')[0])") && \
    python3 -m pip uninstall -y torchvision && \
    git clone --depth 1 --branch "v${TV_VERSION}" https://github.com/pytorch/vision.git /tmp/vision && \
    cd /tmp/vision && \
    sed -i -E 's|list\(CSRS_DIR\.glob\("([^"]+\.cpp)"\)\)|[p for p in CSRS_DIR.glob("\1") if not p.name.endswith("_hip.cpp")]|g' setup.py && \
    FORCE_CUDA=1 TORCHVISION_USE_FFMPEG=0 TORCHVISION_USE_VIDEO_CODEC=0 \
    python3 -m pip install --no-cache-dir --no-build-isolation -v . && \
    cd / && rm -rf /tmp/vision

RUN python3 -m pip install --no-cache-dir --no-build-isolation git+https://github.com/facebookresearch/detectron2.git

# The build workflow passes the commit being built, so this clone (and every layer after it) is redone for each new
# commit instead of being reused from the cache.
ARG REF=main
WORKDIR /

RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

# Install transformers
RUN python3 -m pip install --no-cache-dir -e ./transformers[dev,testing,video,audio]

COPY --from=gpu-common finish-transformers-install.sh /opt/ci/
RUN sh /opt/ci/finish-transformers-install.sh

# Remove nvml and nvidia-ml-py as it is not compatible with ROCm. apex is not tested on NVIDIA either.
RUN python3 -m pip uninstall py3nvml pynvml nvidia-ml-py apex -y


# On ROCm, torchcodec is required to decode audio files. 0.10 matches the
# torch 2.10 C++ ABI in the base image above;
RUN python3 -m pip install --no-cache-dir "torchcodec==0.10"

# Install a pre-built flash-attention (CK backend) wheel: AMD's mirror has no Python 3.11 build, and building it from
# source takes hours. This one comes from https://github.com/mjun0812/flash-attention-prebuild-wheels, built for the
# torch 2.10 of the base image and for gfx90a/gfx942/gfx950 (MI250/MI300/MI355). It is pinned by checksum since it is a
# third-party build, and imported right away so an ABI mismatch with the base image's torch fails the image build.
ARG FLASH_ATTN_WHEEL='flash_attn-2.8.4+rocm7.0torch2.10git4a948e9-cp311-cp311-linux_x86_64.whl'
ARG FLASH_ATTN_SHA256='0add5e1b76e8750f7016589a10f1391e4455cfa524e69b2122ebf7f512d21058'
RUN set -e; \
    curl -fsSL -o "/tmp/${FLASH_ATTN_WHEEL}" \
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v1.4.0/$(echo "${FLASH_ATTN_WHEEL}" | sed 's/+/%2B/')"; \
    echo "${FLASH_ATTN_SHA256}  /tmp/${FLASH_ATTN_WHEEL}" | sha256sum -c -; \
    python3 -m pip install --no-cache-dir "/tmp/${FLASH_ATTN_WHEEL}"; \
    rm "/tmp/${FLASH_ATTN_WHEEL}"; \
    python3 -c "import torch, flash_attn, flash_attn_2_cuda; print('flash_attn', flash_attn.__version__)"

RUN python3 -m pip install --no-cache-dir einops blobfile num2words

# timm is required for many vision models tests
RUN python3 -m pip install --no-cache-dir timm
