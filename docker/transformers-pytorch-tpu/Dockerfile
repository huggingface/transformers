FROM ubuntu:24.04
LABEL maintainer="Hugging Face"

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VER=3.13

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        git-lfs \
        ffmpeg \
        espeak-ng \
        tesseract-ocr \
        libsndfile1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# The CI runs `python3 -m pip` and `python3 -m venv`, so the venv is seeded with pip.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv --python ${PYTHON_VER} --python-preference only-managed --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# The TPU compilation cache is keyed on these builds: change them together with the cache.
ARG TORCH=2.15.0
ARG TORCH_TPU=0.3.0
ARG LIBTPU=0.0.53.1
RUN uv pip install --no-cache-dir \
        torch==${TORCH}+cpu torchvision==0.29.2+cpu torchaudio==2.11.1+cpu torchcodec==0.17.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --no-cache-dir torch-tpu==${TORCH_TPU} libtpu==${LIBTPU} && \
    uv pip freeze | grep -E '^(torch|torchvision|torchaudio|torchcodec|torch-tpu|libtpu)==' > /opt/torch-constraints.txt

# Install the test dependencies only: the CI installs its own checkout of transformers on top.
ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF && \
    uv pip install --no-cache-dir -c /opt/torch-constraints.txt ".[all,testing,sklearn]" pytesseract && \
    uv pip uninstall transformers && \
    cd .. && rm -rf transformers

# `kernels` may give different outputs (within 1e-5 range) even with the same model (weights) and the same inputs
RUN uv pip uninstall kernels
