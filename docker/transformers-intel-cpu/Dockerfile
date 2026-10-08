FROM intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04 AS base
LABEL maintainer="Hugging Face"
SHELL ["/bin/bash", "-c"]

ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update

RUN apt-get update && \
    apt-get -y install \
    apt-utils \
    build-essential \
    ca-certificates \
    clinfo \
    curl \
    git \
    git-lfs \
    vim \
    numactl \
    gnupg2 \
    gpg-agent \
    python3-dev \
    python3-opencv \
    unzip \
    ffmpeg \
    tesseract-ocr \
    espeak-ng \
    wget \
    ncurses-term \
    google-perftools \
    libmimalloc-dev \
    libmimalloc2.0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Use virtual env because Ubuntu:24 does not allowed pip on original python
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip wheel
RUN pip install torch torchvision torchaudio torchcodec torchao --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
RUN pip install av pyctcdecode pytesseract decord galore-torch fire scipy scikit-learn sentencepiece sentence_transformers sacremoses nltk rouge_score librosa soundfile mpi4py pytorch_msssim
RUN pip install onnx optimum onnxruntime
RUN pip install -U datasets timm transformers accelerate peft diffusers opencv-python kenlm evaluate
RUN pip install -U intel-openmp
RUN pip install kernels
RUN pip install gptqmodel bitsandbytes
RUN pip install git+https://github.com/facebookresearch/detectron2.git --no-build-isolation

# MXFP4 requires triton >= 3.5.0
RUN pip install "triton>=3.5.0"

ENV LD_PRELOAD=${LD_PRELOAD}:/opt/venv/lib/libiomp5.so:/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
ENV KMP_AFFINITY=granularity=fine,compact,1,0

RUN touch /entrypoint.sh
RUN chmod +x /entrypoint.sh
RUN echo "#!/bin/bash" >> /entrypoint.sh
RUN echo "/bin/bash" >> /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
