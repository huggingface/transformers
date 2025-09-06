FROM intel/deep-learning-essentials:2025.1.3-0-devel-ubuntu24.04 AS base
SHELL ["/bin/bash", "-c"]

ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update

RUN apt-get update && apt-get -y install apt-utils build-essential ca-certificates \
    clinfo curl git git-lfs vim numactl python3-dev wget google-perftools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Use virtual env because Ubuntu:24 does not allowed pip on original python
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install torch torchvision torchaudio torchcodec --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
RUN pip install -U datasets transformers accelerate intel-openmp

ENV LD_PRELOAD=${LD_PRELOAD}:/opt/venv/lib/libiomp5.so:/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
ENV KMP_AFFINITY=granularity=fine,compact,1,0

RUN touch /entrypoint.sh && chmod +x /entrypoint.sh
RUN echo "#!/bin/bash" >> /entrypoint.sh && echo "/bin/bash" >> /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
