FROM ubuntu:18.04
LABEL maintainer="Hugging Face"
LABEL repository="transformers"

ARG TRANSFORMERS_VERSION=2.5.1
RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    mkl \
    torch

COPY . /workspace/transformers
RUN cd /workspace/transformers && \
    python3 -m pip install --no-cache-dir .

WORKDIR /workspace
RUN chmod -R a+w .

ENTRYPOINT ["/bin/bash"]