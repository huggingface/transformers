FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
LABEL maintainer="Hugging Face"
LABEL repository="transformers"

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
    mkl \
    torch

COPY . /workspace/transformers
RUN cd /workspace/transformers && \
    python3 -m pip install .

WORKDIR /workspace
RUN chmod -R a+w .

ENTRYPOINT ["/bin/bash"]