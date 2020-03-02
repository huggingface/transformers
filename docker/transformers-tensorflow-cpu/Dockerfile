FROM ubuntu:18.04

ARG TRANSFORMERS_VERSION=2.5.1
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
    tensorflow-cpu \
    transformers==$TRANSFORMERS_VERSION


WORKDIR /workspace
RUN chmod -R a+w .

ENTRYPOINT ["/bin/bash"]