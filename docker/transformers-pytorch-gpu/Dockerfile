FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
FROM ubuntu:18.04

ARG PYTHON_VERSION=3.8
ARG TRANSFORMERS_VERSION=2.5.0
RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates

RUN curl -o Miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda.sh && \
    ./Miniconda.sh -b -p /opt/conda && \
    rm Miniconda.sh

RUN /opt/conda/bin/conda update -n base -c defaults conda && \
     /opt/conda/bin/conda create -n pytorch python==$PYTHON_VERSION && \
     /opt/conda/bin/conda install -y -n pytorch -c pytorch pytorch && \
     /opt/conda/bin/conda install -y -n pytorch jupyter && \
     /opt/conda/bin/conda clean -ya && \
     /opt/conda/envs/pytorch/bin/pip install transformers==$TRANSFORMERS_VERSION && \
     echo 'export PATH=/opt/conda/bin:$PATH' >> /etc/profile.d/conda.sh
ENV PATH /opt/conda/envs/pytorch/bin:/opt/conda/bin:$PATH


WORKDIR /workspace
RUN chmod -R a+w .

ENTRYPOINT ["jupyter", "notebook", "--allow-root", "--ip", "0.0.0.0"]