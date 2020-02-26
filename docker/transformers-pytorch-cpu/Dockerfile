FROM ubuntu:18.04

ARG PYTHON_VERSION=3.7
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
     /opt/conda/bin/conda create -n huggingface python==$PYTHON_VERSION && \
     /opt/conda/bin/conda install -y -n huggingface -c pytorch pytorch cpuonly && \
     /opt/conda/bin/conda install -y -n huggingface jupyter && \
     /opt/conda/bin/conda clean -ya && \
     /opt/conda/envs/huggingface/bin/pip install transformers==$TRANSFORMERS_VERSION && \
     echo 'export PATH=/opt/conda/bin:$PATH' >> /etc/profile.d/conda.sh && \
     rm -rf /var/lib/apt/lists

ENV PATH /opt/conda/envs/hugginface/bin:/opt/conda/bin:$PATH


WORKDIR /workspace
RUN chmod -R a+w .

ENTRYPOINT ["/bin/bash"]