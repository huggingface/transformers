FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
USER root
RUN apt-get update && apt-get install -y time git pkg-config make
RUN apt-get install -y cmake g++
ENV VIRTUAL_ENV=/usr/local
RUN pip install uv
RUN uv venv
RUN uv pip install --no-cache-dir -U pip setuptools
RUN git lfs install

RUN pip install --no-cache-dir 'torch' 'torchvision' 'torchaudio' --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache-dir pypi-kenlm protobuf==3.20.3 accelerate tensorflow_probability pytest pytest-xdist parameterized
RUN uv pip install --no-cache-dir "transformers[sklearn,tf-cpu,sentencepiece,vision, testing]"


RUN pip uninstall -y transformers
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip cache remove "nvidia-*"
RUN pip uninstall -y `pip freeze | grep "nvidia-*"` || true
RUN pip uninstall -y `pip freeze | grep "triton-*"` || true



RUN pip cache remove triton
RUN apt-get --purge remove "*nvidia*" || true
RUN apt-get autoremove
RUN apt-get autoclean