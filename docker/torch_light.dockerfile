FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
USER root
RUN apt-get update && apt-get install -y libsndfile1-dev espeak-ng time git
RUN apt-get install -y g++ cmake
ENV VIRTUAL_ENV=/usr/local
RUN pip --no-cache-dir install uv
RUN uv venv
RUN uv pip install --no-cache-dir -U pip setuptools
RUN uv pip install --no-cache-dir "pytest<8.0.1" "fsspec>=2023.5.0,<2023.10.0" pytest-subtests pytest-xdist

RUN uv pip install --no-cache-dir --upgrade 'torch' --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache-dir "transformers[sklearn,torch,testing,sentencepiece,vision,timm,torch-speech]"


RUN pip uninstall -y transformers
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get autoremove  --purge -y cmake g++
RUN pip cache remove "nvidia-*"
RUN pip cache remove triton