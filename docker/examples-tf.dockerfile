FROM python:3.9-slim
ENV PYTHONDONTWRITEBYTECODE=1
ARG REF=main
USER root
RUN apt-get update && apt-get install -y libsndfile1-dev espeak-ng time git
RUN apt-get install -y g++ cmake
ENV UV_PYTHON=/usr/local/bin/python
RUN pip --no-cache-dir install uv && uv venv
RUN uv pip install --no-cache-dir -U pip setuptools albumentations seqeval
RUN uv pip install  --upgrade --no-cache-dir "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[tf-cpu,sklearn,testing,sentencepiece,tf-speech,vision]"
RUN uv pip install --no-cache-dir  "protobuf==3.20.3"
RUN uv pip uninstall transformers
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
