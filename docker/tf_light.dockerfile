FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
USER root
RUN apt-get update && apt-get install -y libsndfile1-dev espeak-ng time git
RUN apt-get install -y cmake g++
ENV VIRTUAL_ENV=/usr/local
RUN pip --no-cache-dir install uv
RUN uv venv
RUN uv pip install --no-cache-dir -U pip setuptools
RUN uv pip install --no-cache-dir "pytest<8.0.1" "fsspec>=2023.5.0,<2023.10.0" pytest-subtests pytest-xdist
RUN uv pip install --no-cache-dir tensorflow_probability
RUN uv pip install  --upgrade --no-cache-dir "transformers[sklearn,tf-cpu,testing,sentencepiece,tf-speech,vision]"
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt remove -y cmake g++