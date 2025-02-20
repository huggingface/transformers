FROM python:3.9-slim
ENV PYTHONDONTWRITEBYTECODE=1
ARG REF=main
USER root
RUN apt-get update &&  apt-get install -y --no-install-recommends libsndfile1-dev espeak-ng time git g++ pkg-config openssh-client git
RUN apt-get install -y  cmake
ENV UV_PYTHON=/usr/local/bin/python
RUN pip --no-cache-dir install uv && uv venv && uv pip install --no-cache-dir -U pip setuptools
RUN pip install  --upgrade --no-cache-dir "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[tf-cpu,sklearn,testing,sentencepiece,tf-speech,vision]"
RUN uv pip install --no-cache-dir  "protobuf==3.20.3" 
RUN pip uninstall -y transformers
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get autoremove && apt-get autoclean