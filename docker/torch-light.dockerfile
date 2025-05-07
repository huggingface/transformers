FROM python:3.9-slim
ENV PYTHONDONTWRITEBYTECODE=1
ARG REF=main
USER root
RUN apt-get update &&  apt-get install -y --no-install-recommends libsndfile1-dev espeak-ng time git g++ cmake pkg-config openssh-client git git-lfs
ENV UV_PYTHON=/usr/local/bin/python
RUN pip --no-cache-dir install uv && uv venv && uv pip install --no-cache-dir -U pip setuptools
RUN uv pip install --no-cache-dir --upgrade 'torch==2.6.0' 'torchaudio==2.6.0' 'torchvision==0.21.0' --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-deps timm accelerate --extra-index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache-dir librosa "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[sklearn,sentencepiece,vision,testing,tiktoken,num2words,video]"
RUN uv pip uninstall transformers
