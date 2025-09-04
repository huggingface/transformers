FROM python:3.9-slim
ENV PYTHONDONTWRITEBYTECODE=1
ARG REF=main
USER root
RUN apt-get update &&  apt-get install -y --no-install-recommends libsndfile1-dev espeak-ng time git g++ cmake pkg-config openssh-client git-lfs ffmpeg
ENV UV_PYTHON=/usr/local/bin/python
RUN pip --no-cache-dir install uv && uv pip install --no-cache-dir -U pip setuptools
RUN uv pip install --no-cache-dir 'torch' 'torchaudio' 'torchvision' 'torchcodec' --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-deps timm accelerate --extra-index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache-dir librosa "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[sklearn,sentencepiece,vision,testing]" seqeval albumentations jiwer
RUN uv pip uninstall transformers
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
