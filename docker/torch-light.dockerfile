FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
USER root
RUN apt-get update && apt-get install -y libsndfile1-dev espeak-ng time git
RUN apt-get install -y g++ cmake
ENV VIRTUAL_ENV=/usr/local
RUN pip --no-cache-dir install uv
RUN uv venv
RUN uv pip install --no-cache-dir -U pip setuptools accelerate
RUN uv pip install --no-cache-dir "fsspec>=2023.5.0,<2023.10.0"
RUN pip install --no-cache-dir 'torch' 'torchvision' 'torchaudio' --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir "transformers[sklearn,sentencepiece,vision,timm,testing]"

# sklearn,sentencepiece,vision,timm,testing fucking install nividia-ci


RUN pip uninstall -y transformers
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get autoremove  --purge -y cmake
RUN pip cache remove "nvidia-*"
RUN pip cache remove triton