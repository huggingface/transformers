FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
USER root
RUN apt-get update &&  apt-get install -y --no-install-recommends libsndfile1-dev espeak-ng time git g++ cmake && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/usr/local
RUN pip --no-cache-dir install uv && uv venv && uv pip install --no-cache-dir -U pip setuptools
RUN pip install --no-cache-dir 'torch' 'torchvision' 'torchaudio' --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache-dir accelerate soundfile "fsspec>=2023.5.0,<2023.10.0" "transformers[sklearn,sentencepiece,vision,timm,testing]"
RUN pip uninstall -y transformers && apt-get clean || apt-get -y --purge remove "*nvidia*" || apt-get autoremove || apt-get autoclean
RUN pip cache remove "nvidia-*" ||pip uninstall -y `pip freeze | grep "nvidia-*"` || pip uninstall -y `pip freeze | grep "triton-*"` || pip cache remove triton || true