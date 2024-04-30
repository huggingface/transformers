FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
USER root
RUN apt-get update && apt-get install -y time git pkg-config make
ENV VIRTUAL_ENV=/usr/local
RUN pip install uv
RUN uv venv
RUN uv pip install --no-cache-dir -U pip setuptools GitPython
RUN uv pip install --no-cache --upgrade 'torch' --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache-dir tensorflow-cpu tf-keras
RUN uv pip install --no-cache-dir "transformers[flax,quality,vision,testing]"
RUN git lfs install

# Cleanup to reduce the size of the docker
RUN pip cache remove "nvidia-*" || true
RUN pip cache remove triton || true
RUN pip uninstall -y transformers
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
