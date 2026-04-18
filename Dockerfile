FROM python:3.10-slim

LABEL maintainer="Hugging Face Team"
LABEL description="Docker image for Hugging Face Transformers - Initial Runtime Environment"
LABEL repository="https://github.com/huggingface/transformers"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV UV_PYTHON=/usr/local/bin/python
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_OFFLINE=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    make \
    build-essential \
    curl \
    ca-certificates \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip setuptools wheel uv

COPY setup.py README.md ./

RUN uv pip install --no-cache-dir --system \
    "torch>=2.4" \
    "huggingface-hub>=1.5.0,<2.0" \
    "numpy>=1.17" \
    "packaging>=20.0" \
    "pyyaml>=5.1" \
    "regex>=2025.10.22" \
    "tokenizers>=0.22.0,<=0.23.0" \
    "safetensors>=0.4.3" \
    "tqdm>=4.27" \
    "typer" \
    "accelerate>=1.1.0"

COPY pyproject.toml ./
COPY src/ ./src/

RUN uv pip install --no-cache-dir --system .

RUN mkdir -p /app/.cache/huggingface \
    && chmod -R 777 /app/.cache

RUN python -c "import transformers; print(f'Transformers v{transformers.__version__} installed successfully!')"

EXPOSE 8000

CMD ["python", "-c", "import transformers; print(f'Welcome to Transformers v{transformers.__version__}!')"]

