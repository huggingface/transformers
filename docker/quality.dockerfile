FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
USER root
RUN apt-get update && apt-get install -y libsndfile1-dev espeak-ng time git 
ENV VIRTUAL_ENV=/usr/local
RUN pip install uv
RUN uv venv
RUN uv pip install --no-cache-dir -U pip setuptools GitPython transformers "ruff==0.1.5" urllib3
RUN apt-get install -y jq curl
RUN apt-get clean && rm -rf /var/lib/apt/lists/*