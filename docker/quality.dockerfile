FROM python:3.9-slim
ENV PYTHONDONTWRITEBYTECODE=1
ARG REF=main
USER root
RUN apt-get update && apt-get install -y time git 
ENV UV_PYTHON=/usr/local/bin/python
RUN pip install uv &&  uv venv
RUN uv pip install --no-cache-dir -U pip setuptools GitPython "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[ruff]" urllib3
RUN apt-get install -y jq curl && apt-get clean && rm -rf /var/lib/apt/lists/*