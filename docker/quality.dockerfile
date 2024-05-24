FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
ARG REF=main
USER root
RUN apt-get update && apt-get install -y time git 
ENV VIRTUAL_ENV=/usr/local
RUN pip install uv==0.1.45 &&  uv venv
RUN uv pip install --no-cache-dir -U pip setuptools GitPython "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[ruff]" urllib3
RUN apt-get install -y jq curl && apt-get clean && rm -rf /var/lib/apt/lists/*