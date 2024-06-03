FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
USER root
RUN apt-get update && apt-get install -y libsndfile1-dev espeak-ng time git
RUN apt-get install -y g++ cmake
ENV UV_PYTHON=/usr/local/bin/python
RUN pip --no-cache-dir install uv && uv venv
RUN uv pip install --no-cache-dir -U pip setuptools albumentations seqeval
RUN pip install  --upgrade --no-cache-dir "transformers[tf-cpu,sklearn,testing,sentencepiece,tf-speech,vision]"
RUN uv pip install --no-cache-dir  "protobuf==3.20.3" 
RUN pip uninstall -y transformers
RUN apt-get clean && rm -rf /var/lib/apt/lists/*