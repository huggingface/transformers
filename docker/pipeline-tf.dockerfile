FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
ARG REF=main
USER root
RUN apt-get update && apt-get install -y libsndfile1-dev espeak-ng time git cmake g++
ENV VIRTUAL_ENV=/usr/local
RUN pip --no-cache-dir install uv==0.1.45 && uv venv && uv pip install --no-cache-dir -U pip setuptools
RUN pip install --no-cache-dir "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[sklearn,tf-cpu,testing,sentencepiece,tf-speech,vision]"
RUN uv pip install --no-cache-dir  "protobuf==3.20.3" tensorflow_probability
RUN apt-get clean && rm -rf /var/lib/apt/lists/*