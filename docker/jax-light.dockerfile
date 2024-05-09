FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
USER root
RUN apt-get update && apt-get install -y libsndfile1-dev espeak-ng time git g++ cmake
ENV VIRTUAL_ENV=/usr/local
RUN pip --no-cache-dir install uv &&  uv venv && uv pip install --no-cache-dir -U pip setuptools
RUN pip install --no-cache-dir "scipy<1.13" "transformers[flax,testing,sentencepiece,flax-speech,vision]"
RUN pip uninstall -y transformers
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get autoremove && apt-get autoclean