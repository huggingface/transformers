FROM python:3.9-slim
ENV PYTHONDONTWRITEBYTECODE=1
ARG REF=main
RUN echo ${REF}
USER root
RUN apt-get update &&  apt-get install -y --no-install-recommends libsndfile1-dev espeak-ng time git g++ cmake pkg-config openssh-client git git-lfs
ENV UV_PYTHON=/usr/local/bin/python
RUN pip --no-cache-dir install uv && uv venv && uv pip install --no-cache-dir -U pip setuptools
RUN uv pip install --no-cache-dir  --no-deps accelerate --extra-index-url https://download.pytorch.org/whl/cpu 
RUN pip install --no-cache-dir 'torch' 'torchvision' 'torchaudio' --index-url https://download.pytorch.org/whl/cpu
RUN git lfs install

RUN uv pip install --no-cache-dir pypi-kenlm
RUN pip install --no-cache-dir  "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[tf-cpu,sklearn,sentencepiece,vision,testing]"
RUN uv pip install --no-cache-dir  "protobuf==3.20.3" librosa


RUN pip uninstall -y transformers
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get autoremove && apt-get autoclean