FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
USER root
ARG REF=main
RUN apt-get update && apt-get install -y time git g++ pkg-config make git-lfs
ENV UV_PYTHON=/usr/local/bin/python
RUN pip install uv && uv pip install --no-cache-dir -U pip setuptools GitPython
RUN uv pip install --no-cache-dir --upgrade 'torch' 'torchaudio' 'torchvision' --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache-dir pypi-kenlm
RUN uv pip install --no-cache-dir "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[quality,testing,torch-speech,vision]"
RUN git lfs install

RUN uv pip uninstall transformers
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get autoremove && apt-get autoclean
