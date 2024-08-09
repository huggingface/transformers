FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
USER root
ARG REF=main
RUN apt-get update && apt-get install -y time git pkg-config make git-lfs
ENV UV_PYTHON=/usr/local/bin/python
RUN pip install uv && uv venv && uv pip install --no-cache-dir -U pip setuptools GitPython
RUN uv pip install --no-cache-dir --upgrade 'torch' --index-url https://download.pytorch.org/whl/cpu
# tensorflow pin matching setup.py
RUN uv pip install --no-cache-dir "tensorflow-cpu<2.16" "tf-keras<2.16"
RUN uv pip install --no-cache-dir "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[flax,quality,torch-speech,vision,testing]"
RUN git lfs install

RUN pip uninstall -y transformers
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get autoremove && apt-get autoclean
