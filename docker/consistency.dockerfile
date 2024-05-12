FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
USER root
RUN apt-get update && apt-get install -y time git pkg-config make git-lfs
ENV VIRTUAL_ENV=/usr/local
RUN pip install uv && uv venv && uv pip install --no-cache-dir -U pip setuptools GitPython
RUN uv pip install --no-cache-dir --upgrade 'torch' --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache-dir tensorflow-cpu tf-keras
RUN uv pip install --no-cache-dir "transformers[flax,quality,vision,testing]"
RUN git lfs install

RUN pip uninstall -y transformers
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get autoremove && apt-get autoclean

