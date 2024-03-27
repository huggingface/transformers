FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
USER root
RUN apt-get update && apt-get install -y time git pkg-config make
ENV VIRTUAL_ENV=/usr/local
RUN pip install uv
RUN uv venv
RUN uv pip install --no-cache-dir -U pip setuptools
RUN uv pip install --no-cache-dir tensorflow-cpu
RUN uv pip install --no-cache --upgrade 'torch' --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade GitPython "transformers[flax,quality]"
RUN pip uninstall -y transformers
RUN apt-get clean && rm -rf /var/lib/apt/lists/*