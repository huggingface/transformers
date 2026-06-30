FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
USER root
ARG REF=main
RUN apt-get update && apt-get install -y time git g++ pkg-config make git-lfs
ENV UV_PYTHON=/usr/local/bin/python
RUN pip install uv && uv pip install --no-cache-dir -U pip setuptools GitPython
RUN uv pip install --no-cache-dir --upgrade 'torch<=2.11.0' 'torchaudio' 'torchvision' --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache-dir pypi-kenlm
RUN uv pip install --no-cache-dir "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[quality,testing,torch-speech,vision]"
RUN git lfs install

# Use a custom patched pytest to force exit the process at the end, to avoid `Too long with no output (exceeded 10m0s): context deadline exceeded` (#40201)
RUN uv pip install --no-cache-dir git+https://github.com/ydshieh/pytest.git@8.4.1-ydshieh
RUN uv pip install --no-cache-dir pytest-random-order
RUN uv pip install --no-cache-dir 'transformers-ci[otel] @ git+https://github.com/huggingface/transformers-ci@main'

RUN uv pip uninstall transformers
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get autoremove && apt-get autoclean
