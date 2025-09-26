# Stage 1: Base environment setup
FROM python:3.10-slim as base
ENV PYTHONDONTWRITEBYTECODE=1
ARG REF=main
USER root
RUN apt-get update &&  apt-get install -y --no-install-recommends libsndfile1-dev espeak-ng time git g++ cmake pkg-config openssh-client git-lfs ffmpeg curl
ENV UV_PYTHON=/usr/local/bin/python
RUN pip --no-cache-dir install uv && uv pip install --no-cache-dir -U pip setuptools

# Stage 2: Install dependencies AND download models (this will be thrown away)
FROM base as downloader
RUN uv pip install --no-cache-dir 'torch' 'torchaudio' 'torchvision' 'torchcodec' --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-deps timm accelerate --extra-index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache-dir librosa "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[sklearn,sentencepiece,vision,testing,tiktoken,num2words,video]"

# Download test data
RUN mkdir test_data && cd test_data && curl -O https://raw.githubusercontent.com/huggingface/transformers/${REF}/utils/fetch_hub_objects_for_ci.py && python3 fetch_hub_objects_for_ci.py

# Download and run the script that caches models
RUN curl -O https://raw.githubusercontent.com/huggingface/transformers/${REF}/call_from_pretrained.py
RUN curl -O https://raw.githubusercontent.com/huggingface/transformers/${REF}/calls.json
RUN python call_from_pretrained.py

# Stage 3: Final image - copy EVERYTHING from downloader (dependencies + cleaned cache)
FROM base as final
# Copy the entire Python environment from the downloader stage
COPY --from=downloader /usr/local /usr/local
# Copy test data and cleaned cache
COPY --from=downloader /test_data /test_data
COPY --from=downloader /root/.cache/huggingface /root/.cache/huggingface

# Uninstall transformers as in your original
RUN uv pip uninstall transformers