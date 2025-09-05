FROM python:3.9-slim
ENV PYTHONDONTWRITEBYTECODE=1
ARG REF=main
USER root
RUN apt-get update &&  apt-get install -y --no-install-recommends libsndfile1-dev espeak-ng time git g++ cmake pkg-config openssh-client git-lfs ffmpeg curl
ENV UV_PYTHON=/usr/local/bin/python
RUN pip --no-cache-dir install uv && uv pip install --no-cache-dir -U pip setuptools
RUN uv pip install --no-cache-dir 'torch' 'torchaudio' 'torchvision' 'torchcodec' --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-deps timm accelerate --extra-index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache-dir librosa "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[sklearn,sentencepiece,vision,testing]" seqeval albumentations jiwer

# fetch test data and hub objects within CircleCI docker images to reduce even more connections
# we don't need a full clone of `transformers` to run `fetch_hub_objects_for_ci.py`
# the data are downloaded to the directory `/test_data` and during CircleCI's CI runtime, we need to move them to the root of `transformers`
RUN mkdir test_data && cd test_data && curl -O https://raw.githubusercontent.com/huggingface/transformers/${REF}/utils/fetch_hub_objects_for_ci.py && python3 fetch_hub_objects_for_ci.py


RUN uv pip uninstall transformers
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
