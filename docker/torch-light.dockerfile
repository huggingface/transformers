FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
ARG REF=main
USER root
RUN apt-get update &&  apt-get install -y --no-install-recommends libsndfile1-dev espeak-ng time git g++ cmake pkg-config openssh-client git-lfs ffmpeg curl
ENV UV_PYTHON=/usr/local/bin/python
RUN pip --no-cache-dir install uv && uv pip install --no-cache-dir -U pip setuptools
RUN uv pip install --no-cache-dir 'torch<=2.11.0' 'torchaudio' 'torchvision' 'torchcodec<=0.11.0' --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-deps timm accelerate --extra-index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache-dir librosa "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[sklearn,sentencepiece,vision,testing,tiktoken,num2words,video]"

# Use a custom patched pytest to force exit the process at the end, to avoid `Too long with no output (exceeded 10m0s): context deadline exceeded` (#40201)
RUN uv pip install --no-cache-dir git+https://github.com/ydshieh/pytest.git@8.4.1-ydshieh
RUN uv pip install --no-cache-dir pytest-random-order
RUN uv pip install --no-cache-dir 'transformers-ci[otel] @ git+https://github.com/huggingface/transformers-ci@main'

# fetch test data and hub objects within CircleCI docker images to reduce even more connections
# we don't need a full clone of `transformers` to run `fetch_hub_objects_for_ci.py`
# the data are downloaded to the directory `/test_data` and during CircleCI's CI runtime, we need to move them to the root of `transformers`
RUN mkdir test_data && cd test_data && curl -O https://raw.githubusercontent.com/huggingface/transformers/${REF}/utils/fetch_hub_objects_for_ci.py && python3 fetch_hub_objects_for_ci.py

RUN uv pip uninstall transformers
