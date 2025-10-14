FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
ARG REF=main
USER root
RUN apt-get update && apt-get install -y libsndfile1-dev espeak-ng time git libgl1 g++ tesseract-ocr git-lfs curl
ENV UV_PYTHON=/usr/local/bin/python
RUN pip --no-cache-dir install uv && uv pip install --no-cache-dir -U pip setuptools
RUN uv pip install --no-cache-dir 'torch' 'torchaudio' 'torchvision' --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache-dir  --no-deps timm accelerate
RUN uv pip install -U --no-cache-dir pytesseract python-Levenshtein opencv-python nltk
# RUN uv pip install --no-cache-dir natten==0.15.1+torch210cpu -f https://shi-labs.com/natten/wheels
RUN uv pip install  --no-cache-dir "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[testing, vision]" 'scikit-learn' 'torch-stft' 'nose'  'dataset'
# RUN git clone https://github.com/facebookresearch/detectron2.git
# RUN python3 -m pip install --no-cache-dir -e detectron2
RUN uv pip install 'git+https://github.com/facebookresearch/detectron2.git@92ae9f0b92aba5867824b4f12aa06a22a60a45d3' --no-build-isolation

# fetch test data and hub objects within CircleCI docker images to reduce even more connections
# we don't need a full clone of `transformers` to run `fetch_hub_objects_for_ci.py`
# the data are downloaded to the directory `/test_data` and during CircleCI's CI runtime, we need to move them to the root of `transformers`
RUN mkdir test_data && cd test_data && curl -O https://raw.githubusercontent.com/huggingface/transformers/${REF}/utils/fetch_hub_objects_for_ci.py && python3 fetch_hub_objects_for_ci.py


RUN uv pip uninstall transformers
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
