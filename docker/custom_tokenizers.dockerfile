FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
USER root
RUN apt-get update && apt-get install -y libsndfile1-dev espeak-ng time git
ENV VIRTUAL_ENV=/usr/local
RUN pip --no-cache-dir install uv
RUN uv venv
RUN uv pip install --no-cache-dir -U pip setuptools
RUN uv pip install --no-cache "pytest<8.0.1" "fsspec>=2023.5.0,<2023.10.0" pytest-subtests pytest-xdist
# END COMMON LAYERS

RUN apt-get update && apt-get install -y cmake wget xz-utils build-essential g++5 libprotobuf-dev protobuf-compiler
RUN wget https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc4/jumanpp-2.0.0-rc4.tar.xz
RUN tar xvf jumanpp-2.0.0-rc4.tar.xz
RUN mkdir jumanpp-2.0.0-rc4/bld
WORKDIR ./jumanpp-2.0.0-rc4/bld
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
RUN make install

RUN uv pip install --no-cache --upgrade 'torch' --index-url https://download.pytorch.org/whl/cpu accelerate
RUN uv pip install  --no-cache-dir "transformers[ja,testing,sentencepiece,jieba,spacy,ftfy,rjieba]" unidic
RUN python3 -m unidic download
RUN pip uninstall -y transformers
RUN uv pip install --no-cache-dir  unidic-lite

RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip cache remove "nvidia-*" triton

RUN apt remove -y g++ cmake  xz-utils build-essential libprotobuf-dev protobuf-compiler