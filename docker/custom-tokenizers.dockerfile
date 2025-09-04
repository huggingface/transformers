FROM python:3.9-slim
ENV PYTHONDONTWRITEBYTECODE=1
ARG REF=main
USER root
RUN apt-get update && apt-get install -y libsndfile1-dev espeak-ng time git cmake wget xz-utils build-essential g++5 libprotobuf-dev protobuf-compiler git-lfs
ENV UV_PYTHON=/usr/local/bin/python
RUN pip --no-cache-dir install uv && uv pip install --no-cache-dir -U pip setuptools

RUN wget https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc3/jumanpp-2.0.0-rc3.tar.xz
RUN tar xvf jumanpp-2.0.0-rc3.tar.xz
RUN mkdir jumanpp-2.0.0-rc3/bld
WORKDIR ./jumanpp-2.0.0-rc3/bld
RUN wget -LO catch.hpp https://github.com/catchorg/Catch2/releases/download/v2.13.8/catch.hpp
RUN mv catch.hpp ../libs/
RUN cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
RUN make install -j 10


RUN uv pip install --no-cache --upgrade 'torch' --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache-dir  --no-deps accelerate --extra-index-url https://download.pytorch.org/whl/cpu
RUN uv pip install  --no-cache-dir "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[ja,testing,sentencepiece,spacy,ftfy,rjieba]" unidic unidic-lite
# spacy is not used so not tested. Causes to failures. TODO fix later
RUN uv run python -m unidic download
RUN uv pip uninstall transformers

RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt remove -y g++ cmake  xz-utils libprotobuf-dev protobuf-compiler
