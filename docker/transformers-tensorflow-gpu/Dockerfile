FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF
RUN python3 -m pip install --no-cache-dir -e ./transformers[dev-tensorflow,testing]

# If set to nothing, will install the latest version
ARG TENSORFLOW='2.12'

RUN [ ${#TENSORFLOW} -gt 0 ] && VERSION='tensorflow=='$TENSORFLOW'.*' ||  VERSION='tensorflow'; python3 -m pip install --no-cache-dir -U $VERSION
RUN python3 -m pip uninstall -y torch flax
RUN python3 -m pip install -U "itsdangerous<2.1.0"

RUN python3 -m pip install --no-cache-dir -U tensorflow_probability

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop
