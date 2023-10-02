FROM rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_2.0.1
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y --no-install-recommends git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools ninja git+https://github.com/facebookresearch/detectron2.git pytesseract "itsdangerous<2.1.0"

# If set to nothing, will install the latest version
ARG PYTORCH='2.0.1'
ARG TORCH_VISION='0.15.2'
ARG TORCH_AUDIO='2.0.2'
ARG ROCM='5.6'

RUN git clone --depth 1 --branch v$TORCH_AUDIO https://github.com/pytorch/audio.git
RUN cd audio && USE_ROCM=1 USE_CUDA=0 python setup.py install

ARG REF=main
WORKDIR /
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF
RUN python3 -m pip install --no-cache-dir -e ./transformers[dev-torch,testing,video]

RUN python3 -m pip uninstall -y tensorflow flax

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop
