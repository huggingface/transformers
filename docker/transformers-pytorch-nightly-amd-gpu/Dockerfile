FROM rocm/dev-ubuntu-20.04:5.7 
# NOTE: we do not use rocm/pytorch-nightly:latest image since it uses RoCm 5.6, and we would rather like to
# test against the publicly recommended PyTorch build at https://pytorch.org/get-started/locally/.

LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y --no-install-recommends git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-dev python3-pip ffmpeg && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.7

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools ninja git+https://github.com/facebookresearch/detectron2.git pytesseract "itsdangerous<2.1.0"

ARG PYTORCH='2.2.0.dev+rocm5.7'
ARG TORCH_VISION='0.17.0.dev+rocm5.7'
ARG TORCH_AUDIO='2.2.0.dev+rocm5.7'
ARG ROCM='5.7'

ARG REF=main
WORKDIR /
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF
RUN python3 -m pip install --no-cache-dir -e ./transformers[dev-torch,testing,video]

RUN python3 -m pip uninstall -y tensorflow flax

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop
