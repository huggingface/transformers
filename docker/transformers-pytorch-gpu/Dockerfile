FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF
RUN python3 -m pip install --no-cache-dir -e ./transformers[dev-torch,testing]

# If set to nothing, will install the latest version
ARG PYTORCH=''

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; python3 -m pip install --no-cache-dir -U $VERSION
RUN python3 -m pip uninstall -y tensorflow flax

RUN python3 -m pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-$(python3 -c "from torch import version; print(version.__version__.split('+')[0])")+cu102.html
RUN python3 -m pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git pytesseract https://github.com/kpu/kenlm/archive/master.zip
RUN python3 -m pip install -U "itsdangerous<2.1.0"

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop
