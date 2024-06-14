FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

# Use login shell to read variables from `~/.profile` (to pass dynamic created variables between RUN commands)
SHELL ["sh", "-lc"]

# The following `ARG` are mainly used to specify the versions explicitly & directly in this docker file, and not meant
# to be used as arguments for docker build (so far).

ARG PYTORCH='2.2.1'
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu118'

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python python3-pip ffmpeg
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; echo "export VERSION='$VERSION'" >> ~/.profile
RUN echo torch=$VERSION
# `torchvision` and `torchaudio` should be installed along with `torch`, especially for nightly build.
# Currently, let's just use their latest releases (when `torch` is installed with a release version)
RUN python3 -m pip install --no-cache-dir -U $VERSION torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN python3 -m pip install --no-cache-dir -e ./transformers[dev-torch]

RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/accelerate@main#egg=accelerate

# needed in bnb and awq
RUN python3 -m pip install --no-cache-dir einops

# Add bitsandbytes for mixed int8 testing
RUN python3 -m pip install --no-cache-dir bitsandbytes

# Add auto-gptq for gtpq quantization testing
RUN python3 -m pip install --no-cache-dir auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

# Add optimum for gptq quantization testing
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/optimum@main#egg=optimum

# Add aqlm for quantization testing
RUN python3 -m pip install --no-cache-dir aqlm[gpu]==1.0.2

# Add hqq for quantization testing
RUN python3 -m pip install --no-cache-dir hqq

# For GGUF tests
RUN python3 -m pip install --no-cache-dir gguf

# Add autoawq for quantization testing
# >=v0.2.3 needed for compatibility with torch 2.2.1
RUN python3 -m pip install --no-cache-dir https://github.com/casper-hansen/AutoAWQ/releases/download/v0.2.3/autoawq-0.2.3+cu118-cp38-cp38-linux_x86_64.whl

# Add quanto for quantization testing
RUN python3 -m pip install --no-cache-dir quanto

# Add eetq for quantization testing
RUN python3 -m pip install git+https://github.com/NetEase-FuXi/EETQ.git

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop
