FROM rocm/pytorch:rocm6.3.4_ubuntu22.04_py3.10_pytorch_release_2.4.0
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

ENV MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
ENV MIOPEN_FIND_MODE=1
ENV HSA_ENABLE_SDMA=0
ENV TORCH_PREFER_HIPBLASLT=0

RUN apt update && \
    apt install -y --no-install-recommends git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-dev python3-pip python3-dev ffmpeg git-lfs && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install

RUN rm -rf $(pip show numpy | grep Location: | awk '{print $2}')/numpy*
RUN python3 -m pip install --no-cache-dir --upgrade pip numpy

RUN python3 -m pip install --no-cache-dir --upgrade importlib-metadata setuptools ninja git+https://github.com/facebookresearch/detectron2.git pytesseract "itsdangerous<2.1.0"

ARG REF=main
WORKDIR /

# Invalidate docker cache from here if new commit is available.
ADD https://api.github.com/repos/huggingface/transformers/git/refs/heads/main version.json
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

RUN python3 -m pip install --no-cache-dir -e ./transformers[dev-torch,testing,video]

RUN python3 -m pip uninstall -y tensorflow flax

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

# Remove nvml and nvidia-ml-py as it is not compatible with ROCm. apex is not tested on NVIDIA either.
RUN python3 -m pip uninstall py3nvml pynvml nvidia-ml-py apex -y
RUN pip install phonemizer pyctcdecode accelerate
RUN pip install --upgrade botocore


ENV CMAKE_VERSION=3.26.4
RUN cd /usr/local && \
    wget -q -O - https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz | tar zxf -

ENV PATH=/usr/local/cmake-${CMAKE_VERSION}-linux-x86_64/bin:${PATH}


RUN python3 -m pip uninstall -y torch torchvision torchaudio
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3

RUN git clone --recurse https://github.com/ROCm/bitsandbytes && \
    cd bitsandbytes && \
    git checkout rocm_enabled_multi_backend && \
    pip install -r requirements-dev.txt && \
    cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx942" -S . && \
    make && \
    pip install .

ARG FLASH_ATT_V2_COMMIT_ROCM=2554f490101742ccdc56620a938f847f61754be6

RUN git clone https://github.com/ROCm/flash-attention.git flash-attention-v2 && \
    cd flash-attention-v2 && git submodule update --init --recursive && \
    GPU_ARCHS="gfx90a;gfx942" PYTORCH_ROCM_ARCH="gfx90a;gfx942" python setup.py install && \
    cd .. && \
    rm -rf flash-attention
