FROM rocm/pytorch:rocm10.1.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y --no-install-recommends git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-dev python3-pip ffmpeg git-lfs libjpeg-turbo8-dev libpng-dev zlib1g-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install

RUN python3 -m pip install --no-cache-dir --upgrade pip numpy importlib-metadata setuptools wheel ninja pytesseract "itsdangerous<2.1.0"

# Rebuild torchvision so decode_image has libjpeg and ROCm image ops stay on GPU.
RUN python3 -m pip install --no-cache-dir setuptools pybind11
RUN TV_VERSION=$(python3 -c "import torchvision; print(torchvision.__version__.split('+')[0])") && \
    python3 -m pip uninstall -y torchvision && \
    git clone --depth 1 --branch "v${TV_VERSION}" https://github.com/pytorch/vision.git /tmp/vision && \
    cd /tmp/vision && \
    sed -i -E 's|list\(CSRS_DIR\.glob\("([^"]+\.cpp)"\)\)|[p for p in CSRS_DIR.glob("\1") if not p.name.endswith("_hip.cpp")]|g' setup.py && \
    FORCE_CUDA=1 TORCHVISION_USE_FFMPEG=0 TORCHVISION_USE_VIDEO_CODEC=0 \
    python3 -m pip install --no-cache-dir --no-build-isolation -v . && \
    cd / && rm -rf /tmp/vision

RUN python3 -m pip install --no-cache-dir --no-build-isolation git+https://github.com/facebookresearch/detectron2.git

ARG REF=main
WORKDIR /

# Invalidate docker cache from here if new commit is available.
ADD https://api.github.com/repos/huggingface/transformers/git/refs/heads/main version.json
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

# Install transformers
RUN python3 -m pip install --no-cache-dir -e ./transformers[dev,testing,video,audio]

# `pyctcdecode` (all releases) pins numpy<2, so on Python 3.14 pip builds numpy 1.26 from source over the image's
# numpy 2.x; pyarrow then fails to import and every test module importing `datasets` errors at collection.
# pyctcdecode works with numpy 2, so restore it.
RUN python3 -m pip install --no-cache-dir --upgrade "numpy>=2.3"

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

# Remove nvml and nvidia-ml-py as it is not compatible with ROCm. apex is not tested on NVIDIA either.
RUN python3 -m pip uninstall py3nvml pynvml nvidia-ml-py apex -y

# `kernels` may causes many failing tests
RUN python3 -m pip uninstall -y kernels

# On ROCm, torchcodec is required to decode audio files. 0.13 matches the
# torch 2.13 C++ ABI in the base image above;
RUN python3 -m pip install --no-cache-dir "torchcodec==0.13.0"

# Install pre-built flash-attention wheel from AMD's mirror. Wheel is pinned
# to the (ROCm, Ubuntu, Python, torch) ABI of the base image above; bumping
# the base image generally requires a matching wheel from AMD.
RUN python3 -m pip install --no-cache-dir \
    https://rocm.frameworks.amd.com/whl-multi-arch/vllm/flash-attn/flash_attn-2.8.3-cp314-cp314-linux_x86_64.whl

# `kenlm` (wav2vec2 + LM tests) is only pulled by the `audio` extra for Python < 3.13, and its published sources
# ship a Cython-generated `kenlm.cpp` that does not compile on 3.14: regenerate it with a recent Cython.
RUN python3 -m pip install --no-cache-dir cython && \
    git clone --depth 1 https://github.com/kpu/kenlm /tmp/kenlm && \
    cd /tmp/kenlm && \
    cython --cplus python/kenlm.pyx -o python/kenlm.cpp && \
    python3 -m pip install --no-cache-dir --no-build-isolation . && \
    cd / && rm -rf /tmp/kenlm

RUN python3 -m pip install --no-cache-dir einops blobfile num2words

# timm is required for many vision models tests
RUN python3 -m pip install --no-cache-dir timm

# Tests that NVIDIA CI runs and that are skipped without it
RUN python3 -m pip install --no-cache-dir onnxruntime

# MI300: two implicit-GEMM 1x1-conv solvers round bf16 differently and MIOpen picks between them by start-up
# benchmarking, which flips under GPU contention (Gemma3/Gemma3n/Qwen2.5-Omni vision towers). Route 1x1 convs
# to the GEMM path and disable runtime benchmarking so results are deterministic.
ENV MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
ENV MIOPEN_FIND_MODE=2

# Fail the build early if the dependency installs replaced the ROCm torch or broke flash-attn.
RUN python3 -c "import torch, torchvision, torchcodec, flash_attn, kenlm, numpy, pyarrow, datasets, transformers; assert torch.version.hip is not None; print(torch.__version__, torch.version.hip, torchvision.__version__, flash_attn.__version__)"
