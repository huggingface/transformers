FROM rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y --no-install-recommends git libsndfile1-dev tesseract-ocr espeak-ng ffmpeg git-lfs libjpeg-turbo8-dev libpng-dev zlib1g-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install

RUN python3 -m pip install --no-cache-dir --upgrade pip numpy importlib-metadata setuptools wheel ninja pytesseract "itsdangerous<2.1.0"

# Rebuild torchvision so decode_image has libjpeg and ROCm image ops stay on GPU.
RUN python3 -m pip install --no-cache-dir "setuptools<81" pybind11
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

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

# Remove nvml and nvidia-ml-py as it is not compatible with ROCm. apex is not tested on NVIDIA either.
RUN python3 -m pip uninstall py3nvml pynvml nvidia-ml-py apex -y

# `kernels` may causes many failing tests
RUN python3 -m pip uninstall -y kernels

# On ROCm, torchcodec is required to decode audio files. 0.17 supports torch >= 2.11.
# Use the CPU build: the PyPI wheel links its image library against CUDA (libnvrtc.so.13) and fails to import on ROCm.
RUN python3 -m pip install --no-cache-dir --no-deps "torchcodec==0.17.0+cpu" --index-url https://download.pytorch.org/whl/cpu

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

# Fail the build early if the dependency installs replaced the ROCm torch or broke flash-attn.
RUN python3 -c "import torch, torchvision, torchcodec, flash_attn, kenlm, transformers; assert torch.version.hip is not None; print(torch.__version__, torch.version.hip, torchvision.__version__, flash_attn.__version__)"
