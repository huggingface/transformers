FROM rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y --no-install-recommends git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-dev python3-pip python3-dev ffmpeg git-lfs libjpeg-turbo8-dev libpng-dev zlib1g-dev && \
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

# On ROCm, torchcodec is required to decode audio files and 0.4 or 0.6 fails
RUN python3 -m pip install --no-cache-dir "torchcodec==0.7"

# Install flash attention from source. Pinned to commit 6387433156558135a998d5568a9d74c1778666d8
# to avoid compilation time exceeding the 6h GitHub Actions job limit.
RUN git clone https://github.com/ROCm/flash-attention/ -b tridao && \
    cd flash-attention && \
    git checkout 6387433156558135a998d5568a9d74c1778666d8 && \
    GPU_ARCHS="gfx942" python setup.py install
# GPU_ARCHS builds for MI300, MI325 but not MI355: we would need to add `;gfx950` but it takes too long to build.

RUN python3 -m pip install --no-cache-dir einops blobfile num2words

# timm is required for many vision models tests
RUN python3 -m pip install --no-cache-dir timm
