# Experimental transformers CI image for AMD Ryzen "Strix Halo" (gfx1151).
#
# Base is plain Ubuntu 24.04 (system python 3.12, matching the cp312 wheels
# below). No ROCm in the base: the torch wheels pull their own ROCm userspace
# from pip (rocm-sdk-core / rocm-sdk-libraries), so a ROCm base image would only
# add ~20 GB of unused libraries. The host still supplies the amdgpu/KFD kernel
# driver; run with --device=/dev/kfd --device=/dev/dri (see the CI workflow).
#
# The ROCm userspace does NOT need to match the pool's driver. This image runs
# on the pool as-is (inbox amdgpu in the 6.17 OEM kernel, 7.2.1-era host, no
# amdgpu-dkms). `hipErrorInvalidImage` on gfx1151 means missing gfx1151 device
# code in the wheels, not driver skew. download.pytorch.org/whl/rocm7.14 now
# includes gfx1151, and on ROCm 7.14 the flash / efficient SDPA backends run
# instead of falling back to math.
FROM ubuntu:24.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

# The torch wheel is not self-contained: it links libatomic1 and dlopens
# libnuma1, libgomp1, libdrm2, libdrm-amdgpu1 and libelf1. A missing libnuma1 in
# particular SEGFAULTS on the first GPU op rather than erroring cleanly --
# install them here.
# apt resilience: a build hit intermittent connect failures to
# archive.ubuntu.com from the pool, so retry a few times before giving up.
# tesseract-ocr + espeak-ng are needed by OCR / TTS model tests.
RUN printf 'Acquire::Retries "8";\nAcquire::http::Timeout "30";\n' > /etc/apt/apt.conf.d/99retries && \
    apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
      git git-lfs ffmpeg libsndfile1-dev espeak-ng \
      tesseract-ocr tesseract-ocr-eng \
      build-essential pkg-config curl ca-certificates \
      python3 python3-dev python3-pip python3-venv \
      libatomic1 libnuma1 libgomp1 libdrm2 libdrm-amdgpu1 libelf1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install

# Install into a venv: Ubuntu 24.04 marks the system python externally-managed
# (PEP 668), which would otherwise block pip.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}

RUN pip install --no-cache-dir --upgrade pip \
    numpy importlib-metadata setuptools wheel ninja pytesseract "itsdangerous<2.1.0"

# gfx1151 torch stack, from PyTorch's own ROCm 7.14 channel (cp312, matching
# the base image's python 3.12). torchvision includes libpng/libjpeg, so
# torchvision.io.decode_image works.
ARG TORCH_INDEX=https://download.pytorch.org/whl/rocm7.14
ARG TORCH_VERSION=2.14.0+rocm7.14
ARG TORCHVISION_VERSION=0.29.0+rocm7.14

# --- Keep only one GPU architecture (saves ~10 GB) ---------------------------
# torch requires rocm[device-all], which installs ROCm libraries for all 25
# supported architectures (torch itself has kernels for 13 of them). When
# TRIM_OTHER_ARCHS=1, the install step below deletes the files of every
# rocm-sdk-device-* package except GPU_ARCH. Their metadata is kept, so later
# pip calls still see torch's requirements as met. This has to happen in the
# same RUN as the install: deleting in a later RUN would not shrink the image.
#   GPU_ARCH           must match the GPU this image runs on. For another GPU,
#                      also change FA_WHEEL below, which is built for gfx1151.
#   TRIM_OTHER_ARCHS   1 = keep only GPU_ARCH (default); 0 = keep all 25.
#                      Any other value fails the build.
ARG GPU_ARCH=gfx1151
ARG TRIM_OTHER_ARCHS=1
RUN python3 -m pip install --no-cache-dir --index-url ${TORCH_INDEX} \
      "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" && \
    case "${TRIM_OTHER_ARCHS}" in \
    0) echo "TRIM_OTHER_ARCHS=0: keeping device code for all architectures" ;; \
    1) python3 -c "import glob, os, re, sysconfig; \
site = sysconfig.get_paths()['purelib']; \
files = lambda rec: {line.split(',')[0] for line in open(rec)}; \
kept = glob.glob(site + '/rocm_sdk_device_${GPU_ARCH}-*.dist-info/RECORD'); \
assert kept, 'no rocm-sdk-device-${GPU_ARCH} package installed'; \
keep = files(kept[0]); \
drop = {p for rec in glob.glob(site + '/rocm_sdk_device_gfx*.dist-info/RECORD') for p in files(rec) \
        if re.match(r'_rocm_sdk_libraries/(lib|share|\.kpack)/', p)} - keep; \
paths = [os.path.join(site, p) for p in drop if os.path.isfile(os.path.join(site, p))]; \
print('removing', sum(map(os.path.getsize, paths)) >> 20, 'MB of non-${GPU_ARCH} device code'); \
list(map(os.remove, paths))" ;; \
    *) echo "TRIM_OTHER_ARCHS must be 0 or 1, got '${TRIM_OTHER_ARCHS}'" >&2; exit 1 ;; \
    esac

# torchaudio: 2.11 is its final release and works with every later torch. The
# PyPI wheel is built for CUDA and fails to load here, so this is AMD's ROCm
# build (download.pytorch.org has none for rocm7.14). --no-deps so it cannot
# pull a different torch. Do not bump to 2.11.0.2 without testing: its GPU
# kernels do not include gfx1151.
ARG TORCHAUDIO_INDEX=https://repo.amd.com/rocm/whl-multi-arch/
ARG TORCHAUDIO_VERSION=2.11.0+rocm7.14.1
RUN python3 -m pip install --no-cache-dir --no-deps --index-url ${TORCHAUDIO_INDEX} \
      "torchaudio==${TORCHAUDIO_VERSION}"

# Pin torch/torchvision for every later pip call. Without it, a package that
# depends on torch (e.g. timm) can let pip replace the ROCm build with a CUDA
# wheel from PyPI: the image still builds, and the tests silently run on CPU.
RUN printf 'torch==%s\ntorchvision==%s\n' "${TORCH_VERSION}" "${TORCHVISION_VERSION}" > /opt/rocm-constraints.txt
ENV PIP_CONSTRAINT=/opt/rocm-constraints.txt

# Flash-Attention (Triton-AMD wheel for gfx1151). --no-deps so it cannot pull a
# non-gfx1151 torch over the one installed above. Pinned to the torch/ROCm ABI
# of the wheels above; bumping the stack generally needs a matching FA wheel.
ARG FA_WHEEL=https://rocm.frameworks-nightlies.amd.com/whl/gfx1151/flash_attn-2.8.3-py3-none-any.whl
RUN python3 -m pip install --no-cache-dir --no-deps ${FA_WHEEL}

ARG REF=main
WORKDIR /

# Clone transformers at REF. (No api.github.com cache-buster ADD here: on the
# GitHub-hosted buildx builder that unauthenticated call is rate-limited to 403,
# and the scheduled builder has no persistent layer cache to invalidate anyway.)
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

# Install transformers (editable) + test deps.
RUN python3 -m pip install --no-cache-dir -e ./transformers[testing]

# When installing in editable mode, `transformers` is not always recognized as
# a package; this line makes python aware of it (same as the MI300 image).
RUN cd transformers && python3 setup.py develop

# Remove nvml / nvidia-ml as they are not compatible with ROCm. apex is not
# tested on NVIDIA either.
RUN python3 -m pip uninstall -y py3nvml pynvml nvidia-ml-py apex || true

# `kernels` may cause many failing tests.
RUN python3 -m pip uninstall -y kernels || true

# Common extra deps used across the test suite. scipy, librosa and soundfile
# are easy to miss: transformers' lazy imports turn their absence into errors
# like `NameError: name 'EncodecFeatureExtractor' is not defined`, not
# ImportError.
RUN python3 -m pip install --no-cache-dir einops blobfile num2words timm \
      scipy librosa soundfile

# Media decoders needed by audio/video model tests:
#  - PyAV (`av`): torchvision video ops
#  - torchcodec: audio/video decoding (pinned to the torch 2.14 ABI; --no-deps
#    so it can't drag in a different torch). Best-effort: a torchcodec mismatch
#    must not fail the whole image build.
RUN python3 -m pip install --no-cache-dir av
RUN python3 -m pip install --no-cache-dir --no-deps torchcodec==0.13.0 || \
    python3 -m pip install --no-cache-dir --no-deps torchcodec || \
    echo "torchcodec install skipped"

# Pre-fetch the test fixtures into the image, as docker/torch-light.dockerfile
# does. Otherwise every job downloads them from the Hub, and with many jobs
# behind one IP the Hub returns HTTP 429 -- which load_image() hands to PIL, so
# it shows up as `PIL.UnidentifiedImageError`, not as a rate limit.
# Runs from /transformers: url_to_local_path() resolves against the CWD.
# HF_TOKEN is optional, passed as a BuildKit secret so it stays out of the
# image:  docker build --secret id=hf_token,env=HF_TOKEN ...
RUN --mount=type=secret,id=hf_token \
    cd /transformers && \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
      python3 utils/fetch_hub_objects_for_ci.py

# Fail the build, rather than the test run, if the torch stack is wrong.
RUN python3 -c "import torch; v=torch.__version__; assert 'rocm' in v, \
      f'torch is {v} -- a non-ROCm build replaced it'; print('verified torch:', v)"
RUN python3 -c "import torch, torchaudio, torchaudio.transforms as T; \
      T.MelSpectrogram(16000)(torch.randn(1,16000)); \
      print('torchaudio ok:', torchaudio.__version__)"

# --- Runtime environment -----------------------------------------------------
# REQUIRED for the FA Triton-AMD backend on gfx1151 -- without it the backend
# SIGSEGVs and takes pytest down with it.
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

# NOTE: do NOT point TMPDIR / Triton / inductor caches at a /scratch that the
# workflow bind-mounts over -- the mount hides the dirs created here, leaving
# TMPDIR non-existent, and ROCm then segfaults on the first kernel launch.
# Container /tmp (overlay, disk-backed) is fine. If a dedicated cache is wanted
# later, create it at runtime inside whatever is mounted.
