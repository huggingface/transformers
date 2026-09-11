# Experimental transformers CI image for AMD Ryzen "Strix Halo" (gfx1151).
#
# Unlike the MI300 image (transformers-pytorch-amd-gpu), there is no
# rocm/pytorch base for gfx1151 yet, so we start from python-slim and pull the
# torch stack from AMD's ROCm manylinux wheels. The host still needs the
# amdgpu/KFD kernel driver; run the container with --device=/dev/kfd
# --device=/dev/dri (see the CI workflow).
#
# The wheels are pinned to ROCm 7.2.1 to MATCH THE POOL'S DRIVER (measured on a
# runner: ROCm userspace 7.2.1, HIP 7.2, inbox amdgpu in the 6.17 OEM kernel --
# no amdgpu-dkms). Two dead ends that both give `hipErrorInvalidImage`:
#   * download.pytorch.org/whl/rocm* -- built against upstream's gfx target list,
#     which omits gfx1151.
#   * AMD TheRock gfx1151 nightlies -- only 7.0rc / 7.9+ exist there, i.e. a
#     userspace far newer than the 7.2 kernel driver can load.
# repo.radeon.com's rocm-rel-7.2.1 builds advertise gfx1151 AND match the driver.
FROM python:3.12-slim
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git git-lfs ffmpeg libsndfile1-dev espeak-ng tesseract-ocr \
      build-essential pkg-config curl ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    numpy importlib-metadata setuptools wheel ninja pytesseract "itsdangerous<2.1.0"

# gfx1151 torch stack, ROCm 7.2.1 manylinux wheels from repo.radeon.com (cp312,
# matching the python:3.12 base). --find-links lets pip resolve torch's matching
# pytorch-triton-rocm from the same directory. Bump all three together (and mind
# the git-hash in each filename) if the pool's ROCm version changes.
ARG ROCM_REPO=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1
ARG TORCH_WHL=torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl
ARG TORCHVISION_WHL=torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl
ARG TORCHAUDIO_WHL=torchaudio-2.9.0+rocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl
RUN python3 -m pip install --no-cache-dir --find-links ${ROCM_REPO}/ \
      ${ROCM_REPO}/${TORCH_WHL} \
      ${ROCM_REPO}/${TORCHVISION_WHL} \
      ${ROCM_REPO}/${TORCHAUDIO_WHL}

# Flash-Attention (Triton-AMD wheel for gfx1151). --no-deps so it cannot pull a
# non-gfx1151 torch over the one installed above. Pinned to the torch/ROCm ABI
# of the wheels above; bumping the stack generally needs a matching FA wheel.
ARG FA_WHEEL=https://rocm.frameworks-nightlies.amd.com/whl/gfx1151/flash_attn-2.8.3-py3-none-any.whl
RUN python3 -m pip install --no-cache-dir --no-deps ${FA_WHEEL}

ARG REF=main
WORKDIR /

# Invalidate the docker cache from here if a new commit is available on main.
ADD https://api.github.com/repos/huggingface/transformers/git/refs/heads/main version.json
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

# Common extra deps used across the test suite.
RUN python3 -m pip install --no-cache-dir einops blobfile num2words timm

# --- Runtime environment -----------------------------------------------------
# REQUIRED for the FA Triton-AMD backend on gfx1151 -- without it the backend
# SIGSEGVs and takes pytest down with it.
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

# Keep Triton / inductor / tmp caches on disk rather than RAM: this APU has a
# single 62 GiB pool shared between CPU and GPU, so a RAM-backed /tmp steals VRAM.
# At CI runtime, back /scratch with a host volume on the runner's large disk.
ENV TRITON_CACHE_DIR=/scratch/triton \
    TORCHINDUCTOR_CACHE_DIR=/scratch/inductor \
    TMPDIR=/scratch/tmp
RUN mkdir -p /scratch/triton /scratch/inductor /scratch/tmp
