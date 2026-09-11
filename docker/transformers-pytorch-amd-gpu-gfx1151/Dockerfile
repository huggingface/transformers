# Experimental transformers CI image for AMD Ryzen "Strix Halo" (gfx1151).
#
# Base is a ROCm 7.2.1 image (Ubuntu 24.04 -> system python 3.12, matching the
# cp312 wheels below). We need the ROCm *userspace* libraries here: the
# repo.radeon.com manylinux torch wheels link against system ROCm (unlike the
# TheRock wheels, which bundle it) -- on a plain python-slim base torch fails to
# import with `libroctx64.so.4: cannot open shared object file`. The host still
# supplies the amdgpu/KFD kernel driver; run with --device=/dev/kfd
# --device=/dev/dri (see the CI workflow).
#
# ROCm is pinned to 7.2.1 to MATCH THE POOL'S DRIVER (measured on a runner: ROCm
# userspace 7.2.1, HIP 7.2, inbox amdgpu in the 6.17 OEM kernel -- no
# amdgpu-dkms). Two dead ends that both give `hipErrorInvalidImage`:
#   * download.pytorch.org/whl/rocm* -- built against upstream's gfx target list,
#     which omits gfx1151.
#   * AMD TheRock gfx1151 nightlies -- only 7.0rc / 7.9+ exist there, i.e. a
#     userspace far newer than the 7.2 kernel driver can load.
# repo.radeon.com's rocm-rel-7.2.1 builds advertise gfx1151 AND match the driver.
FROM rocm/dev-ubuntu-24.04:7.2.1-complete
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

# apt resilience: a build hit intermittent connect failures to
# archive.ubuntu.com from the pool, so retry a few times before giving up.
# (espeak-ng / tesseract-ocr are omitted for now -- not needed by the current
# smoke + test slice; add them back once the apt path is reliable.)
RUN printf 'Acquire::Retries "8";\nAcquire::http::Timeout "30";\n' > /etc/apt/apt.conf.d/99retries && \
    apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
      git git-lfs ffmpeg libsndfile1-dev \
      build-essential pkg-config curl ca-certificates \
      python3 python3-dev python3-pip python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install

# Install into a venv: Ubuntu 24.04 marks the system python externally-managed
# (PEP 668), which would otherwise block pip. The ROCm libs live in /opt/rocm
# (on the system ld path) and are reachable from the venv all the same.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}

RUN pip install --no-cache-dir --upgrade pip \
    numpy importlib-metadata setuptools wheel ninja pytesseract "itsdangerous<2.1.0"

# gfx1151 torch stack, ROCm 7.2.1 manylinux wheels from repo.radeon.com (cp312,
# matching the base image's python 3.12). --find-links lets pip resolve torch's matching
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
