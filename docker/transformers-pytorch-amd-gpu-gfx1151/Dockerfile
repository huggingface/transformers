# transformers CI image for AMD Ryzen AI Max "Strix Halo" (gfx1151) and Radeon
# AI Pro R9700 (gfx1201); GPU_ARCH below picks the GPU.
#
# Base is plain Ubuntu 24.04 (system python 3.12, matching the cp312 wheels
# below). No ROCm in the base: the torch wheels (AMD's ROCm 10.1 build by
# default) pull their own ROCm userspace from pip (rocm-sdk-core /
# rocm-sdk-libraries), so a ROCm base image would only add ~20 GB of unused
# libraries. The host still supplies the amdgpu/KFD kernel driver; run with
# --device=/dev/kfd --device=/dev/dri (see the CI workflow).
#
# The ROCm userspace does NOT need to match the pool's driver. This image runs
# on the pool as-is (inbox amdgpu in the 6.17 OEM kernel, no amdgpu-dkms).
# `hipErrorInvalidImage` means the wheels lack device code for this GPU, not
# driver skew.
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

# The GPU this image runs on (gfx1151 Strix Halo, gfx1201 Radeon AI Pro R9700).
# Declared here because the torch stack below picks its device code from it.
ARG GPU_ARCH=gfx1151

# torch stack: AMD's ROCm 10.1 wheels (cp312, matching the base image's python
# 3.12), the stack this file was validated on for gfx1151 and gfx1201 (full
# fast and slow suites, 2026-10-26). AMD splits its index into product trees,
# torch in one and rocm-sdk-* in another, so it needs an extra index, and its
# torch pulls the GPU's device code through an extra such as [device-gfx1201]
# rather than requiring rocm[device-all].
#
# PyTorch's own ROCm 7.14 channel also works but was last validated before the
# kernel and pin changes in this file: build it with
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/rocm7.14
#   --build-arg TORCH_EXTRA_INDEX= --build-arg TORCH_DEVICE_EXTRA=
#   --build-arg TORCH_VERSION=2.14.0+rocm7.14 --build-arg TORCHVISION_VERSION=0.29.0+rocm7.14
#   --build-arg TORCHAUDIO_INDEX=https://repo.amd.com/rocm/whl-multi-arch/
#   --build-arg TORCHAUDIO_VERSION=2.11.0+rocm7.14.1
# On that index, do not move to torchaudio 2.11.0.2 without testing: its GPU
# kernels do not include gfx1151.
ARG TORCH_INDEX=https://stable.repo.amd.com/rocm/pytorch/whl-next/
ARG TORCH_EXTRA_INDEX=https://stable.repo.amd.com/rocm/core/whl-next/
ARG TORCH_DEVICE_EXTRA=[device-${GPU_ARCH}]
ARG TORCH_VERSION=2.14.0+rocm10.1.0
ARG TORCHVISION_VERSION=0.29.0a0+rocm10.1.0

# --- Keep only one GPU architecture (saves ~10 GB) ---------------------------
# torch requires rocm[device-all], which installs ROCm libraries for all 25
# supported architectures (torch itself has kernels for 13 of them). When
# TRIM_OTHER_ARCHS=1, the install step below deletes the files of every
# rocm-sdk-device-* package except GPU_ARCH. Their metadata is kept, so later
# pip calls still see torch's requirements as met. This has to happen in the
# same RUN as the install: deleting in a later RUN would not shrink the image.
#   GPU_ARCH           (above) must match the GPU this image runs on.
#   TRIM_OTHER_ARCHS   1 = keep only GPU_ARCH (default); 0 = keep all 25.
#                      Any other value fails the build.
ARG TRIM_OTHER_ARCHS=1
RUN python3 -m pip install --no-cache-dir --index-url ${TORCH_INDEX} \
      ${TORCH_EXTRA_INDEX:+--extra-index-url ${TORCH_EXTRA_INDEX}} \
      "torch${TORCH_DEVICE_EXTRA}==${TORCH_VERSION}" \
      "torchvision${TORCH_DEVICE_EXTRA}==${TORCHVISION_VERSION}" && \
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
# build, from the same tree as torch. --no-deps so it cannot pull a different
# torch.
ARG TORCHAUDIO_INDEX=https://stable.repo.amd.com/rocm/pytorch/whl-next/
ARG TORCHAUDIO_VERSION=2.11.0+rocm10.1.0
RUN python3 -m pip install --no-cache-dir --no-deps --index-url ${TORCHAUDIO_INDEX} \
      "torchaudio==${TORCHAUDIO_VERSION}"

# Pin torch/torchvision for every later pip call. Without it, a package that
# depends on torch (e.g. timm) can let pip replace the ROCm build with a CUDA
# wheel from PyPI: the image still builds, and the tests silently run on CPU.
RUN printf 'torch==%s\ntorchvision==%s\n' "${TORCH_VERSION}" "${TORCHVISION_VERSION}" > /opt/rocm-constraints.txt
ENV PIP_CONSTRAINT=/opt/rocm-constraints.txt

# torchvision image codecs. PyTorch's rocm7.14 channel ships them; AMD's ROCm
# 10.x torchvision (a TheRock build) is built without libPNG, so decode_png
# raises "torchvision not compiled with libPNG support" and every fast image
# processor test fails (transformers has no PIL fallback there). The codecs are
# their own extension, separate from the ops and the device package (the GPU
# kernels), so building just that one against the installed torch and dropping
# it in leaves the GPU code untouched. Built from the commit the wheel records (an a0
# version has no release tag), and only when a real PNG fails to decode, so
# the PyTorch-channel build is unchanged.
#
# Only the image extension is built: setup.py's list is cut to make_image*.
# Building _C as well fails to link against ROCm torch (hipify adds a
# vision_hip.cpp beside vision.cpp, both defining cuda_version), and it is
# not needed. The extension was image.so up to 0.28 and is image_stable.so in
# 0.29, hence the glob. TV_DIR is resolved before cd: inside the checkout,
# `import torchvision` finds the unbuilt source tree, not the installed one.
RUN probe='import torch, struct, zlib; from torchvision.io import decode_image, ImageReadMode; \
c=lambda t,d: struct.pack(">I",len(d))+t+d+struct.pack(">I",zlib.crc32(t+d)&0xffffffff); \
png=b"\x89PNG\r\n\x1a\n"+c(b"IHDR",struct.pack(">IIBBBBB",1,1,8,2,0,0,0))+c(b"IDAT",zlib.compress(b"\x00\xff\x00\x00"))+c(b"IEND",b""); \
o=decode_image(torch.frombuffer(bytearray(png),dtype=torch.uint8),mode=ImageReadMode.RGB); \
assert o.flatten().tolist()==[255,0,0], o; print("torchvision libPNG ok")' && \
    if python3 -c "$probe" 2>/dev/null; then \
      echo "torchvision already decodes PNG; not rebuilding its image extension"; \
    else \
      TV_SHA="$(python3 -c 'import torchvision.version as v; print(v.git_version)')" && \
      TV_VER="$(python3 -c 'import torchvision; print(torchvision.__version__.split("+")[0])')" && \
      TV_DIR="$(python3 -c 'import torchvision, os; print(os.path.dirname(torchvision.__file__))')" && \
      echo "rebuilding the torchvision image extension from ${TV_SHA} (${TV_VER})" && \
      apt-get update && apt-get install -y --no-install-recommends \
        libpng-dev libjpeg-turbo8-dev zlib1g-dev && \
      git init -q /tmp/vision && cd /tmp/vision && \
      git fetch -q --depth 1 https://github.com/pytorch/vision.git "${TV_SHA}" && \
      git checkout -q FETCH_HEAD && \
      python3 -c "import re; s=open('setup.py').read(); \
m=re.search(r'extensions = \[(.*?)\]', s, re.S); assert m, 'no extensions list'; \
keep=[e for e in m.group(1).split(',') if 'make_image' in e]; assert len(keep)==1, keep; \
open('setup.py','w').write(s[:m.start(1)]+keep[0]+','+s[m.end(1):])" && \
      TORCHVISION_USE_PNG=1 TORCHVISION_USE_JPEG=1 \
      TORCHVISION_USE_NVJPEG=0 TORCHVISION_USE_VIDEO_CODEC=0 \
      FORCE_CUDA=0 USE_ROCM=0 MAX_JOBS=16 BUILD_VERSION="${TV_VER}" \
        python3 setup.py build_ext --inplace && \
      so="$(ls torchvision/image*.so)" && [ "$(echo "$so" | wc -l)" = 1 ] && \
      cp "$so" "${TV_DIR}/" && \
      cd / && rm -rf /tmp/vision && \
      apt-get clean && rm -rf /var/lib/apt/lists/* && \
      python3 -c "$probe"; \
    fi

# Flash-Attention wheel: off by default. Without the flash_attn package,
# transformers serves attn_implementation="flash_attention_2" from the
# kernels-community/aiter-flash-attn hub kernel (through the `kernels` package)
# ("You do not have `flash_attn` installed, using ... aiter-flash-attn"). On
# Llama-3.1-8B fp16 it matched SDPA's output, and its decode and 4k-token prefill
# speed, within ~2% on gfx1151 and gfx1201 (probe 2026-10-27).
# To use AMD's Triton wheel instead (gfx1151 only; the nightlies have no gfx1201
# build), pass FA_WHEEL=https://rocm.frameworks-nightlies.amd.com/whl/gfx1151/flash_attn-2.8.3-py3-none-any.whl
# --no-deps so it cannot pull a different torch; it is pinned to a torch/ROCm
# ABI, so check it against the stack above before using it.
ARG FA_WHEEL=
RUN if [ -n "${FA_WHEEL}" ]; then \
      python3 -m pip install --no-cache-dir --no-deps "${FA_WHEEL}"; \
    else echo "FA_WHEEL empty: no flash-attn in this image"; fi

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

# `kernels` is pinned in below, not uninstalled; see the note there.

# Common extra deps used across the test suite. scipy, librosa and soundfile
# are easy to miss: transformers' lazy imports turn their absence into errors
# like `NameError: name 'EncodecFeatureExtractor' is not defined`, not
# ImportError. pycocotools (DETR-family image processors) and peft (adapter
# tests, e.g. jina_embeddings_v3) are not in the [testing] extra.
#
# bitsandbytes: its absence is a *skip*, not a failure -- `require_bitsandbytes`
# gates the test rather than the import -- so it costs coverage silently. The
# PyPI wheel has no CUDA in its tag, and at RUN_SLOW=1 across ministral,
# muse_glimmer_assistant and shieldgemma2 not one bitsandbytes test skipped or
# failed on gfx1151.
#
# NOT optimum-quanto, the obvious companion: it compiles a HIP extension at
# first use and the build fails on this stack (`ninja: build stopped` from
# quanto_hip), costing a failure on ministral3 for one test gained on
# shieldgemma2.
#
# matplotlib: auto's test_custom_tokenizer_init loads a remote-code tokenizer
# that imports it, so without it that slow test fails with an ImportError naming
# matplotlib, on every GPU.
#
# kernels, pinned to transformers' range (setup.py: kernels>=0.17.0,<0.18).
# It used to be uninstalled: adding it took gpt_oss from 5 failures to 25, all
# resolving the CUDA-only kernels-community/vllm-flash-attn3. Retested on ROCm
# 10.1 with kernels 0.17.2: the kernel tests in ministral3, granite_swa and
# granitemoe_swa pass, and ministral3's fp8 slow tests now load
# finegrained-fp8's torch-rocm build and run. Compiled kernels still fail to
# load on ROCm 10.1; see "Hub kernels" at the end of this file.
# urllib3 is pinned here, not globally, because two requirements conflict:
# transformers[testing] (its "quality" deps) wants urllib3<2.0.0, while kernels
# 0.17.2 -> sigstore -> tuf wants urllib3~=2.0. Unpinned, the transformers
# install takes 1.26.x and this step upgrades it. A global constraint would
# apply to the transformers step too and make it unresolvable.
RUN python3 -m pip install --no-cache-dir einops blobfile num2words timm \
      scipy librosa soundfile pycocotools peft bitsandbytes matplotlib \
      "kernels==0.17.2" "urllib3==2.8.0"

# Media decoders needed by audio/video model tests:
#  - PyAV (`av`): torchvision video ops
#  - torchcodec: audio/video decoding (pinned to the torch 2.14 ABI; --no-deps
#    so it can't drag in a different torch). Best-effort: a torchcodec mismatch
#    must not fail the whole image build.
RUN python3 -m pip install --no-cache-dir av
RUN python3 -m pip install --no-cache-dir --no-deps torchcodec==0.13.0 || \
    python3 -m pip install --no-cache-dir --no-deps torchcodec || \
    echo "torchcodec install skipped"

# transformers' [audio] extra, which upstream's AMD image installs and [testing]
# does not: phonemizer (wav2vec2_phoneme, and one wav2vec2 tokenizer test that
# fails with ImportError rather than skipping), pyctcdecode and kenlm
# (wav2vec2_with_lm). espeak-ng, phonemizer's backend, is already in the base.
# --no-deps because pyctcdecode 0.5.0 pins numpy<2: resolved normally, it
# downgrades numpy to 1.26 under scipy/librosa/opencv and conftest.py no longer
# imports. Its other deps are listed by hand, and everything is pinned, since
# --no-deps cannot notice a new release growing a dep (phonemizer 3.4.0 needs
# attrs/joblib/typing-extensions, already in the base; older releases also
# needed segments). Not hypothesis, though pyctcdecode lists it: only its own
# tests use it, and it is a pytest plugin that would load in every session.
# The check decodes, not just imports, so a numpy-2 break in pyctcdecode fails
# the build rather than looking like a wav2vec2_with_lm test failure.
RUN python3 -m pip install --no-cache-dir --no-deps phonemizer==3.4.0 \
      dlinfo==2.0.0 pyctcdecode==0.5.0 pygtrie==2.6.2 \
 && python3 -c "import numpy as np; from pyctcdecode import build_ctcdecoder; \
      from phonemizer.backend import EspeakBackend; \
      assert EspeakBackend.is_available(), 'espeak-ng missing'; \
      build_ctcdecoder(['', 'a', 'b']).decode(np.log(np.full((6, 3), 1/3, 'f4')))"
# kenlm compiles from source: best-effort, like torchcodec, so a broken
# compile costs wav2vec2_with_lm's LM tests rather than every image build. The
# marker keeps a skip visible where results are read, so missing-kenlm
# failures aren't taken for test problems.
RUN python3 -m pip install --no-cache-dir --no-deps kenlm==0.3.0 || \
    { echo "kenlm install skipped"; echo kenlm >> /etc/hf-ci-optional-missing; }

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

# Last: urllib3 must still be 2.x (pinned on the kernels step above, against
# transformers[testing]'s <2) and kernels must still import. A later step that
# resolved deps would silently move urllib3 back to 1.26 and break kernels' tuf.
RUN python3 -c "import urllib3, kernels; assert urllib3.__version__.startswith('2.'), urllib3.__version__"
