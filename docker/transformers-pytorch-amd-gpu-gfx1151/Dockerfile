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
#  - torchcodec: audio/video decoding. Since 0.12 one release supports every
#    torch >= 2.11, so it need not move with torch. The CPU build, from
#    PyTorch's cpu index: PyPI's wheels are CUDA builds from 0.14 on (they link
#    libcudart/libnvrtc and fail to load here); 0.13 on PyPI happened to be
#    +cpu. --no-deps so it can't drag in a different torch. Best-effort, but
#    checked by decoding-library import: a wheel that installs and cannot load
#    is recorded as missing rather than shipped silently.
ARG TORCHCODEC_INDEX=https://download.pytorch.org/whl/cpu
RUN python3 -m pip install --no-cache-dir av
RUN { python3 -m pip install --no-cache-dir --no-deps --index-url ${TORCHCODEC_INDEX} \
        "torchcodec==0.17.0+cpu" \
      && python3 -c "from torchcodec.decoders import AudioDecoder"; } || \
    { echo "torchcodec unusable"; python3 -m pip uninstall -y torchcodec || true; \
      echo torchcodec >> /etc/hf-ci-optional-missing; }

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

# --- Hub kernels the ROCm 10.1 loader rejects by tag ---------------------------
# kernels 0.17.2 picks the backend tag from torch.version.hip, so on ROCm 10.1
# (HIP 7.16) it wants a "rocm716" build, and no Hub kernel publishes one: every
# compiled kernel fails with "Cannot find a build variant for this system", even
# where a rocm72 build exists and runs correctly here. LOCAL_KERNELS points the
# loader at a variant directory and skips that check (resolver.py at v0.17.2).
# Measured on gfx1151 for Atlas-Inference/gdn: qwen3_5 and qwen3_5_moe's
# kernel-load tests 0/4 -> 4/4, and the layer matches transformers' own forward
# to <1% relative L2. This section goes away once kernels matches ROCm builds by
# HIP major version (rocm72 and rocm101 are both HIP 7), or the Hub publishes
# builds for ROCm 10.1 with these GPUs in them.
#
# gdn is the only Hub build used as-is, and only on gfx1151, because only that
# pairing has a build with our code objects in it: transformers maps gdn to ROCm
# capability 115 at this revision, and its rocm72 build carries gfx1151 alone.
# megablocks' ROCm builds carry gfx942/gfx950 and activation's stop at gfx1101,
# so pointing the loader at them would only trade one error for another; those
# builds with the RDNA archs added, as megablocks has below.
# The revision is read from the transformers checked out at REF, because
# LOCAL_KERNELS overrides by repo name and ignores the revision transformers
# asks for: a revision written here goes stale silently when REF moves (it did,
# dff7b2f -> 13462bf). It takes the entry covering gfx1151's capability (115),
# so a later mapping for another ROCm GPU doesn't break this build. If
# transformers stops mapping gdn for gfx1151, the build fails here rather than
# shipping an override nothing uses.
# The ENV is set on every arch: Dockerfiles cannot set one conditionally, and on
# gfx1201 nothing asks for gdn, so the override is never consulted there.
RUN if [ "${GPU_ARCH}" = "gfx1151" ]; then \
      rev="$(python3 -c "import transformers.integrations.hub_kernels as h; \
r=[x for d, x in h._build_kernel_mapping()['Qwen3_5GatedDeltaNet'].items() if getattr(d, 'type', d) == 'rocm' \
   and getattr(getattr(d, 'properties', None), 'min_capability', 0) <= 115 <= getattr(getattr(d, 'properties', None), 'max_capability', 0)]; \
assert len(r) == 1, r; print(r[0]._revision)")" \
      && echo "gdn revision from transformers: ${rev}" \
      && python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Atlas-Inference/gdn', repo_type='kernel', revision='${rev}', allow_patterns=['build/torch214-cxx11-rocm72-x86_64-linux/*'], local_dir='/opt/hub-kernels/gdn')" \
      && test -f /opt/hub-kernels/gdn/build/torch214-cxx11-rocm72-x86_64-linux/metadata.json; \
    fi

# megablocks (gpt_oss) is a build AMD made: the Hub's ROCm builds carry
# gfx942/gfx950 only. Built with kernel-builder from kernels-community/megablocks
# 460047c with gfx1151 and gfx1201 added to build.toml's rocm-archs; the tar holds
# SOURCE-kernels-community-rev.txt and the build.toml diff, and the hash pins
# exactly that artefact. Measured on gfx1151: gpt_oss's two
# kernel tests 0/2 -> 2/2, nothing else changes. Unlike aiter it needs compiling,
# so it is downloaded, from HF_KERNELS_URL. That has no default on purpose: this
# file names no host of AMD's. Leave it empty (HF's builders) and the step is
# skipped: those two tests fail as they do upstream, and nothing else changes.
# The fix belongs on the Hub (gfx1151/gfx1201 in megablocks' rocm-archs).
ARG HF_KERNELS_URL=
ARG MEGABLOCKS_SHA256=83be38ac204da9a1e61b056a66f90e9b35c152dea1e6098af38d5a040e48a57c
RUN if [ -n "${HF_KERNELS_URL}" ]; then \
      curl -fsSL --retry 5 --retry-all-errors -o /tmp/megablocks.tar.gz \
        "${HF_KERNELS_URL}/megablocks/megablocks-torch214-rocm714-gfx1151-gfx1201.tar.gz" \
      && echo "${MEGABLOCKS_SHA256}  /tmp/megablocks.tar.gz" | sha256sum -c - \
      && mkdir -p /opt/hub-kernels && tar -xzf /tmp/megablocks.tar.gz -C /opt/hub-kernels \
      && rm /tmp/megablocks.tar.gz \
      && python3 -c "import json; a=json.load(open('/opt/hub-kernels/megablocks/torch214-rocm714-x86_64-linux/metadata.json'))['backend']['archs']; assert '${GPU_ARCH}' in a, a"; \
    else echo "HF_KERNELS_URL empty: no local megablocks build"; fi
# megablocks is named only when it was fetched: an entry for a missing directory
# would replace the Hub lookup with a failing local one.
ENV LOCAL_KERNELS=Atlas-Inference/gdn=/opt/hub-kernels/gdn/build/torch214-cxx11-rocm72-x86_64-linux${HF_KERNELS_URL:+:kernels-community/megablocks=/opt/hub-kernels/megablocks/torch214-rocm714-x86_64-linux}

# Last: urllib3 must still be 2.x (pinned on the kernels step above, against
# transformers[testing]'s <2) and kernels must still import. A later step that
# resolved deps would silently move urllib3 back to 1.26 and break kernels' tuf.
RUN python3 -c "import urllib3, kernels; assert urllib3.__version__.startswith('2.'), urllib3.__version__"
