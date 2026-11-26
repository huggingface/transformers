# transformers CI image for AMD Ryzen AI Max "Strix Halo" (gfx1151) and Radeon AI Pro R9700 (gfx1201).
# No ROCm base image: AMD's torch wheels bring their own ROCm userspace. The host provides the amdgpu driver;
# run with --device=/dev/kfd --device=/dev/dri.
FROM ubuntu:24.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

# libatomic1, libnuma1, libgomp1, libdrm2, libdrm-amdgpu1 and libelf1 are loaded by the torch wheel but not
# declared by it; without libnuma1 the first GPU op segfaults.
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

# Ubuntu 24.04's system python is externally managed (PEP 668), so install in a venv.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}

RUN pip install --no-cache-dir --upgrade pip \
    numpy importlib-metadata setuptools wheel ninja pytesseract "itsdangerous<2.1.0"

# AMD's ROCm wheels. torch lives in the pytorch tree and the rocm-sdk packages in the core tree, so both indexes
# are needed. The [device-<arch>] extra installs device code for one GPU only.
ARG GPU_ARCH=gfx1151
ARG TORCH_INDEX=https://stable.repo.amd.com/rocm/pytorch/whl-next/
ARG TORCH_EXTRA_INDEX=https://stable.repo.amd.com/rocm/core/whl-next/
ARG TORCH_VERSION=2.15.0+rocm10.2.0
ARG TORCHVISION_VERSION=0.29.2+rocm10.2.0
ARG TORCHAUDIO_VERSION=2.11.1+rocm10.2.0
RUN python3 -m pip install --no-cache-dir --index-url ${TORCH_INDEX} --extra-index-url ${TORCH_EXTRA_INDEX} \
      "torch[device-${GPU_ARCH}]==${TORCH_VERSION}" \
      "torchvision[device-${GPU_ARCH}]==${TORCHVISION_VERSION}"
# PyPI's torchaudio is a CUDA build, so use AMD's. --no-deps so it cannot pull a different torch.
RUN python3 -m pip install --no-cache-dir --no-deps --index-url ${TORCH_INDEX} "torchaudio==${TORCHAUDIO_VERSION}"

# Keep later installs (e.g. timm) from replacing the ROCm torch with a CUDA wheel from PyPI.
RUN printf 'torch==%s\ntorchvision==%s\n' "${TORCH_VERSION}" "${TORCHVISION_VERSION}" > /opt/rocm-constraints.txt
ENV PIP_CONSTRAINT=/opt/rocm-constraints.txt

ARG REF=main
WORKDIR /

RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

# The [kernels] extra installs `kernels` in the range transformers supports; outside it the Hub-kernel tests
# test nothing.
RUN python3 -m pip install --no-cache-dir -e "./transformers[testing,kernels]"

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

# Remove nvml and nvidia-ml-py as it is not compatible with ROCm. apex is not tested on NVIDIA either.
RUN python3 -m pip uninstall -y py3nvml pynvml nvidia-ml-py apex || true

# pytest-rerunfailures before 16.7 hangs the run when an xdist worker dies (pytest-rerunfailures#358).
RUN python3 -m pip install --no-cache-dir --no-deps "pytest-rerunfailures==16.7"

# Test deps outside [testing]. Missing scipy/librosa/soundfile shows up as NameErrors, not ImportErrors.
RUN python3 -m pip install --no-cache-dir einops blobfile num2words timm \
      scipy librosa soundfile pycocotools peft bitsandbytes matplotlib

# torchcodec decodes audio, video and images: AMD's torchvision is built without image codecs, and transformers
# uses torchcodec first. The CPU build, because PyPI's wheels are CUDA builds. Optional: if it cannot decode,
# it is removed and listed in /etc/hf-ci-optional-missing.
RUN python3 -m pip install --no-cache-dir av
RUN { python3 -m pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/cpu \
        "torchcodec==0.17.0+cpu" \
      && python3 -c "import io, torch; from PIL import Image; \
           from torchcodec.decoders import AudioDecoder, decode_image; \
           b = io.BytesIO(); Image.new('RGB', (2, 2), (255, 0, 0)).save(b, 'PNG'); \
           o = decode_image(torch.frombuffer(bytearray(b.getvalue()), dtype=torch.uint8)); \
           assert o[:, 0, 0].tolist() == [255, 0, 0], o"; } || \
    { echo "torchcodec unusable"; python3 -m pip uninstall -y torchcodec || true; \
      echo torchcodec >> /etc/hf-ci-optional-missing; }

# The [audio] extra's packages. --no-deps because pyctcdecode pins numpy<2; its other deps are listed by hand.
RUN python3 -m pip install --no-cache-dir --no-deps phonemizer==3.4.0 \
      dlinfo==2.0.0 pyctcdecode==0.5.0 pygtrie==2.6.2 \
 && python3 -c "import numpy as np; from pyctcdecode import build_ctcdecoder; \
      from phonemizer.backend import EspeakBackend; \
      assert EspeakBackend.is_available(), 'espeak-ng missing'; \
      build_ctcdecoder(['', 'a', 'b']).decode(np.log(np.full((6, 3), 1/3, 'f4')))"
# kenlm compiles from source; optional like torchcodec.
RUN python3 -m pip install --no-cache-dir --no-deps kenlm==0.3.0 || \
    { echo "kenlm install skipped"; echo kenlm >> /etc/hf-ci-optional-missing; }

# Pre-fetch test fixtures, as docker/torch-light.dockerfile does, so jobs don't hit Hub rate limits.
# HF_TOKEN is optional: docker build --secret id=hf_token,env=HF_TOKEN ...
RUN --mount=type=secret,id=hf_token \
    cd /transformers && \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
      python3 utils/fetch_hub_objects_for_ci.py

# Fail the build, not the tests, if the torch stack or the pins above were replaced.
RUN python3 -c "import torch; v=torch.__version__; assert 'rocm' in v, \
      f'torch is {v} -- a non-ROCm build replaced it'; print('verified torch:', v)"
RUN python3 -c "import torch, torchaudio, torchaudio.transforms as T; \
      T.MelSpectrogram(16000)(torch.randn(1,16000)); \
      print('torchaudio ok:', torchaudio.__version__)"
RUN python3 -c "from transformers.utils import is_kernels_available; assert is_kernels_available(), 'kernels missing or outside transformers range'; \
from importlib.metadata import version as v; assert v('pytest-rerunfailures') == '16.7', v('pytest-rerunfailures')"

# Deterministic MIOpen conv kernel selection, otherwise bf16 outputs can change between runners (as in the MI300 image).
ENV MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
ENV MIOPEN_FIND_MODE=2

RUN python3 -m pip freeze --all > /opt/pip-freeze.txt
