FROM rocm/dev-ubuntu-22.04:5.6
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH='2.1.1'
ARG TORCH_VISION='0.16.1'
ARG TORCH_AUDIO='2.1.1'
ARG ROCM='5.6'

RUN apt update && \
    apt install -y --no-install-recommends \
    libaio-dev \
    git \
    # These are required to build deepspeed.
    python3-dev \
    python-is-python3 \
    rocrand-dev \
    rocthrust-dev \
    hipsparse-dev \
    hipblas-dev \
    rocblas-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip ninja "pydantic<2"
RUN python3 -m pip uninstall -y apex torch torchvision torchaudio
RUN python3 -m pip install torch==$PYTORCH torchvision==$TORCH_VISION torchaudio==$TORCH_AUDIO --index-url https://download.pytorch.org/whl/rocm$ROCM --no-cache-dir

# Pre-build DeepSpeed, so it's be ready for testing (to avoid timeout)
RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 python3 -m pip install deepspeed --global-option="build_ext" --global-option="-j8" --no-cache-dir -v --disable-pip-version-check 2>&1

ARG REF=main
WORKDIR /

# Invalidate docker cache from here if new commit is available.
ADD https://api.github.com/repos/huggingface/transformers/git/refs/heads/main version.json
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

RUN python3 -m pip install --no-cache-dir ./transformers[accelerate,testing,sentencepiece,sklearn]

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

RUN python3 -c "from deepspeed.launcher.runner import main"

# Remove nvml as it is not compatible with ROCm
RUN python3 -m pip uninstall py3nvml pynvml -y
