FROM rocm/dev-ubuntu-22.04:6.2.4
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH='2.6.0'
ARG TORCH_VISION='0.21.0'
ARG TORCH_AUDIO='2.6.0'
ARG ROCM='6.2.4'

RUN apt update && \
    apt install -y --no-install-recommends \
    libaio-dev \
    git \
    curl \
    # These are required to build deepspeed.
    rocrand-dev \
    rocthrust-dev \
    rocblas-dev \
    hipsolver-dev \
    hipsparse-dev \
    hipblas-dev \
    hipblaslt-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# Python comes from uv, at the version in the repo's `.python-version` (passed as the `PYTHON_VERSION` build arg by the
# build workflow), instead of Ubuntu 22.04's 3.10.
ARG PYTHON_VERSION
ENV VIRTUAL_ENV="/opt/venv" UV_PYTHON_INSTALL_DIR="/opt/uv/python"
ENV PATH="$VIRTUAL_ENV/bin:/root/.local/bin:$PATH"
COPY --from=gpu-common install-python.sh /opt/ci/
RUN sh /opt/ci/install-python.sh "$PYTHON_VERSION"

RUN python3 -m pip install --no-cache-dir --upgrade pip ninja "pydantic>=2.0.0"
RUN python3 -m pip uninstall -y apex torch torchvision torchaudio
RUN python3 -m pip install torch==$PYTORCH torchvision==$TORCH_VISION torchaudio==$TORCH_AUDIO --index-url https://download.pytorch.org/whl/rocm$ROCM --no-cache-dir

# Pre-build DeepSpeed, so it's be ready for testing (to avoid timeout)
RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 python3 -m pip install deepspeed --no-build-isolation --config-settings="--build-option=build_ext" --config-settings="--build-option=-j8" --no-cache-dir -v --disable-pip-version-check 2>&1

# The build workflow passes the commit being built, so this clone (and every layer after it) is redone for each new
# commit instead of being reused from the cache.
ARG REF=main
WORKDIR /

RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

RUN python3 -m pip install --no-cache-dir ./transformers[accelerate,testing,sentencepiece,sklearn]

COPY --from=gpu-common finish-transformers-install.sh /opt/ci/
RUN sh /opt/ci/finish-transformers-install.sh

RUN python3 -c "from deepspeed.launcher.runner import main"

# Remove nvml as it is not compatible with ROCm
RUN python3 -m pip uninstall py3nvml pynvml nvidia-ml-py apex -y
