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

# Python comes from uv, at the version in the repo's `.python-version` (passed as a build arg by the build
# workflow), instead of Ubuntu 22.04's 3.10. `python`, `python3` and `pip` all resolve to the venv.
ARG PYTHON_VERSION
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3 -m pip install --no-cache-dir --upgrade pip ninja "pydantic>=2.0.0"
RUN python3 -m pip uninstall -y apex torch torchvision torchaudio
RUN python3 -m pip install torch==$PYTORCH torchvision==$TORCH_VISION torchaudio==$TORCH_AUDIO --index-url https://download.pytorch.org/whl/rocm$ROCM --no-cache-dir

# Pre-build DeepSpeed, so it's be ready for testing (to avoid timeout)
RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 python3 -m pip install deepspeed --no-build-isolation --config-settings="--build-option=build_ext" --config-settings="--build-option=-j8" --no-cache-dir -v --disable-pip-version-check 2>&1

ARG REF=main
WORKDIR /

RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

RUN python3 -m pip install --no-cache-dir ./transformers[accelerate,testing,sentencepiece,sklearn]

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

RUN python3 -c "from deepspeed.launcher.runner import main"

# Remove nvml as it is not compatible with ROCm
RUN python3 -m pip uninstall py3nvml pynvml nvidia-ml-py apex -y

# `kernels` may causes many failing tests
RUN python3 -m pip uninstall -y kernels
