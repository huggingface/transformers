FROM intel/deep-learning-essentials:2025.1.3-0-devel-ubuntu22.04 AS base
LABEL maintainer="Hugging Face"

SHELL ["/bin/bash", "-c"]

ARG PYTHON_VER=3.12
ENV TORCH_DEVICE_BACKEND_AUTOLOAD=0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python$PYTHON_VER python$PYTHON_VER-dev python3-pip && \
    ln -sf /usr/bin/python$PYTHON_VER /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get -y install \
        apt-utils \
        build-essential \
        ca-certificates \
        curl \
        git \
        git-lfs \
        vim \
        numactl \
        gnupg2 \
        gpg-agent \
        zlib1g-dev \
        rsync \
        sudo \
        libnl-genl-3-200 \
        unzip \
        ffmpeg \
        tesseract-ocr \
        espeak-ng \
        wget \
        ncurses-term && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y \
        linux-headers-$(uname -r) linux-modules-extra-$(uname -r) \
        flex bison \
        intel-fw-gpu intel-i915-dkms xpu-smi intel-ocloc clinfo\
        intel-opencl-icd libze-intel-gpu1 libze1 \
        intel-media-va-driver-non-free libmfx-gen1 libvpl2 \
        libegl-mesa0 libegl1 libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
        libglapi-mesa libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
        mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo \
        libigc-dev intel-igc-cm libigdfcl-dev libigfxcmrt-dev libze-dev && \
    apt-get clean && \
    rm -rf  /var/lib/apt/lists/*

# Use virtual env because Ubuntu-24 does not allowed pip on original python
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv --python ${PYTHON_VER} --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip wheel
RUN pip install triton==3.4.0

RUN pip install torch==2.8.0+xpu torchvision==0.23.0+xpu torchaudio==2.8.0+xpu --index-url https://download.pytorch.org/whl/xpu --no-cache-dir

RUN pip install torchcodec torchdata --no-cache-dir

RUN pip install evaluate pyctcdecode pytesseract decord galore-torch fire scipy scikit-learn sentencepiece sacremoses nltk rouge_score librosa soundfile g2p_en mpi4py requests_mock
RUN pip install pretty_midi essentia resampy Levenshtein av sacrebleu phonemizer invisible_watermark schedulefree setuptools
RUN pip install gptqmodel --no-build-isolation
RUN pip install gguf hqq compressed_tensors deepspeed torchao onnx auto_round
RUN pip install hf_transfer huggingface-hub hf-doc-builder datasets optimum-quanto timm transformers accelerate optimum peft diffusers trl kernels

# install liger-kernel
RUN pip install git+https://github.com/linkedin/Liger-Kernel.git --extra-index-url https://download.pytorch.org/whl/test/xpu

# install mergekit
RUN pip install --break-system-packages git+https://github.com/arcee-ai/mergekit.git@v0.1.3

# install bitsandbytes
RUN pip install git+https://github.com/bitsandbytes-foundation/bitsandbytes.git

ENV OCL_ICD_VENDORS=/etc/OpenCL/vendors
ENV FI_PROVIDER_PATH=${I_MPI_ROOT}/lib/libfabric/prov:/usr/lib/x86_64-linux-gnu/libfabric
ENV CCL_ROOT=/usr/local
ENV CCL_ATL_TRANSPORT=ofi
ENV I_MPI_ROOT=/usr/local
ENV CLASSPATH=${I_MPI_ROOT}/lib/mpi.jar
ENV PATH=${I_MPI_ROOT}/bin/libfabric:${PATH}
ENV LD_LIBRARY_PATH=${I_MPI_ROOT}/lib/libfabric:${LD_LIBRARY_PATH}

RUN touch /entrypoint.sh
RUN chmod +x /entrypoint.sh
RUN echo "#!/bin/bash" >> /entrypoint.sh
RUN echo "source /opt/intel/oneapi/setvars.sh --force && /bin/bash" >> /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
