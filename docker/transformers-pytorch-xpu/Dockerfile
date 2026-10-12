FROM intel/deep-learning-essentials:2026.1.2-devel-ubuntu24.04 AS base
LABEL maintainer="Hugging Face"

SHELL ["/bin/bash", "-c"]

ARG PYTHON_VER=3.12
ENV TORCH_DEVICE_BACKEND_AUTOLOAD=0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python$PYTHON_VER python$PYTHON_VER-dev && \
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
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics && \
    apt-get install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc && \
    apt-get install -y intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo && \
    apt-get install -y libze-dev intel-ocloc && \
    apt-get install -y libze-intel-gpu-raytracing &&\
    apt-get clean && \
    rm -rf  /var/lib/apt/lists/*

# Use virtual env because Ubuntu-24 does not allowed pip on original python
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv --python ${PYTHON_VER} --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN uv pip install --upgrade pip wheel
RUN uv pip install triton==3.8.0

RUN uv pip install torch==2.14.0+xpu torchvision==0.29.0+xpu torchaudio==2.11.0+xpu torchao==0.18.0+xpu --index-url https://download.pytorch.org/whl/xpu --no-cache-dir

RUN uv pip install torchcodec==0.16.0 --index-url https://download.pytorch.org/whl/cpu

RUN uv pip install torchdata --no-cache-dir

RUN uv pip install evaluate pyctcdecode pytesseract decord fire scipy scikit-learn sentencepiece sacremoses nltk rouge_score librosa soundfile g2p_en mpi4py requests_mock onnx deepspeed kenlm opencv-python albumentations --no-cache-dir
RUN uv pip install pretty_midi essentia resampy Levenshtein av sacrebleu phonemizer invisible_watermark schedulefree parameterized mistral-common --no-cache-dir

RUN uv pip install gguf hqq fp-quant compressed_tensors bitsandbytes --no-cache-dir --extra-index-url https://download.pytorch.org/whl/xpu "torch==2.14.0+xpu" "triton==3.8.0"
RUN uv pip install gptqmodel --no-cache-dir
RUN uv pip install auto-round --no-cache-dir
RUN uv pip install hf_transfer optimum-quanto optimum hf-doc-builder transformers[dev] diffusers trl sentence_transformers pytorch_msssim --no-cache-dir
RUN uv pip install pydantic openai uvicorn fastapi --no-cache-dir

# install liger-kernel
RUN uv pip install git+https://github.com/linkedin/Liger-Kernel.git --extra-index-url https://download.pytorch.org/whl/xpu

# install mergekit
RUN uv pip install --no-deps git+https://github.com/arcee-ai/mergekit.git

RUN uv pip install onnxscript onnxruntime --no-cache-dir

ENV OCL_ICD_VENDORS=/etc/OpenCL/vendors
ENV CCL_ATL_TRANSPORT=ofi
ENV CLASSPATH=/opt/venv/share/java/mpi.jar
ENV LD_LIBRARY_PATH=/opt/venv/lib
