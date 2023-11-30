FROM rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_2.0.1
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH='2.1.0'
ARG TORCH_VISION='0.16.0'
ARG TORCH_AUDIO='2.1.0'
ARG ROCM='5.6'

RUN apt update && \
    apt install -y --no-install-recommends libaio-dev git && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN python3 -m pip uninstall -y apex

RUN python3 -m pip uninstall -y torch torchvision torchaudio

RUN python3 -m pip install torch==$PYTORCH torchvision==$TORCH_VISION torchaudio==$TORCH_AUDIO --index-url https://download.pytorch.org/whl/rocm$ROCM

ARG REF=main
WORKDIR /
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF
RUN python3 -m pip install --no-cache-dir ./transformers[deepspeed-testing]

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

RUN python3 -c "from deepspeed.launcher.runner import main"