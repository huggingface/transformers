FROM rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH='2.0.1'
ARG ROCM='5.7'

RUN apt update && \
    apt install -y --no-install-recommends libaio-dev git && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN python3 -m pip uninstall -y apex

ARG REF=main
WORKDIR /
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF
RUN python3 -m pip install --no-cache-dir ./transformers[deepspeed-testing]

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

RUN python3 -c "from deepspeed.launcher.runner import main"