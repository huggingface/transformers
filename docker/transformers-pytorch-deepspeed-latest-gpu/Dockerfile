FROM nvcr.io/nvidia/pytorch:21.03-py3
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt -y update
RUN apt install -y libaio-dev
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF
RUN python3 -m pip install --no-cache-dir -e ./transformers[deepspeed-testing]

RUN git clone https://github.com/microsoft/DeepSpeed && cd DeepSpeed && rm -rf build && \
    DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 python3 -m pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

RUN python3 -c "from deepspeed.launcher.runner import main"
