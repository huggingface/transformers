FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

# Use login shell to read variables from `~/.profile` (to pass dynamic created variables between RUN commands)
SHELL ["sh", "-lc"]

# The following `ARG` are mainly used to specify the versions explicitly & directly in this docker file, and not meant
# to be used as arguments for docker build (so far).

ARG PYTORCH='2.6.0'
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu121'
# Disable kernel mapping for quantization tests
ENV DISABLE_KERNEL_MAPPING=1

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; echo "export VERSION='$VERSION'" >> ~/.profile
RUN echo torch=$VERSION
# `torchvision` and `torchaudio` should be installed along with `torch`, especially for nightly build.
# Currently, let's just use their latest releases (when `torch` is installed with a release version)
RUN python3 -m pip install --no-cache-dir -U $VERSION torchvision torchaudio torchcodec --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/accelerate@main#egg=accelerate

# needed in bnb and awq
RUN python3 -m pip install --no-cache-dir einops

# Add bitsandbytes for mixed int8 testing
RUN python3 -m pip install --no-cache-dir bitsandbytes

# Add gptqmodel for gtpq quantization testing, installed from source for pytorch==2.6.0 compatibility
RUN python3 -m pip install lm_eval
RUN git clone https://github.com/ModelCloud/GPTQModel.git && cd GPTQModel && pip install -v . --no-build-isolation

# Add optimum for gptq quantization testing
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/optimum@main#egg=optimum

# Add PEFT
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/peft@main#egg=peft

# Add aqlm for quantization testing
RUN python3 -m pip install --no-cache-dir aqlm[gpu]==1.0.2

# Add vptq for quantization testing
RUN pip install vptq

# Add spqr for quantization testing
# Commented for now as No matching distribution found we need to reach out to the authors
# RUN python3 -m pip install --no-cache-dir spqr_quant[gpu]

# Add hqq for quantization testing
RUN python3 -m pip install --no-cache-dir hqq

# For GGUF tests
RUN python3 -m pip install --no-cache-dir gguf

# Add autoawq for quantization testing
# New release v0.2.8
RUN python3 -m pip install --no-cache-dir autoawq[kernels]

# Add quanto for quantization testing
RUN python3 -m pip install --no-cache-dir optimum-quanto

# Add eetq for quantization testing
RUN git clone https://github.com/NetEase-FuXi/EETQ.git && cd EETQ/ && git submodule update --init --recursive && pip install .

# # Add flute-kernel and fast_hadamard_transform for quantization testing
# # Commented for now as they cause issues with the build
# # TODO: create a new workflow to test them
# RUN python3 -m pip install --no-cache-dir flute-kernel==0.4.1
# RUN python3 -m pip install --no-cache-dir git+https://github.com/Dao-AILab/fast-hadamard-transform.git

# Add fp-quant for quantization testing
# Requires py3.11 but our CI runs on 3.9
# RUN python3 -m pip install --no-cache-dir "fp-quant>=0.1.6"

# Add compressed-tensors for quantization testing
RUN python3 -m pip install --no-cache-dir compressed-tensors

# Add AMD Quark for quantization testing
RUN python3 -m pip install --no-cache-dir amd-quark

# Add AutoRound for quantization testing
RUN python3 -m pip install --no-cache-dir "auto-round>=0.5.0"

# Add transformers in editable mode
RUN python3 -m pip install --no-cache-dir -e ./transformers[dev-torch]

# `kernels` may give different outputs (within 1e-5 range) even with the same model (weights) and the same inputs
RUN python3 -m pip uninstall -y kernels

# Uninstall flash-attn installed by autoawq, it causes issues here : https://github.com/huggingface/transformers/actions/runs/15915442841/job/44892146131
RUN python3 -m pip uninstall -y flash-attn

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop
