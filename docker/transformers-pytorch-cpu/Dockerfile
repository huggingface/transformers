# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

ARG IMAGE_NAME=ubuntu
ARG IMAGE_TAG=22.04
FROM ${IMAGE_NAME}:${IMAGE_TAG}

LABEL maintainer="Hugging Face"
LABEL repository="transformers"

ARG PY_VER=3.10

RUN apt-get update && \
    apt-get install --no-install-recommends --fix-missing -y \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists

RUN ln -sf $( which python${PY_VER} ) /usr/bin/python && \
    ln -sf $( which python${PY_VER} ) /usr/bin/python3

ARG TORCH_CPU_URL=https://download.pytorch.org/whl/cpu/torch_stable.html
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
    jupyter \
    torch -f ${TORCH_CPU_URL}

WORKDIR /workspace
COPY . transformers/
RUN cd transformers/ && \
    python -m pip install --no-cache-dir .

CMD ["/bin/bash"]
