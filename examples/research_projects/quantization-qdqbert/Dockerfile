# coding=utf-8
# Copyright 2021 NVIDIA Corporation. All rights reserved.
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
FROM nvcr.io/nvidia/pytorch:22.02-py3
LABEL maintainer="Hugging Face"
LABEL repository="transformers"

RUN apt-get update
RUN apt-get install sudo

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir --ignore-installed pycuda
RUN python3 -m pip install --no-cache-dir \
    pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
RUN python3 -m pip install --no-cache-dir onnxruntime-gpu==1.11

WORKDIR /workspace
COPY . transformers/
RUN cd transformers/ && \
    python3 -m pip install --no-cache-dir .

RUN python3 -m pip install --no-cache-dir datasets \
    accelerate
