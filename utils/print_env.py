#!/usr/bin/env python3

# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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

# this script dumps information about the environment

import os
import sys

import transformers
from transformers import is_torch_hpu_available, is_torch_xpu_available


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print("Python version:", sys.version)
print("transformers version:", transformers.__version__)

try:
    import torch

    print("Torch version:", torch.__version__)
    accelerator = "NA"
    if torch.cuda.is_available():
        accelerator = "CUDA"
    elif is_torch_xpu_available():
        accelerator = "XPU"
    elif is_torch_hpu_available():
        accelerator = "HPU"

    print("Torch accelerator:", accelerator)

    if accelerator == "CUDA":
        print("Cuda version:", torch.version.cuda)
        print("CuDNN version:", torch.backends.cudnn.version())
        print("Number of GPUs available:", torch.cuda.device_count())
        print("NCCL version:", torch.cuda.nccl.version())
    elif accelerator == "XPU":
        print("SYCL version:", torch.version.xpu)
        print("Number of XPUs available:", torch.xpu.device_count())
    elif accelerator == "HPU":
        print("HPU version:", torch.__version__.split("+")[-1])
        print("Number of HPUs available:", torch.hpu.device_count())
except ImportError:
    print("Torch version:", None)

try:
    import deepspeed

    print("DeepSpeed version:", deepspeed.__version__)
except ImportError:
    print("DeepSpeed version:", None)


try:
    import torchcodec

    versions = torchcodec._core.get_ffmpeg_library_versions()
    print("FFmpeg version:", versions["ffmpeg_version"])
except ImportError:
    print("FFmpeg version:", None)
except (AttributeError, KeyError, RuntimeError):
    print("Failed to get FFmpeg version")
