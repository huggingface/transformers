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

import torch


print("Torch version:", torch.__version__)
print("Cuda available:", torch.cuda.is_available())
print("Cuda version:", torch.version.cuda)
print("CuDNN version:", torch.backends.cudnn.version())
print("Number of GPUs available:", torch.cuda.device_count())
print("NCCL version:", torch.cuda.nccl.version())
