# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import torch
from packaging import version

from .utils import logging


logger = logging.get_logger(__name__)


def torch_int_div(tensor1, tensor2):
    """
    A function that performs integer division across different versions of PyTorch.
    """
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        return tensor1 // tensor2
    else:
        return torch.div(tensor1, tensor2, rounding_mode="floor")
