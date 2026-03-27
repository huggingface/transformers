# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from ..utils import logging


logger = logging.get_logger(__name__)

def get_best_loading_strategy(config, model_name_or_path=None):
    """
    Analyzes hardware and returns a dictionary of recommended kwargs for from_pretrained.
    """
    has_cuda = torch.cuda.is_available()
    vram_gb = 0
    if has_cuda:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # Heuristic parameter count
    params = 0
    if hasattr(config, "num_hidden_layers") and hasattr(config, "hidden_size"):
        params = config.num_hidden_layers * (config.hidden_size**2) * 12
    elif hasattr(config, "n_layer") and hasattr(config, "n_embd"):
        params = config.n_layer * (config.n_embd**2) * 12

    params_b = params / (10**9)
    vram_fp16 = params_b * 2
    vram_int4 = params_b * 0.7

    strategy = {"device_map": "auto"}

    if not has_cuda:
        strategy["device_map"] = None # Fallback to CPU
        return strategy

    if vram_gb > vram_fp16 + 2:
        # Plenty of VRAM
        strategy["torch_dtype"] = torch.float16
    elif vram_gb > vram_int4 + 1.5:
        # Use 4-bit
        strategy["load_in_4bit"] = True
    else:
        # Tight VRAM
        strategy["load_in_4bit"] = True
        strategy["llm_int8_enable_fp32_cpu_offload"] = True

    return strategy
