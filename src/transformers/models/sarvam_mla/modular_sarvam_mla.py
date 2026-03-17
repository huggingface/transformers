# Copyright 2026 Sarvam AI and the HuggingFace Inc. team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...utils import auto_docstring
from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config


@auto_docstring(checkpoint="sarvamai/sarvam-105b")
@strict(accept_kwargs=True)
class SarvamMLAConfig(DeepseekV3Config):
    r"""
    n_group (`int`, *optional*, defaults to 16):
        Number of groups for routed experts.
    rope_interleave (`bool`, *optional*, defaults to `True`):
        Whether to interleave the rotary position embeddings.
    first_k_dense_replace (`int`, *optional*, defaults to 1):
        Number of dense layers in shallow layers(embed->dense->moe->moe...->lm_head).
                                                        \--k dense layers--/

    Example:

    ```python
    >>> from transformers import SarvamMLAConfig

    >>> # Initializing a SarvamMLA style configuration
    >>> configuration = SarvamMLAConfig()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "sarvam_mla"

    vocab_size: int = 262144
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_hidden_layers: int = 32
    num_attention_heads: int = 64
    num_key_value_heads: int | None = None
    n_routed_experts: int = 128
    q_lora_rank: int | None = None
    n_group: int | None = 16
    topk_group: int | None = 2
    first_k_dense_replace: int | None = 1
    initializer_range: float = 0.006


__all__ = ["SarvamMLAConfig"]
