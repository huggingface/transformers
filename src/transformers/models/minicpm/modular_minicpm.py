# Copyright 2026 The OpenBMB Team and the HuggingFace Inc. team. All rights reserved.
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

import math

from huggingface_hub.dataclasses import strict

from ...utils import auto_docstring
from ..llama.configuration_llama import LlamaConfig


@auto_docstring(checkpoint="openbmb/MiniCPM4-8B")
@strict
class MiniCPMConfig(LlamaConfig):
    r"""
    scale_emb (`int` or `float`, *optional*, defaults to 12):
        Multiplier applied to input embeddings.
    scale_depth (`int` or `float`, *optional*, defaults to 1.4):
        Multiplier for residual connections. The effective scale is
        `scale_depth / sqrt(num_hidden_layers)`.
    dim_model_base (`int`, *optional*, defaults to 256):
        Base model dimension used to scale hidden states before the language model head.
    mup_denominator (`int`, *optional*, defaults to 32):
        Width denominator used by compatible speculative decoding heads.
    sparse_config (`dict`, *optional*):
        Configuration for the optional InfLLM-v2 sparse attention implementation.

    Example:

    ```python
    >>> from transformers import MiniCPMConfig

    >>> configuration = MiniCPMConfig()
    >>> configuration.hidden_size
    4096
    ```
    """

    model_type = "minicpm"

    # Defaults match the openbmb/MiniCPM4-8B checkpoint.
    vocab_size: int = 73448
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 2
    max_position_embeddings: int = 32768
    initializer_range: float = 0.1
    rms_norm_eps: float = 1e-6
    pad_token_id: int | None = 2
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    scale_emb: int | float = 12
    scale_depth: int | float | None = 1.4
    dim_model_base: int | None = 256
    mup_denominator: int | None = 32
    sparse_config: dict | None = None

    def __post_init__(self, **kwargs):
        if self.scale_depth is None:
            self.scale_depth = math.sqrt(self.num_hidden_layers)
        if self.dim_model_base is None:
            self.dim_model_base = self.hidden_size
        super().__post_init__(**kwargs)

    @property
    def logits_scaling(self) -> float:
        return self.hidden_size / self.dim_model_base


__all__ = ["MiniCPMConfig"]
