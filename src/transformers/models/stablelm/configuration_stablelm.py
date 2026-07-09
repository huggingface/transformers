# Copyright 2024 Stability AI and The HuggingFace Inc. team. All rights reserved.
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
"""StableLM model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="stabilityai/stablelm-3b-4e1t")
@strict
class StableLmConfig(PreTrainedConfig):
    r"""
    use_parallel_residual (`bool`, *optional*, defaults to `False`):
        Whether to use a "parallel" formulation in each Transformer layer, which can provide a slight training
        speedup at large scales.
    hidden_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio after applying the MLP to the hidden states.

    Example:

    ```python
    >>> from transformers import StableLmModel, StableLmConfig

    >>> # Initializing a StableLM stablelm-3b style configuration
    >>> configuration = StableLmConfig()
    ```"""

    model_type = "stablelm"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 50304
    intermediate_size: int = 6912
    hidden_size: int = 2560
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    layer_norm_eps: float = 1.0e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    use_qkv_bias: bool = False
    qk_layernorm: bool = False
    use_parallel_residual: bool = False
    hidden_dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 0
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.25)  # assign default for BC
        super().__post_init__(**kwargs)


__all__ = ["StableLmConfig"]
