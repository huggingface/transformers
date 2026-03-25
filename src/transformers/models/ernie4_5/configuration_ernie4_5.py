# Copyright (c) 2025 Baidu, Inc. and HuggingFace Inc. team. All Rights Reserved.
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
"""Ernie 4.5 model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="baidu/ERNIE-4.5-0.3B-PT")
@strict
class Ernie4_5Config(PreTrainedConfig):
    r"""
    use_bias (`bool`, *optional*, defaults to `False`):
        Whether to use a bias in any of the projections including mlp and attention for example.

    Example:

    ```python
    >>> from transformers import Ernie4_5Model, Ernie4_5Config

    >>> # Initializing a Ernie4_5 0.3B style configuration
    >>> configuration = Ernie4_5Config()

    >>> # Initializing a model from the 0.3B style configuration
    >>> model = Ernie4_5Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ernie4_5"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 500000.0
    # Default tensor parallel plan for base model `Ernie4_5Model`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 103424
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 18
    num_attention_heads: int = 16
    num_key_value_heads: int | None = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-05
    use_cache: int | None = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    use_bias: bool | None = False
    head_dim: int | None = 128

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)


__all__ = ["Ernie4_5Config"]
