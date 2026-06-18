# Copyright 2024 BigCode and the HuggingFace Inc. team. All rights reserved.
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
"""Starcoder2 model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="bigcode/starcoder2-7b")
@strict
class Starcoder2Config(PreTrainedConfig):
    r"""
    use_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias term on linear layers of the model.

    ```python
    >>> from transformers import Starcoder2Model, Starcoder2Config

    >>> # Initializing a Starcoder2 7B style configuration
    >>> configuration = Starcoder2Config()

    >>> # Initializing a model from the Starcoder2 7B style configuration
    >>> model = Starcoder2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "starcoder2"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `Starcoder2`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.c_fc": "colwise",
        "layers.*.mlp.c_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 49152
    hidden_size: int = 3072
    intermediate_size: int = 12288
    num_hidden_layers: int = 30
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    hidden_act: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.018042
    norm_epsilon: float = 1e-5
    use_cache: bool = True
    bos_token_id: int | None = 50256
    eos_token_id: int | list[int] | None = 50256
    pad_token_id: int | None = None
    rope_parameters: RopeParameters | dict | None = None
    sliding_window: int | None = None
    attention_dropout: float | int = 0.0
    residual_dropout: float | int = 0.0
    embedding_dropout: float | int = 0.0
    use_bias: bool = True
    tie_word_embeddings: bool = True


__all__ = ["Starcoder2Config"]
