# Copyright 2024 The Kyutai and HuggingFace Inc. teams. All rights reserved.
#
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

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="kyutai/helium-1-preview")
@strict
class HeliumConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import HeliumModel, HeliumConfig
    >>> # Initializing a Helium 2b style configuration
    >>> configuration = HeliumConfig()
    >>> # Initializing a model from the Helium 2b style configuration
    >>> model = HeliumModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "helium"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 100000.0
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

    vocab_size: int = 48000
    hidden_size: int = 2560
    intermediate_size: int = 7040
    num_hidden_layers: int = 24
    num_attention_heads: int = 20
    num_key_value_heads: int = 20
    head_dim: int = 128
    hidden_act: str = "silu"
    attention_dropout: float | int = 0.0
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-8
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    pad_token_id: int | None = 3
    eos_token_id: int | list[int] | None = 2
    bos_token_id: int | None = 1
    attention_bias: bool = False
    mlp_bias: bool = False


__all__ = ["HeliumConfig"]
