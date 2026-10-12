# Copyright 2025 The GLM4 & ZhipuAI team and HuggingFace Inc. team. All rights reserved.
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


@auto_docstring(checkpoint="zai-org/GLM-OCR")
@strict
class Glm4Config(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import Glm4Model, Glm4Config
    >>> # Initializing a Glm4 glm4-4-9b-chat style configuration
    >>> configuration = Glm4Config()
    >>> # Initializing a model from the glm4-4-9b-chat style configuration
    >>> model = Glm4Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm4"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_up_proj": "colwise_gather_output",  # we need to replicate here due to the `chunk` operation
        "layers.*.mlp.down_proj": "rowwise_split_input",  # input is replicated due to the `chunk` operation
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 151552
    hidden_size: int = 4096
    intermediate_size: int = 13696
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    hidden_act: str = "silu"
    attention_dropout: float | int = 0.0
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 0.00000015625
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    pad_token_id: int | None = 151329
    eos_token_id: int | list[int] | None = None
    bos_token_id: int | None = None
    attention_bias: bool = True

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC
        if self.eos_token_id is None:
            self.eos_token_id = [151329, 151336, 151338]
        super().__post_init__(**kwargs)


__all__ = ["Glm4Config"]
