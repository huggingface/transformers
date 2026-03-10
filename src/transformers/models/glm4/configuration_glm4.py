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


from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="zai-org/GLM-OCR")
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

    def __init__(
        self,
        vocab_size: int | None = 151552,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 13696,
        num_hidden_layers: int | None = 40,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 2,
        head_dim: int | None = 128,
        hidden_act: str | None = "silu",
        attention_dropout: float | None = 0.0,
        max_position_embeddings: int | None = 131072,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 0.00000015625,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        pad_token_id: int | None = 151329,
        eos_token_id: list[int] | None = [151329, 151336, 151338],
        bos_token_id: int | None = None,
        attention_bias: bool | None = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(**kwargs)


__all__ = ["Glm4Config"]
