# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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


@auto_docstring(checkpoint="karpathy/nanochat-d32")
@strict
class NanoChatConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import NanoChatModel, NanoChatConfig

    >>> # Initializing a NanoChat style configuration
    >>> configuration = NanoChatConfig()

    >>> # Initializing a model from the NanoChat style configuration
    >>> model = NanoChatModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "nanochat"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.fc1": "colwise",
        "layers.*.mlp.fc2": "rowwise",
    }

    vocab_size: int = 50304
    hidden_size: int = 768
    intermediate_size: int = 8192
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    num_key_value_heads: int | None = None
    max_position_embeddings: int = 2048
    hidden_act: str = "relu2"
    attention_dropout: float | int = 0.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    rope_parameters: RopeParameters | dict | None = None
    use_cache: bool = True
    final_logit_softcapping: float | None = 15.0
    attention_bias: bool = False
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    pad_token_id: int | None = 1
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)


__all__ = ["NanoChatConfig"]
