# Copyright 2025 Bytedance-Seed Ltd and the HuggingFace Inc. team. All rights reserved.
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
"""SeedOss model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="ByteDance-Seed/Seed-OSS-36B-Instruct")
@strict
class SeedOssConfig(PreTrainedConfig):
    r"""
    attention_out_bias (`bool`, *optional*, defaults to `False`):
        Whether to use a bias in the output projection layer during self-attention.

    ```python
    >>> from transformers import SeedOssModel, SeedOssConfig

    >>> # Initializing a SeedOss-36b style configuration
    >>> configuration = SeedOssConfig()

    >>> # Initializing a model from the SeedOss-36b style configuration
    >>> model = SeedOssModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "seed_oss"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `SeedOssModel`
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

    vocab_size: int = 155136
    hidden_size: int = 4096
    intermediate_size: int = 27648
    num_hidden_layers: int = 64
    num_attention_heads: int = 80
    num_key_value_heads: int | None = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 524288
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = True
    attention_out_bias: bool = False
    attention_dropout: float | int = 0.1
    residual_dropout: float | int = 0.1
    mlp_bias: bool = False
    head_dim: int | None = 128

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)


__all__ = ["SeedOssConfig"]
