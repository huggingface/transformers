# Copyright 2024 Databricks Mosaic Research and The HuggingFace Inc. team. All rights reserved.
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
"""DBRX model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@strict
@auto_docstring(
    custom_intro="This config is used to instantiate attention layers.",
    checkpoint="transformers-community/dbrx-instruct",
)
class DbrxAttentionConfig(PreTrainedConfig):
    r"""
    attn_pdrop (`float`, *optional*, defaults to 0.0):
        The dropout probability for the attention layers.
    clip_qkv (`float`, *optional*):
        If set, clip the queries, keys, and values in the attention layer to this value.
    kv_n_heads (`int`, *optional*, defaults to 1):
        For grouped_query_attention only, allow user to specify number of kv heads.
    """

    base_config_key = "attn_config"

    attn_pdrop: float | int = 0.0
    clip_qkv: int | float | None = None
    kv_n_heads: int = 1


@strict
@auto_docstring(
    custom_intro="This config is used to instantiate feedforward layers.",
    checkpoint="transformers-community/dbrx-instruct",
)
class DbrxFFNConfig(PreTrainedConfig):
    r"""
    ffn_act_fn (`dict`, *optional*, defaults to `None`):
        A dict specifying activation function for the FFN.
        The dict should have a key 'name' with the value being the name of the activation function along with
        any additional keyword arguments. If `None`, then set to `{"name": "silu"}`.
    ffn_hidden_size (`int`, *optional*, defaults to 3584):
        The hidden size of the feedforward network.
    moe_num_experts (`int`, *optional*, defaults to 4):
        The number of experts in the mixture of experts layer.
    moe_top_k (`int`, *optional*, defaults to 1):
        The number of experts to use in the mixture of experts layer.
    moe_jitter_eps (`float`, *optional*, defaults to `None`):
        If not `None`, the jitter epsilon for the mixture of experts layer.
    moe_loss_weight (`float`, *optional*, defaults to 0.01):
        The loss weight for the mixture of experts layer.
    moe_normalize_expert_weights (`float`, *optional*, defaults to 1.0):
        The normalization factor for the expert weights.
    """

    base_config_key = "ffn_config"

    hidden_size: int = 6144
    ffn_act_fn: dict | None = None
    ffn_hidden_size: int = 3584
    moe_num_experts: int = 4
    moe_top_k: int = 1
    moe_jitter_eps: float | None = None
    moe_loss_weight: float = 0.01
    moe_normalize_expert_weights: float | None = 1.0

    def __post_init__(self, **kwargs):
        if self.ffn_act_fn is None:
            self.ffn_act_fn = {"name": "silu"}

        for k in [
            "model_type",
            "attn_implementation",
            "experts_implementation",
            "transformers_version",
            "_commit_hash",
            "torch_dtype",
            "dtype",
        ]:
            if k in kwargs:
                kwargs.pop(k)
        if len(kwargs) != 0:
            raise ValueError(f"Found unknown {kwargs=}")

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="transformers-community/dbrx-instruct")
@strict
class DbrxConfig(PreTrainedConfig):
    r"""
    max_seq_len (`int`, *optional*, defaults to 2048):
        The maximum sequence length of the model.
    attn_config (`dict`, *optional*):
        A dictionary used to configure the model's attention module.
    ffn_config (`dict`, *optional*):
        A dictionary used to configure the model's FFN module.

    Example:
    ```python
    >>> from transformers import DbrxConfig, DbrxModel

    >>> # Initializing a Dbrx configuration
    >>> configuration = DbrxConfig(n_layers=2, d_model=256, n_heads=8, vocab_size=128)

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = DbrxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "dbrx"
    sub_configs = {"attn_config": DbrxAttentionConfig, "ffn_config": DbrxFFNConfig}
    attribute_map = {
        "num_attention_heads": "n_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layers",
        "max_position_embeddings": "max_seq_len",
    }

    d_model: int | None = 2048
    n_heads: int | None = 16
    n_layers: int | None = 24
    max_seq_len: int | None = 2048
    vocab_size: int = 32000
    resid_pdrop: float | None = 0.0
    emb_pdrop: float | None = 0.0
    attn_config: DbrxAttentionConfig | dict | None = None
    ffn_config: DbrxFFNConfig | dict | None = None
    use_cache: bool = True
    initializer_range: float = 0.02
    output_router_logits: bool | None = False
    rope_parameters: RopeParameters | dict | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if self.attn_config is None:
            self.attn_config = DbrxAttentionConfig()
        elif isinstance(self.attn_config, dict):
            self.attn_config = DbrxAttentionConfig(**self.attn_config)

        if self.ffn_config is None:
            self.ffn_config = DbrxFFNConfig()
        elif isinstance(self.ffn_config, dict):
            self.ffn_config = DbrxFFNConfig(**self.ffn_config)

        self.num_key_value_heads = self.attn_config.kv_n_heads
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.tie_word_embeddings:
            raise ValueError("tie_word_embeddings is not supported for DBRX models.")


__all__ = ["DbrxConfig"]
