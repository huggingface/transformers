# Copyright 2025 The rednote-hilab team and the HuggingFace Inc. team. All rights reserved.
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

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_rope_utils import RopeParameters, RotaryEmbeddingConfigMixin
from ...utils import logging


logger = logging.get_logger(__name__)


class Dots1Config(PreTrainedConfig, RotaryEmbeddingConfigMixin):
    r"""
    This is the configuration class to store the configuration of a [`Dots1Model`]. It is used to instantiate a
    `dots.llm1` model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    [rednote-hilab/dots.llm1.base](https://huggingface.co/rednote-hilab/dots.llm1.base).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 152064):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `input_ids` passed when calling [`Dots1Model`].
        hidden_size (`int`, *optional*, defaults to 4608):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 10944):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 1408):
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 62):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            Number of key/value heads for Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, Multi
            Head Attention (MHA) is used. If `num_key_value_heads=1`, Multi Query Attention (MQA) is used. Otherwise,
            Grouped Query Attention (GQA) is used. If not specified, defaults to `num_attention_heads`.
        n_shared_experts (`int`, *optional*, default=None):
            Number of shared experts. None means dense model.
        n_routed_experts (`int`, *optional*, default=None):
            Number of routed experts. None means dense model.
        n_group (`int`, *optional*, defaults to 1):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 1):
            Number of selected groups for each token (selected experts only within `topk_group` groups).
        num_experts_per_tok (`int`, *optional*, default=None):
            Number of selected experts. None means dense model.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of dense layers at the beginning of the model before the first MoE layer.
        norm_topk_prob (`bool`, *optional*, defaults to `False`):
            Whether to normalize the weights of the routed experts.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string).
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            Maximum sequence length the model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions. Only relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the input and output word embeddings.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the self-attention projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for the attention probabilities.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for routed experts.
        sliding_window (`int`, *optional*, defaults to 4096):
            Size of the sliding window for attention. If not specified, defaults to `4096`.
        max_window_layers (`int`, *optional*, defaults to 62):
            The number of layers using full attention. The first `max_window_layers` layers will use full attention, while any
            additional layer afterwards will use SWA (Sliding Window Attention).
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*):
            End of stream token id.

    Examples:
        ```python
        >>> from transformers import Dots1Model, Dots1Config

        >>> # Initializing a Dots1 style configuration
        >>> configuration = Dots1Config()

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """

    model_type = "dots1"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "rowwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(
        self,
        vocab_size: int | None = 152064,
        hidden_size: int | None = 4608,
        intermediate_size: int | None = 10944,
        moe_intermediate_size: int | None = 1408,
        num_hidden_layers: int | None = 62,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 32,
        n_shared_experts: int | None = None,
        n_routed_experts: int | None = None,
        n_group: int | None = 1,
        topk_group: int | None = 1,
        num_experts_per_tok: int | None = None,
        first_k_dense_replace: int | None = 0,
        norm_topk_prob: bool | None = False,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 2048,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-6,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        routed_scaling_factor: float | None = 1.0,
        sliding_window: int | None = 4096,
        max_window_layers: int | None = 62,
        layer_types: list[str] | None = None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.routed_scaling_factor = routed_scaling_factor
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.rope_parameters = rope_parameters

        super().__init__(**kwargs)


__all__ = ["Dots1Config"]
