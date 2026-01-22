# Copyright 2025 IBM and the HuggingFace Inc. team. All rights reserved.
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
"""GraniteMoeHybrid model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters, RotaryEmbeddingConfigMixin
from ...utils import logging


logger = logging.get_logger(__name__)


class GraniteMoeHybridConfig(PreTrainedConfig, RotaryEmbeddingConfigMixin):
    r"""
    This is the configuration class to store the configuration of a [`GraniteMoeHybridConfig`]. It is used to
    instantiate an GraniteMoeHybrid model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the GraniteMoeHybrid model. Defines the number of different tokens that
            can be represented by the `inputs_ids` passed when calling [`GraniteMoeHybridModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
            Only relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        embedding_multiplier (`float`, *optional*, defaults to 1.0): embedding multiplier.
        logits_scaling (`float`, *optional*, defaults to 1.0): divisor for output logits.
        residual_multiplier (`float`, *optional*, defaults to 1.0): residual multiplier.
        attention_multiplier (`float`, *optional*, defaults to 1.0): attention multiplier.
        num_local_experts (`int`, *optional*, defaults to 8): total number of experts.
        num_experts_per_tok (`int`, *optional*, defaults to 2): number of experts per token.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001): router auxiliary loss coefficient
        shared_intermediate_size (`int`, *optional*, defaults to 1024): intermediate size for shared experts.
        position_embedding_type (`str`, *optional*):
            Positional embedding type to be used; defaults to None. Allowed options: `[None, "rope"]`
        layer_types (`List`, *optional*): list of strings to be used as layer types.
            Allowed choices: "mamba", "attention".
        mamba_n_heads (`int`, *optional*, defaults to 128):
            The number of mamba heads used.
        mamba_n_groups (`int`, *optional*, defaults to 1):
            The number of the mamba groups used.
        mamba_d_state (`int`, *optional*, defaults to 256):
            The dimension the mamba latent state space.
        mamba_d_head (`int`, *optional*, defaults to `"auto"`):
            Head embedding dimension size.
        mamba_d_conv (`int`, *optional*, defaults to 4):
            The size of the mamba convolution kernel.
        mamba_expand (`int`, *optional*, defaults to 2):
            Expanding factor (relative to hidden_size) used to determine the mamba intermediate size.
        mamba_chunk_size (`int`, *optional*, defaults to 256):
            The chunks in which to break the sequence when doing prefill/training.
        mamba_conv_bias (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use bias in the convolution layer of the mamba mixer block.
        mamba_proj_bias (`bool`, *optional*, defaults to `False`):
            Flag indicating whether or not to use bias in the input and output projections (["in_proj", "out_proj"])
            of the mamba mixer block.
    ```python
    >>> from transformers import GraniteMoeHybridModel, GraniteMoeHybridConfig

    >>> # Initializing a GraniteMoeHybrid config
    >>> configuration = GraniteMoeHybridConfig()


    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "granitemoehybrid"
    attribute_map = {
        "layers_block_type": "layer_types",
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int | None = 32000,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 11008,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 2048,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-6,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        embedding_multiplier: float | None = 1.0,
        logits_scaling: float | None = 1.0,
        residual_multiplier: float | None = 1.0,
        attention_multiplier: float | None = 1.0,
        num_local_experts: int | None = 8,
        num_experts_per_tok: int | None = 2,
        output_router_logits: bool | None = False,
        router_aux_loss_coef: float | None = 0.001,
        shared_intermediate_size: int | None = 1024,
        position_embedding_type: str | None = None,
        layer_types: list[str] | None = None,
        mamba_n_heads: int | None = 128,
        mamba_n_groups: int | None = 1,
        mamba_d_state: int | None = 256,
        mamba_d_head: str | None = "auto",
        mamba_d_conv: int | None = 4,
        mamba_expand: int | None = 2,
        mamba_chunk_size: int | None = 256,
        mamba_conv_bias: bool | None = True,
        mamba_proj_bias: bool | None = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.embedding_multiplier = embedding_multiplier
        self.logits_scaling = logits_scaling
        self.residual_multiplier = residual_multiplier
        self.attention_multiplier = attention_multiplier
        self.attention_dropout = attention_dropout
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.shared_intermediate_size = shared_intermediate_size
        self.position_embedding_type = position_embedding_type
        self.rope_parameters = rope_parameters

        mamba_intermediate = mamba_expand * hidden_size

        if layer_types is not None and any(layer_type not in ["mamba", "attention"] for layer_type in layer_types):
            raise ValueError("layer_types must be a list strings in  [`mamba` `attention`]")

        if mamba_intermediate % mamba_n_heads != 0:
            raise ValueError("mamba_n_heads must divide mamba_expand * hidden_size")

        # for the mamba_v2, must satisfy the following
        if mamba_d_head == "auto":
            mamba_d_head = mamba_intermediate // mamba_n_heads

        if mamba_d_head * mamba_n_heads != mamba_intermediate:
            raise ValueError("The dimensions for the Mamba head state do not match the model intermediate_size")

        self.mamba_n_heads = mamba_n_heads
        self.mamba_d_head = mamba_d_head
        self.mamba_n_groups = mamba_n_groups
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_chunk_size = mamba_chunk_size
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.mamba_expand = mamba_expand
        self.layer_types = layer_types

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(**kwargs)

    # overwrite the function to use in `HybridMambaAttentionDynamicCache`
    @property
    def layers_block_type(self):
        return self.layer_types if self.layer_types else ["mamba"] * self.num_hidden_layers


__all__ = ["GraniteMoeHybridConfig"]
