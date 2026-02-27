# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

"""PyTorch Phi-MoE model."""

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import logging


logger = logging.get_logger(__name__)


class PhimoeConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PhimoeModel`]. It is used to instantiate a Phi-moe
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the
    [microsoft/Phi-3.5-MoE-instruct](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct).
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 32064):
            Vocabulary size of the Phimoe model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`PhimoeModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 6400):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `8`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to `4096*32`):
            The maximum sequence length that this model might ever be used with. Mixtral's sliding window attention
            allows sequence of up to 4096*32 tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        sliding_window (`int`, *optional*):
            Sliding window attention window size. If not specified, will default to `262144`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            The number of experts to root per-token, can be also interpreted as the `top-p` routing
            parameter
        num_local_experts (`int`, *optional*, defaults to 16):
            Number of experts per Sparse MLP layer.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss. See [here]() for more details
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        router_jitter_noise (`float`, *optional*, defaults to 0.01):
            Amount of noise to add to the router.
        input_jitter_noise (`float`, *optional*, defaults to 0.0): Input jitter noise
        attention_bias (`bool`, *optional*, defaults to `False`): Attention bias
        lm_head_bias (`bool`, *optional*, defaults to `False`): LM head bias

    Example:

    ```python
    >>> from transformers import PhimoeModel, PhimoeConfig
    >>> # Initializing a Phi-3 style configuration
    >>> configuration = PhimoeConfig.from_pretrained("microsoft/Phi-3.5-MoE-instruct")
    >>> # Initializing a model from the configuration
    >>> model = PhimoeModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "phimoe"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0

    def __init__(
        self,
        vocab_size: int | None = 32064,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 6400,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 8,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 4096 * 32,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-5,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: int | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        sliding_window: int | None = None,
        attention_dropout: float | None = 0.0,
        num_experts_per_tok: int | None = 2,
        num_local_experts: int | None = 16,
        output_router_logits: bool | None = False,
        router_aux_loss_coef: float | None = 0.001,
        router_jitter_noise: float | None = 0.01,
        input_jitter_noise: float | None = 0.0,
        attention_bias: bool | None = False,
        lm_head_bias: bool | None = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.attention_bias = attention_bias
        self.lm_head_bias = lm_head_bias
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.input_jitter_noise = input_jitter_noise
        self.rope_parameters = rope_parameters

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(**kwargs)

    def validate_rope(self, ignore_keys=None):
        """
        Validate the `rope_parameters` configuration.
        """
        super().validate_rope(ignore_keys=ignore_keys)

        # Run model-specific rope validation
        if self.rope_parameters["rope_type"] != "default":
            if "original_max_position_embeddings" in self.rope_parameters:
                self.original_max_position_embeddings = self.rope_parameters["original_max_position_embeddings"]
            rope_parameters_short_mscale = self.rope_parameters.get("short_mscale", None)
            rope_parameters_long_mscale = self.rope_parameters.get("long_mscale", None)
            if not isinstance(rope_parameters_short_mscale, (int, float)):
                raise TypeError(
                    f"`rope_parameters`'s short_mscale field must be a number, got {rope_parameters_short_mscale}"
                )
            if not isinstance(rope_parameters_long_mscale, (int, float)):
                raise TypeError(
                    f"`rope_parameters`'s long_mscale field must be a number, got {rope_parameters_long_mscale}"
                )


__all__ = ["PhimoeConfig"]
