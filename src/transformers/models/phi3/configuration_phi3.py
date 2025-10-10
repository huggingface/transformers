# coding=utf-8
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

"""Phi-3 model configuration"""

from typing import Optional

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters, rope_config_validation, standardize_rope_params
from ...utils import logging


logger = logging.get_logger(__name__)


class Phi3Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Phi3Model`]. It is used to instantiate a Phi-3
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the
    [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32064):
            Vocabulary size of the Phi-3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Phi3Model`].
        hidden_size (`int`, *optional*, defaults to 3072):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 8192):
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
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            Dropout probability for mlp outputs.
        embd_pdrop (`int`, *optional*, defaults to 0.0):
            The dropout ratio for the embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio after computing the attention scores.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        original_max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model was trained with. This is used to determine the size of the
            original RoPE embeddings when using long scaling.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value used for the RMSNorm.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`. Whether to tie weight embeddings or not.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionaty should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        partial_rotary_factor (`float`, *optional*, defaults to 1.0):
            Percentage of the query and keys which will have rotary embedding. Must be between 0.0 and 1.0.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 32000):
            The id of the "end-of-sequence" token.
        pad_token_id (`int`, *optional*, defaults to 32000):
            The id of the padding token.
        sliding_window (`int`, *optional*):
            Sliding window attention window size. If `None`, no sliding window is applied.

    Example:

    ```python
    >>> from transformers import Phi3Model, Phi3Config

    >>> # Initializing a Phi-3 style configuration
    >>> configuration = Phi3Config.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    >>> # Initializing a model from the configuration
    >>> model = Phi3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "phi3"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.qkv_proj": "colwise_rep",  # we need to replicate here due to the slicing of qkv
        "layers.*.self_attn.o_proj": "rowwise_rep",  # we need to replicate here due to the slicing of qkv
        "layers.*.mlp.gate_up_proj": "colwise_rep",  # we need to replicate here due to the `chunk` operation
        "layers.*.mlp.down_proj": "rowwise_rep",  # we need to replicate here due to the `chunk` operation
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: Optional[int] = 32064,
        hidden_size: Optional[int] = 3072,
        intermediate_size: Optional[int] = 8192,
        num_hidden_layers: Optional[int] = 32,
        num_attention_heads: Optional[int] = 32,
        num_key_value_heads: Optional[int] = None,
        resid_pdrop: Optional[float] = 0.0,
        embd_pdrop: Optional[float] = 0.0,
        attention_dropout: Optional[float] = 0.0,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 4096,
        original_max_position_embeddings: Optional[int] = 4096,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[int] = 1e-5,
        use_cache: Optional[bool] = True,
        tie_word_embeddings: Optional[bool] = False,
        rope_parameters: Optional[RopeParameters | dict[RopeParameters]] = None,
        partial_rotary_factor: Optional[float] = 1.0,
        bos_token_id: Optional[int] = 1,
        eos_token_id: Optional[int] = 32000,
        pad_token_id: Optional[int] = 32000,
        sliding_window: Optional[int] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.partial_rotary_factor = partial_rotary_factor
        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or rope_parameters

        # Validate the correctness of rotary position embeddings parameters
        rope_theta = kwargs.get("rope_theta", 10000.0)
        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)
        self._rope_parameters_adjustment()
        self._rope_parameters_validation()
        self.sliding_window = sliding_window

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_parameters_adjustment(self):
        """
        Adjust the `type` of the `rope_parameters` configuration for backward compatibility.
        """
        rope_parameters_type = self.rope_parameters.get("rope_type", None)

        # For backward compatibility if previous version used "su" or "yarn"
        if rope_parameters_type is not None and rope_parameters_type in ["su", "yarn"]:
            self.rope_parameters["rope_type"] = "longrope"

    def _rope_parameters_validation(self):
        """
        Validate the `rope_parameters` configuration.
        """
        if not isinstance(self.rope_parameters, dict):
            raise ValueError(f"`rope_parameters` must be a dictionary but got {self.rope_parameters}")
        rope_parameters_type = self.rope_parameters.get("rope_type", None)
        rope_parameters_short_factor = self.rope_parameters.get("short_factor", None)
        rope_parameters_long_factor = self.rope_parameters.get("long_factor", None)
        rotary_ndims = int(self.hidden_size // self.num_attention_heads * self.partial_rotary_factor)
        if rope_parameters_type not in ["default", "longrope"]:
            raise ValueError(f"`rope_parameters`'s type field must be one of ['longrope'], got {rope_parameters_type}")

        if rope_parameters_short_factor is not None:
            if not (
                isinstance(rope_parameters_short_factor, list)
                and all(isinstance(x, (int, float)) for x in rope_parameters_short_factor)
            ):
                raise ValueError(
                    f"`rope_parameters`'s short_factor field must be a list of numbers, got {rope_parameters_short_factor}"
                )
            if not len(rope_parameters_short_factor) == rotary_ndims // 2:
                raise ValueError(
                    f"`rope_parameters`'s short_factor field must have length {rotary_ndims // 2}, got {len(rope_parameters_short_factor)}"
                )

        if rope_parameters_long_factor is not None:
            if not (
                isinstance(rope_parameters_long_factor, list)
                and all(isinstance(x, (int, float)) for x in rope_parameters_long_factor)
            ):
                raise ValueError(
                    f"`rope_parameters`'s long_factor field must be a list of numbers, got {rope_parameters_long_factor}"
                )
            if not len(rope_parameters_long_factor) == rotary_ndims // 2:
                raise ValueError(
                    f"`rope_parameters`'s long_factor field must have length {rotary_ndims // 2}, got {len(rope_parameters_long_factor)}"
                )


__all__ = ["Phi3Config"]
