# coding=utf-8
# Copyright 2024 Stability AI and The HuggingFace Inc. team. All rights reserved.
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
"""StableLM model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class StableLmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~StableLmModel`].
    It is used to instantiate an StableLM model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the StableLM [stabilityai/stablelm-3b-4e1t](https://huggingface.co/stabilityai/stablelm-3b-4e1t) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50304):
            Vocabulary size of the StableLM model. Defines the number of different tokens that
            can be represented by the `inputs_ids` passed when calling [`StableLmModel`].
        intermediate_size (`int`, *optional*, defaults to 6912):
            Dimension of the MLP representations.
        hidden_size (`int`, *optional*, defaults to 2560):
            Number of hidden layers in the Transformer decoder.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string).
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing
             all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions
            (not used by all models). Only relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to `10000.0`):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This
            is an experimental feature, subject to breaking API changes in future versions.
        yarn_rope_scaling (`Dict`, *optional*, defaults to `{'original_max_position_embeddings': 2048, 'extrapolation_factor': 1.0, 'attention_factor': 1.0, 'beta_fast': 32.0, 'beta_slow': 1.0, 'finetuned': False}`):
            Dictionary containing the YaRN-specific scaling configuration for the RoPE embeddings. The expected format is
            `{"original_max_position_embeddings": int, "extrapolation_factor": float, "attention_factor": float,
            "beta_fast": float, "beta_slow": float,"finetuned": bool}`.
            Fields:
                original_max_position_embeddings (`int`, *optional*, defaults to 2048):
                    The original maximum sequence length. This is used to scale the RoPE embeddings.
                extrapolation_factor (`float`, defaults to 1):
                    Factor to ajust the n-dimensional rotational scaling for extrapolation.
                attention_factor (`float`, *optional*, defaults to 1):
                    Factor to adjust the weight attention scaling mechanism.
                beta_fast (`float`, *optional*, defaults to 32):
                    Parameter to set the boundary for extrapolation (only) in the linear ramp function.
                beta_slow (`float`, *optional*, defaults to 1):
                    Parameter to set the boundary for interpolation (only) in the linear ramp function.
                finetuned (`bool`, *optional*, defaults to `False`):
                    [Dynamic] Whether the model is finetuned or not.
            For more details please refer to https://arxiv.org/abs/2309.00071.
        use_qkv_bias (`bool`, *optional*, defaults to `False`):
            Whether or not the model should use bias for qkv layers.
        qk_layernorm (`bool`, *optional*, defaults to `False`):
            Whether or not to normalize, per head, the Queries and Keys after projecting the hidden states.
        use_parallel_residual (`bool`, *optional*, defaults to `False`):
            Whether to use a "parallel" formulation in each Transformer layer, which can provide a slight training
            speedup at large scales.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio after applying the MLP to the hidden states.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        partial_rotary_factor (`float`, *optional*, defaults to 0.25):
            Percentage of the query and keys which will have rotary embedding.
        bos_token_id (int, *optional*, defaults to 0):
            The id of the `BOS` token in the vocabulary.
        eos_token_id (int, *optional*, defaults to 0):
            The id of the `EOS` token in the vocabulary.

    Example:

    ```python
    >>> from transformers import StableLmModel, StableLmConfig

    >>> # Initializing a StableLM stablelm-3b style configuration
    >>> configuration = StableLmConfig()
    ```"""

    model_type = "stablelm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50304,
        intermediate_size=6912,
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        layer_norm_eps=1.0e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10_000,
        rope_scaling=None,
        yarn_rope_scaling={
            "original_max_position_embeddings": 2048,
            "extrapolation_factor": 1.0,
            "attention_factor": 1.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "finetuned": False,
        },
        use_qkv_bias=False,
        qk_layernorm=False,
        use_parallel_residual=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        partial_rotary_factor=0.25,
        bos_token_id=0,
        eos_token_id=0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.yarn_rope_scaling = yarn_rope_scaling
        self.use_qkv_bias = use_qkv_bias
        self.qk_layernorm = qk_layernorm
        self.use_parallel_residual = use_parallel_residual
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.partial_rotary_factor = partial_rotary_factor
        self._rope_scaling_validation()
        self._yarn_rope_scaling_validation()

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # Copied from transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with two fields, `type` and `factor`, " f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic", "yarn", "dynamic-yarn"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic', 'yarn', 'dynamic-yarn'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")

    # Copied from transformers.models.llama.configuration_llama.LlamaConfig._yarn_rope_scaling_validation
    def _yarn_rope_scaling_validation(self):
        """
        Validate the `yarn_rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) > 6:
            raise ValueError(
                "`yarn_rope_scaling` must be a dictionary with a maximum of six fields, `original_max_position_embeddings`, "
                "`extrapolation_factor`, `attention_factor`, `beta_fast`, `beta_slow`, `finetuned`, "
                f"got {self.rope_scaling}"
            )
        original_max_position_embeddings = self.rope_scaling.get("original_max_position_embeddings", None)
        extrapolation_factor = self.rope_scaling.get("extrapolation_factor", None)
        attention_factor = self.rope_scaling.get("attention_factor", None)
        beta_fast = self.rope_scaling.get("beta_fast", None)
        beta_slow = self.rope_scaling.get("beta_slow", None)
        finetuned = self.rope_scaling.get("finetuned", None)

        if original_max_position_embeddings is not None and not isinstance(original_max_position_embeddings, int):
            raise ValueError(
                f"`yarn_rope_scaling`'s original_max_position_embeddings field must be an int, got {original_max_position_embeddings}"
            )
        if (
            extrapolation_factor is not None
            and not isinstance(extrapolation_factor, float)
            or extrapolation_factor < 0
            or extrapolation_factor > 1
        ):
            raise ValueError(
                f"`yarn_rope_scaling`'s extrapolation_factor field must be a float between 0 and 1, got {extrapolation_factor}"
            )
        if attention_factor is not None and not isinstance(attention_factor, float) or attention_factor < 0:
            raise ValueError(
                f"`yarn_rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
            )
        if beta_fast is not None and not isinstance(beta_fast, float):
            raise ValueError(f"`yarn_rope_scaling`'s beta_fast field must be a float, got {beta_fast}")
        if beta_slow is not None and not isinstance(beta_slow, float):
            raise ValueError(f"`yarn_rope_scaling`'s beta_slow field must be a float, got {beta_slow}")
        if finetuned is not None and not isinstance(finetuned, bool):
            raise ValueError(f"`yarn_rope_scaling`'s finetuned field must be a bool, got {finetuned}")

        b_fast = beta_fast if beta_fast is not None else 32
        b_slow = beta_slow if beta_slow is not None else 1
        if b_fast < b_slow:
            raise ValueError(
                f"`yarn_rope_scaling`'s beta_fast field must be greater than beta_slow, got beta_fast={b_fast} and beta_slow={b_slow}"
            )
