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

from typing import Optional

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters, rope_config_validation, standardize_rope_params


class SeedOssConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SeedOssModel`]. It is used to instantiate an SeedOss
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the SeedOss-36B.
    e.g. [ByteDance-Seed/SeedOss-36B](https://huggingface.co/ByteDance-Seed/SeedOss-36B)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 155136):
            Vocabulary size of the SeedOss model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`SeedOssModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 27648):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 64):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 80):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `8`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 524288):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to
            understand more about it. This value is necessary to ensure exact reproducibility of the pretraining
            results. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionaty should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to use a bias in the query, key, value layers during self-attention.
        attention_out_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the output projection layer during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        residual_dropout (`float`, *optional*, defaults to 0.1):
            Residual connection dropout value.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.

    ```python
    >>> from transformers import SeedOssModel, SeedOssConfig

    >>> # Initializing a SeedOss-36b style configuration
    >>> configuration = SeedOssConfig()

    >>> # Initializing a model from the SeedOss-36b style configuration
    >>> model = SeedOssModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

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

    def __init__(
        self,
        vocab_size: Optional[int] = 155136,
        hidden_size: Optional[int] = 4096,
        intermediate_size: Optional[int] = 27648,
        num_hidden_layers: Optional[int] = 64,
        num_attention_heads: Optional[int] = 80,
        num_key_value_heads: Optional[int] = 8,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 524288,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[float] = 1e-6,
        use_cache: Optional[bool] = True,
        pad_token_id: Optional[int] = 1,
        bos_token_id: Optional[int] = 0,
        eos_token_id: Optional[int] = 2,
        pretraining_tp: Optional[int] = 1,
        tie_word_embeddings: Optional[bool] = False,
        rope_parameters: Optional[RopeParameters | dict[RopeParameters]] = None,
        attention_bias: Optional[bool] = True,
        attention_out_bias: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.1,
        residual_dropout: Optional[float] = 0.1,
        mlp_bias: Optional[bool] = False,
        head_dim: Optional[int] = 128,
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
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_out_bias = attention_out_bias
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or rope_parameters

        # Validate the correctness of rotary position embeddings parameters
        rope_theta = kwargs.get("rope_theta", 10000.0)
        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["SeedOssConfig"]
