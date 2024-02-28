# coding=utf-8
# Copyright 2022 LLaMA-MoE Team and The HuggingFace Inc. team. All rights reserved.
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
""" LLaMA-MoE model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

LLAMA_MOE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "llama-moe/LLaMA-MoE-v1-3_5B-2_8": "https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_5B-2_8/resolve/main/config.json",
    "llama-moe/LLaMA-MoE-v1-3_5B-4_16": "https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_5B-4_16/resolve/main/config.json",
    "llama-moe/LLaMA-MoE-v1-3_0B-2_16": "https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_0B-2_16/resolve/main/config.json",
    # See all LLaMA-MoE models at https://huggingface.co/models?filter=llama_moe
}


class LlamaMoEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~LlamaMoEModel`].
    It is used to instantiate an LLaMA-MoE model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the LLaMA-MoE [llama-moe/LLaMA-MoE-v1-3_5B-2_8](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_5B-2_8) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA-MoE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~LlamaMoEModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer decoder.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
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
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts (`int`, *optional*, defaults to 8):
            The total number of expert for each layer.
        num_selects (`int`, *optional*, defaults to 2):
            The number of selected experts for each token.
        size_experts (`List[int]`, *optional*):
            The size of each expert. The total number of `size_experts` equals to `intermediate_size`
        gate_network (`str`, *optional*, defaults to `"mlp"`):
            The type of gate network. Currently supports `mlp` and `linear`.
            `linear` is a single MLP layer. `mlp` is a two-layer MLP with tanh activation.
        gate_use_softmax (`bool`, *optional*, defaults to `True`):
            Whether to use softmax for gating.
        gate_use_balance (`bool`, *optional*, defaults to `True`):
            Whether to use balance loss for gating.
        gate_balance_loss_weight (`float`, *optional*, defaults to 0.01):
            The weight of balance loss.
        gate_add_noise (`bool`, *optional*, defaults to `True`):
            Whether to add noise to gate scores.
        gate_noise_epsilon (`float`, *optional*, defaults to 0.01):
            The epsilon value for noise.
        multiply_gate_scores (`bool`, *optional*, defaults to `True`):
            Whether to multiply gate scores with the input.
        score_scale_factor (`float`, *optional*):
            The scale factor for the gate scores. If not specified, will default to `num_experts / num_selects`.

    Example:
        ```python
        >>> from transformers import LlamaMoEModel, LlamaMoEConfig

        >>> # Initializing a LLaMA-MoE llama-moe/LLaMA-MoE-v1-3_5B-2_8 style configuration
        >>> configuration = LlamaMoEConfig()

        >>> # Initializing a model from the llama-moe/LLaMA-MoE-v1-3_5B-2_8 style configuration
        >>> model = LlamaMoEModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """

    model_type = "llama_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        # MoE Expert Configs
        num_experts=8,
        num_selects=2,
        size_experts=None,
        # MoE Gate Configs
        gate_network="mlp",
        gate_use_softmax=True,
        gate_use_balance=True,
        gate_balance_loss_weight=1e-2,
        gate_add_noise=True,
        gate_noise_epsilon=1e-2,
        # MoE Calculator Configs
        multiply_gate_scores=True,
        score_scale_factor=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.num_experts = num_experts
        self.num_selects = num_selects
        self.size_experts = size_experts

        self.gate_network = gate_network
        self.gate_use_softmax = gate_use_softmax
        self.gate_use_balance = gate_use_balance
        self.gate_balance_loss_weight = gate_balance_loss_weight
        self.gate_add_noise = gate_add_noise
        self.gate_noise_epsilon = gate_noise_epsilon

        self.multiply_gate_scores = multiply_gate_scores
        if score_scale_factor is None:
            score_scale_factor = num_experts / num_selects
        self.score_scale_factor = score_scale_factor

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `name` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s name field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}")
