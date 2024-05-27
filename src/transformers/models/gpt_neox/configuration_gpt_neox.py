# coding=utf-8
# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
"""GPTNeoX model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class GPTNeoXConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GPTNeoXModel`]. It is used to instantiate an
    GPTNeoX model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the GPTNeoX
    [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50432):
            Vocabulary size of the GPTNeoX model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPTNeoXModel`].
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 44):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 24576):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        rotary_pct (`float`, *optional*, defaults to 0.25):
            percentage of hidden dimensions to allocate to rotary embeddings
        rotary_emb_base (`int`, *optional*, defaults to 10000)
            base for computing rotary embeddings frequency
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio probability of the attention score.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio of (1) the word embeddings, (2) the post-attention hidden states, and (3) the post-mlp
            hidden states.
        classifier_dropout (`float`, *optional*, defaults to 0.1):
            Argument used when doing token classification, used in the model [`GPTNeoXForTokenClassification`].

            The dropout ratio for the hidden layer.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 1e-5):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        use_parallel_residual (`bool`, *optional*, defaults to `True`):
            Whether to use a "parallel" formulation in each Transformer layer, which can provide a slight training
            speedup at large scales (e.g. 20B).
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, *optional*, defaults to `True`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.

        Example:

    ```python
    >>> from transformers import GPTNeoXConfig, GPTNeoXModel

    >>> # Initializing a GPTNeoX gpt-neox-20b style configuration
    >>> configuration = GPTNeoXConfig()

    >>> # Initializing a model (with random weights) from the gpt-neox-20b style configuration
    >>> model = GPTNeoXModel(configuration)  # doctest: +SKIP

    >>> # Accessing the model configuration
    >>> configuration = model.config  # doctest: +SKIP
    ```"""

    model_type = "gpt_neox"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50432,
        hidden_size=6144,
        num_hidden_layers=44,
        num_attention_heads=64,
        intermediate_size=24576,
        hidden_act="gelu",
        rotary_pct=0.25,
        rotary_emb_base=10000,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        classifier_dropout=0.1,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_parallel_residual=True,
        rope_scaling=None,
        attention_bias=True,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_parallel_residual = use_parallel_residual
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self._rope_scaling_validation()

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them!"
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
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
