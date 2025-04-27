# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Hindi Causal Language Model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

HINDI_CAUSAL_LM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "convaiinnovations/hindi-foundational-model-base": "https://huggingface.co/convaiinnovations/hindi-foundational-model-base/resolve/main/config.json",
}


class HindiCausalLMConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`HindiCausalLMModel`]. It is used to instantiate a
    Hindi Causal Language Model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    `convaiinnovations/hindi-foundational-model-base` architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 16000):
            Vocabulary size of the Hindi Causal LM model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`HindiCausalLMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder. Must divide hidden_size.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (often named feed-forward) layer in the Transformer decoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder. `"gelu"`, `"relu"`, `"silu"` and
            `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings and decoder.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
             The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers (applicable to LayerNorm and RMSNorm).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        unk_token_id (`int`, *optional*, defaults to 3):
             Unknown token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie input and output embeddings.
        normalization_layer (`str`, *optional*, defaults to `"rmsnorm"`):
             Type of normalization layer to use. Options: "layernorm", "rmsnorm".
        positional_encoding_type (`str`, *optional*, defaults to `"rope"`):
             Type of positional encoding. Options: "rope", "absolute", "learned". Currently, only "rope" is fully
             implemented in the provided code.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period for the Rotary Positional Embedding frequency calculation.
        architectures (`List[str]`, *optional*):
            Model architectures that can use this configuration class.

    Examples:

    ```python
    >>> from transformers import HindiCausalLMConfig, HindiCausalLMForCausalLM

    >>> # Initializing a Hindi Causal LM style configuration
    >>> configuration = HindiCausalLMConfig()

    >>> # Initializing a model from the configuration
    >>> model = HindiCausalLMForCausalLM(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "hindi_causal_lm"
    # Add keys used with AutoConfig mapping
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=16000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=16,
        intermediate_size=3072,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=3, # Added unk_token_id
        tie_word_embeddings=True,
        normalization_layer="rmsnorm",
        positional_encoding_type="rope", # Default to rope
        rope_theta=10000.0, # Added rope_theta
        architectures=None, # Set default below
        **kwargs,
    ):
        # Set default architectures if None
        if architectures is None:
            architectures = ["HindiCausalLMForCausalLM"]

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # Ensure hidden_size is divisible by num_attention_heads
        if hidden_size % num_attention_heads != 0:
             raise ValueError(
                 f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
             )
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.normalization_layer = normalization_layer
        self.positional_encoding_type = positional_encoding_type
        self.rope_theta = rope_theta
        self.unk_token_id = unk_token_id

        # Pass relevant arguments to the parent class __init__
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id, # Pass unk token id
            tie_word_embeddings=tie_word_embeddings,
            architectures=architectures, # Pass architectures
            **kwargs, # Pass remaining kwargs
        )