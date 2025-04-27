# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the MIT License.
#

"""HindiCausalLM model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

HINDI_CAUSAL_LM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "convaiinnovations/hindi-foundational-model-base": "https://huggingface.co/convaiinnovations/hindi-foundational-model-base/resolve/main/config.json",
}


class HindiCausalLMConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`HindiCausalLMModel`]. It is used to
    instantiate a Hindi Causal Language Model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 16000):
            Vocabulary size of the Hindi Causal LM model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`HindiCausalLMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the model.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings and encoder.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            The ID of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The ID of the beginning of sequence token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The ID of the end of sequence token.
        positional_encoding_type (`str`, *optional*, defaults to `"rope"`):
            The type of positional encoding to use. Can be one of: `"absolute"`, `"learned"`, or `"rope"`.
        normalization_layer (`str`, *optional*, defaults to `"rmsnorm"`):
            The type of normalization layer to use. Can be one of: `"layernorm"` or `"rmsnorm"`.
    """

    model_type = "hindi_causal_lm"

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
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=3,
        tie_word_embeddings=True,
        positional_encoding_type="rope",
        normalization_layer="rmsnorm",
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.positional_encoding_type = positional_encoding_type
        self.normalization_layer = normalization_layer
        self.unk_token_id = unk_token_id

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
