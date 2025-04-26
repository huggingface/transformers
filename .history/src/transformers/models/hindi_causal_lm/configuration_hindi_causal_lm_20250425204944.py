# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team. All rights reserved.
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
"""HindiCausalLM configuration"""

# **** ADD THIS IMPORT ****
from ...activations import ACT2FN

# ***********************
from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

HINDI_CAUSAL_LM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "convaiinnovations/hindi-embedding-foundational-model": "https://huggingface.co/convaiinnovations/hindi-embedding-foundational-model/resolve/main/config.json",
    # Add other checkpoints here if needed
}


class HindiCausalLMConfig(PretrainedConfig):
    """
    Configuration class for HindiCausalLM model.

    This is the configuration class to store the configuration of a [`HindiCausalLMModel`].
    It is used to instantiate a Hindi causal language model according to the specified arguments.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 16000):
            Vocabulary size of the HindiCausalLM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`HindiCausalLMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the hidden layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            *Note*: For SwiGLU/GeGLU activations, this is the size of the projection *before* gating.
            Standard SwiGLU often uses `intermediate_size = round(8/3 * hidden_size)`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
            `"silu"` corresponds to SwiGLU FFN structure.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            The ID of the padding token in the vocabulary.
        bos_token_id (`int`, *optional*, defaults to 1):
            The ID of the beginning-of-sequence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 2):
            The ID of the end-of-sequence token in the vocabulary.
        unk_token_id (`int`, *optional*, defaults to 3):
            The ID of the unknown token in the vocabulary.
        tie_word_embeddings (`bool`, *optional*, defaults to True):
            Whether the model's input and output word embeddings should be tied. Tying weights reduces parameter count
            and is common practice.
        normalization_layer (`str`, *optional*, defaults to "rmsnorm"):
            The normalization layer to use. Supports "layernorm" and "rmsnorm".
        positional_encoding_type (`str`, *optional*, defaults to "rope"):
            The positional encoding to use. Supports "absolute", "learned", and "rope".
    """

    model_type = "hindi_causal_lm"
    # Add keys_to_ignore_at_inference if needed, e.g., ["past_key_values"]

    def __init__(
        self,
        vocab_size=16000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=16,
        intermediate_size=3072,  # Keep existing default, but document standard SwiGLU size
        hidden_act="silu",  # Changed from activation_function, default to "silu" for SwiGLU
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=3,
        tie_word_embeddings=True,
        normalization_layer="rmsnorm",
        positional_encoding_type="rope",
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            unk_token_id=unk_token_id,  # Make sure unk_token_id is passed to parent
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act  # Changed from activation_function
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.normalization_layer = normalization_layer
        self.positional_encoding_type = positional_encoding_type
        # self.unk_token_id = unk_token_id # Already handled in super().__init__

        # Validate activation function - NOW ACT2FN IS DEFINED
        if isinstance(self.hidden_act, str) and self.hidden_act not in ACT2FN:
            logger.warning(
                f"Activation function '{self.hidden_act}' not found in ACT2FN, loading may fail if model code relies on it."
                " Note: For GLU variants like SwiGLU, set hidden_act to the internal activation (e.g., 'silu')."
            )
