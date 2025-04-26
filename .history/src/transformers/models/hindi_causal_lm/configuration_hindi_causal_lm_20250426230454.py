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
    Hindi Causal Language Model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 16000):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`HindiCausalLMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
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
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of sequence token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of sequence token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie input and output embeddings.
        normalization_layer (`str`, *optional*, defaults to `"rmsnorm"`):
            Type of normalization layer, options are "layernorm" and "rmsnorm".
        positional_encoding_type (`str`, *optional*, defaults to `"rope"`):
            Type of positional encoding, options are "absolute", "learned", and "rope".
        architectures (`List[str]`, *optional*):
            Model architectures that can use this configuration class.

    Examples:

    ```python
    >>> from transformers import HindiCausalLMConfig, HindiCausalLMModel

    >>> # Initializing a Hindi Causal LM style configuration
    >>> configuration = HindiCausalLMConfig()

    >>> # Initializing a model from the Hindi Causal LM style configuration
    >>> model = HindiCausalLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

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
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        tie_word_embeddings=True,
        normalization_layer="rmsnorm",
        positional_encoding_type="rope",
        unk_token_id=3,
        architectures=None,
        **kwargs,
    ):
        if architectures is None:
            architectures = ["HindiCausalLMForCausalLM"]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            architectures=architectures,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.normalization_layer = normalization_layer
        self.positional_encoding_type = positional_encoding_type
        self.unk_token_id = unk_token_id