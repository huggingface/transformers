# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Bark model configuration"""


from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

BARK_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "sanchit-gandhi/bark": "https://huggingface.co/sanchit-gandhi/bark/resolve/main/config.json",
}


class BarkConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BarkModel`]. It is used to instantiate a GPT Neo
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Bark
    [sanchit-gandhi/bark](https://huggingface.co/sanchit-gandhi/bark) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        input_vocab_size (`int`, *optional*, defaults to 10048):
            Input vocabulary size of the Bark model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BarkModel`]. Defines the different tokens
            that can be represented by the *inputs_ids* passed to the forward method of [`BarkModel`].
        output_vocab_size (`int`, *optional*, defaults to 10048):
            Output vocabulary size of the Bark model. Defines the number of different tokens that can be represented by the
            output obtained when calling [`BarkModel`]. Defines the different tokens
            that can be represented by the *logits* returned by the forward method of [`BarkModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        activation_function (`str` or `function`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        embed_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically, set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models). Only relevant if
            `config.is_decoder=True`.
        use_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the projection layers in the transformer block.
        num_codebooks (`int`, *optional*, defaults to 8):
            Number of codebooks (or quantizers) used to reconstruct the waveform.

    Example:

    ```python
    >>> from transformers import BarkConfig, BarkModel

    >>> # Initializing a Bark sanchit-gandhi/bark style configuration
    >>> configuration = BarkConfig()

    >>> # Initializing a model (with random weights) from the sanchit-gandhi/bark style configuration
    >>> model = BarkModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "bark"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    def __init__(
        self,
        vocab_size=10048,
        max_position_embeddings=1024,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=None,
        activation_function="gelu_new",
        resid_dropout=0.0,
        embed_dropout=0.0,
        attention_dropout=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        use_bias=True,
        num_codebooks=8,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.activation_function = activation_function
        self.resid_dropout = resid_dropout
        self.embed_dropout = embed_dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.use_bias = use_bias
        self.num_codebooks = num_codebooks

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
