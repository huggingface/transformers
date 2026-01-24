# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# limitations under the License.from transformers import PretrainedConfig
from transformers import PretrainedConfig


BEIT3_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Raghavan/beit3_base_patch16_224_in1k": (
        "https://huggingface.co/Raghavan/beit3_base_patch16_224_in1k/blob/main/config.json"
    ),
    # See all BEiT models at https://huggingface.co/models?filter=beit
}


class Beit3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Beit3Model`]. It is used to instantiate a BEiT3
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the BEiT3
    [microsoft/beit3-base-patch16-224-pt22k](https://huggingface.co/microsoft/beit3-base-patch16-224-pt22k)
    architecture.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        encoder_normalize_before (`bool`, *optional*, defaults to `False`):
            Whether to normalize before the encoder block.
        activation_fn (`str`, *optional*, defaults to `"gelu"`):
            Activation function to apply within Mega encoder blocks. Choose one of `"silu"`, `"relu"`, `"linear"`,
            `"gelu"`, or `"gelu_accurate"`
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to be used in Beit3FeedForwardNetwork and Beit3Encoder.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability of the attention layer.
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability of the activation layer.
        sub_layernorm (`bool`, *optional*, defaults to `True`):
            Whether to apply sub layer norm
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        vocab_size (`int`, *optional*, defaults to 64010):
            Vocabulary size of the BEiT3 model. Defines the number of different image tokens that can be used during
            pre-training
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        logit_scale_init_value (`float`, *optional*, defaults to 2.65926):
            The inital value of the *logit_scale* parameter. Default is used as per the original CLIP implementation.

    Example:

    ```python
    >>> from transformers import Beit3Config, Beit3Model

    >>> # Initializing a BEiT3 beit3-base-patch16-224-pt22k style configuration
    >>> configuration = Beit3Config()

    >>> # Initializing a model (with random weights) from the beit3-base-patch16-224-pt22k style configuration
    >>> model = Beit3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "beit3"

    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        num_hidden_layers=12,
        encoder_normalize_before=False,
        activation_fn="gelu",
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        sub_layernorm=True,
        max_position_embeddings=1024,
        layer_norm_eps=1e-5,
        vocab_size=64010,
        image_size=224,
        patch_size=16,
        num_channels=3,
        initializer_range=0.02,
        logit_scale_init_value=2.65926,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.encoder_normalize_before = encoder_normalize_before
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.sub_layernorm = sub_layernorm
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        # Text
        self.vocab_size = vocab_size
        # Vision
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.logit_scale_init_value = logit_scale_init_value
        self.layer_norm_eps = layer_norm_eps


__all__ = ["Beit3Config"]
