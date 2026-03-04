# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""VilT model configuration"""

from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class ViltConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViLTModel`]. It is used to instantiate an ViLT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViLT
    [dandelin/vilt-b32-mlm](https://huggingface.co/dandelin/vilt-b32-mlm) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the text part of the model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`ViltModel`].
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`ViltModel`]. This is used when encoding
            text.
        modality_type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the modalities passed when calling [`ViltModel`]. This is used after concatenating the
            embeddings of the text and image modalities.
        max_position_embeddings (`int`, *optional*, defaults to 40):
            The maximum sequence length that this model might ever be used with.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 384):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        max_image_length (`int`, *optional*, defaults to -1):
            The maximum number of patches to take as input for the Transformer encoder. If set to a positive integer,
            the encoder will sample `max_image_length` patches at maximum. If set to -1, will not be taken into
            account.
        num_images (`int`, *optional*, defaults to -1):
            The number of images to use for natural language visual reasoning. If set to a positive integer, will be
            used by [`ViltForImagesAndTextClassification`] for defining the classifier head.

    Example:

    ```python
    >>> from transformers import ViLTModel, ViLTConfig

    >>> # Initializing a ViLT dandelin/vilt-b32-mlm style configuration
    >>> configuration = ViLTConfig()

    >>> # Initializing a model from the dandelin/vilt-b32-mlm style configuration
    >>> model = ViLTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vilt"

    vocab_size: int = 30522
    type_vocab_size: int = 2
    modality_type_vocab_size: int = 2
    max_position_embeddings: int = 40
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    image_size: int | list[int] | tuple[int, int] = 384
    patch_size: int | list[int] | tuple[int, int] = 32
    num_channels: int = 3
    qkv_bias: bool = True
    max_image_length: int = -1
    tie_word_embeddings: bool = True
    num_images: int = -1
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        kwargs.pop("tie_word_embeddings", None)
        self.tie_word_embeddings = True  # force it
        super().__post_init__(**kwargs)


__all__ = ["ViltConfig"]
