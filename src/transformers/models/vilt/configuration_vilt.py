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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="dandelin/vilt-b32-mlm")
@strict
class ViltConfig(PreTrainedConfig):
    r"""
    modality_type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the modalities passed when calling [`ViltModel`]. This is used after concatenating the
        embeddings of the text and image modalities.
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
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
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
