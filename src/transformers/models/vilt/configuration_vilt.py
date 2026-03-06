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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="dandelin/vilt-b32-mlm")
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

    def __init__(
        self,
        vocab_size=30522,
        type_vocab_size=2,
        modality_type_vocab_size=2,
        max_position_embeddings=40,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=384,
        patch_size=32,
        num_channels=3,
        qkv_bias=True,
        max_image_length=-1,
        tie_word_embeddings=True,
        num_images=-1,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.modality_type_vocab_size = modality_type_vocab_size
        self.max_position_embeddings = max_position_embeddings

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.max_image_length = max_image_length
        self.num_images = num_images
        self.tie_word_embeddings = True  # force it


__all__ = ["ViltConfig"]
