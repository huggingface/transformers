# Copyright 2022 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""LayoutLMv3 model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/layoutlmv3-base")
class LayoutLMv3Config(PreTrainedConfig):
    r"""
    max_2d_position_embeddings (`int`, *optional*, defaults to 1024):
        The maximum value that the 2D position embedding might ever be used with. Typically set this to something
        large just in case (e.g., 1024).
    coordinate_size (`int`, *optional*, defaults to `128`):
        Dimension of the coordinate embeddings.
    shape_size (`int`, *optional*, defaults to `128`):
        Dimension of the width and height embeddings.
    has_relative_attention_bias (`bool`, *optional*, defaults to `True`):
        Whether or not to use a relative attention bias in the self-attention mechanism.
    rel_pos_bins (`int`, *optional*, defaults to 32):
        The number of relative position bins to be used in the self-attention mechanism.
    max_rel_pos (`int`, *optional*, defaults to 128):
        The maximum number of relative positions to be used in the self-attention mechanism.
    max_rel_2d_pos (`int`, *optional*, defaults to 256):
        The maximum number of relative 2D positions in the self-attention mechanism.
    rel_2d_pos_bins (`int`, *optional*, defaults to 64):
        The number of 2D relative position bins in the self-attention mechanism.
    has_spatial_attention_bias (`bool`, *optional*, defaults to `True`):
        Whether or not to use a spatial attention bias in the self-attention mechanism.
    text_embed (`bool`, *optional*, defaults to `True`):
        Whether or not to add text embeddings.
    visual_embed (`bool`, *optional*, defaults to `True`):
        Whether or not to add patch embeddings.
    input_size (`int`, *optional*, defaults to `224`):
        The size (resolution) of the images.

    Example:

    ```python
    >>> from transformers import LayoutLMv3Config, LayoutLMv3Model

    >>> # Initializing a LayoutLMv3 microsoft/layoutlmv3-base style configuration
    >>> configuration = LayoutLMv3Config()

    >>> # Initializing a model (with random weights) from the microsoft/layoutlmv3-base style configuration
    >>> model = LayoutLMv3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "layoutlmv3"

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        max_2d_position_embeddings=1024,
        coordinate_size=128,
        shape_size=128,
        has_relative_attention_bias=True,
        rel_pos_bins=32,
        max_rel_pos=128,
        rel_2d_pos_bins=64,
        max_rel_2d_pos=256,
        has_spatial_attention_bias=True,
        text_embed=True,
        visual_embed=True,
        input_size=224,
        num_channels=3,
        patch_size=16,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.has_relative_attention_bias = has_relative_attention_bias
        self.rel_pos_bins = rel_pos_bins
        self.max_rel_pos = max_rel_pos
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.max_rel_2d_pos = max_rel_2d_pos
        self.text_embed = text_embed
        self.visual_embed = visual_embed
        self.input_size = input_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.classifier_dropout = classifier_dropout


__all__ = ["LayoutLMv3Config"]
