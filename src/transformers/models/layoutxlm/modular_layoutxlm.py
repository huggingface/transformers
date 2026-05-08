# Copyright Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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


from huggingface_hub.dataclasses import strict

from ...utils import auto_docstring
from ..layoutlmv2.configuration_layoutlmv2 import LayoutLMv2Config


@auto_docstring(checkpoint="microsoft/layoutxlm-base")
@strict
class LayoutXLMConfig(LayoutLMv2Config):
    r"""
    max_2d_position_embeddings (`int`, *optional*, defaults to 1024):
        The maximum value that the 2D position embedding might ever be used with. Typically set this to something
        large just in case (e.g., 1024).
    max_rel_pos (`int`, *optional*, defaults to 128):
        The maximum number of relative positions to be used in the self-attention mechanism.
    rel_pos_bins (`int`, *optional*, defaults to 32):
        The number of relative position bins to be used in the self-attention mechanism.
    fast_qkv (`bool`, *optional*, defaults to `True`):
        Whether or not to use a single matrix for the queries, keys, values in the self-attention layers.
    max_rel_2d_pos (`int`, *optional*, defaults to 256):
        The maximum number of relative 2D positions in the self-attention mechanism.
    rel_2d_pos_bins (`int`, *optional*, defaults to 64):
        The number of 2D relative position bins in the self-attention mechanism.
    convert_sync_batchnorm (`bool`, *optional*, defaults to `True`):
        Whether or not to convert batch normalization layers to synchronized batch normalization layers.
    image_feature_pool_shape (`list[int]`, *optional*, defaults to `[7, 7, 256]`):
        The shape of the average-pooled feature map.
    coordinate_size (`int`, *optional*, defaults to 128):
        Dimension of the coordinate embeddings.
    shape_size (`int`, *optional*, defaults to 128):
        Dimension of the width and height embeddings.
    has_relative_attention_bias (`bool`, *optional*, defaults to `True`):
        Whether or not to use a relative attention bias in the self-attention mechanism.
    has_spatial_attention_bias (`bool`, *optional*, defaults to `True`):
        Whether or not to use a spatial attention bias in the self-attention mechanism.
    has_visual_segment_embedding (`bool`, *optional*, defaults to `False`):
        Whether or not to add visual segment embeddings.
    detectron2_config_args (`dict`, *optional*):
        Dictionary containing the configuration arguments of the Detectron2 visual backbone. Refer to [this
        file](https://github.com/microsoft/unilm/blob/master/layoutlmft/layoutlmft/models/layoutxlm/detectron2_config.py)
        for details regarding default values.

    Example:

    ```python
    >>> from transformers import LayoutXLMConfig, LayoutXLMModel

    >>> # Initializing a LayoutXLM microsoft/layoutxlm-base style configuration
    >>> configuration = LayoutXLMConfig()

    >>> # Initializing a model (with random weights) from the microsoft/layoutxlm-base style configuration
    >>> model = LayoutXLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    pass


__all__ = ["LayoutXLMConfig"]
