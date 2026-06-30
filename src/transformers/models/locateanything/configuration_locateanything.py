# coding=utf-8
# Copyright 2026 NVIDIA and The HuggingFace Inc. team. All rights reserved.
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
"""LocateAnything model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="nvidia/LocateAnything-3B")
@strict
class LocateAnythingVisionConfig(PreTrainedConfig):
    r"""
    patch_size (`int`, *optional*, defaults to 14):
        The size (resolution) of each patch.
    init_pos_emb_height (`int`, *optional*, defaults to 64):
        The height of the learnable 2D positional embedding grid that is interpolated to the input resolution.
    init_pos_emb_width (`int`, *optional*, defaults to 64):
        The width of the learnable 2D positional embedding grid that is interpolated to the input resolution.
    num_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the MoonViT encoder.
    num_hidden_layers (`int`, *optional*, defaults to 27):
        Number of hidden layers in the MoonViT encoder.
    hidden_size (`int`, *optional*, defaults to 1152):
        Dimensionality of the encoder layers and the pooler layer.
    intermediate_size (`int`, *optional*, defaults to 4304):
        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the MoonViT encoder.
    merge_kernel_size (`tuple[int, int]`, *optional*, defaults to `(2, 2)`):
        The spatial kernel size used by the patch merger to group neighbouring patches before projection.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import LocateAnythingVisionConfig

    >>> # Initializing a MoonViT vision encoder configuration
    >>> configuration = LocateAnythingVisionConfig()
    ```"""

    model_type = "locateanything_vision"
    base_config_key = "vision_config"

    patch_size: int = 14
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    merge_kernel_size: tuple[int, int] | list[int] = (2, 2)
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.merge_kernel_size = tuple(self.merge_kernel_size)
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="nvidia/LocateAnything-3B")
@strict
class LocateAnythingConfig(PreTrainedConfig):
    r"""
    image_token_id (`int`, *optional*, defaults to 151665):
        The image placeholder token id that is replaced by visual features.
    block_size (`int`, *optional*, defaults to 6):
        Size of the structured output block used by Parallel Box Decoding (PBD). Each block encodes a complete
        bounding box / point unit and is predicted in parallel.
    causal_attn (`bool`, *optional*, defaults to `False`):
        Whether the tokens inside a Parallel Box Decoding block attend causally. When `False`, the block is
        bidirectional (the default block-diffusion behaviour).
    downsample_ratio (`float`, *optional*, defaults to 0.5):
        Factor by which to downsample the image features (kept for compatibility with the projector configuration).
    box_start_token_id (`int`, *optional*, defaults to 151668):
        Token id marking the beginning of a bounding-box block (`<box>`).
    box_end_token_id (`int`, *optional*, defaults to 151669):
        Token id marking the end of a bounding-box block (`</box>`).
    coord_start_token_id (`int`, *optional*, defaults to 151677):
        First token id of the contiguous range used to encode quantized coordinates.
    coord_end_token_id (`int`, *optional*, defaults to 152677):
        Last token id of the contiguous range used to encode quantized coordinates.
    ref_start_token_id (`int`, *optional*, defaults to 151672):
        Token id marking the beginning of a referring-expression block (`<ref>`).
    ref_end_token_id (`int`, *optional*, defaults to 151673):
        Token id marking the end of a referring-expression block (`</ref>`).
    none_token_id (`int`, *optional*, defaults to 4064):
        Token id used to denote an empty box (`none`).
    null_token_id (`int`, *optional*, defaults to 152678):
        Padding token id used to fill unused positions inside a Parallel Box Decoding block.
    switch_token_id (`int`, *optional*, defaults to 152679):
        Token id used to switch between Multi-Token Prediction and Auto-Regressive decoding.
    text_mask_token_id (`int`, *optional*, defaults to 151676):
        Mask token id appended to the sequence to query the next block during Multi-Token Prediction.
    tie_word_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to tie the input and output word embeddings.

    Example:

    ```python
    >>> from transformers import LocateAnythingConfig, LocateAnythingForConditionalGeneration

    >>> # Initializing a LocateAnything style configuration
    >>> configuration = LocateAnythingConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = LocateAnythingForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "locateanything"
    sub_configs = {"text_config": AutoConfig, "vision_config": LocateAnythingVisionConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151665
    block_size: int = 6
    causal_attn: bool = False
    downsample_ratio: float = 0.5
    box_start_token_id: int = 151668
    box_end_token_id: int = 151669
    coord_start_token_id: int = 151677
    coord_end_token_id: int = 152677
    ref_start_token_id: int = 151672
    ref_end_token_id: int = 151673
    none_token_id: int = 4064
    null_token_id: int = 152678
    switch_token_id: int = 152679
    text_mask_token_id: int = 151676
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = LocateAnythingVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = LocateAnythingVisionConfig()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()

        super().__post_init__(**kwargs)


__all__ = ["LocateAnythingVisionConfig", "LocateAnythingConfig"]
