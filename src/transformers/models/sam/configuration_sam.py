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
"""SAM model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/sam-vit-huge")
@strict
class SamPromptEncoderConfig(PreTrainedConfig):
    r"""
    mask_input_channels (`int`, *optional*, defaults to 16):
        The number of channels to be fed to the `MaskDecoder` module.
    num_point_embeddings (`int`, *optional*, defaults to 4):
        The number of point embeddings to be used.
    """

    base_config_key = "prompt_encoder_config"

    hidden_size: int = 256
    image_size: int | list[int] | tuple[int, int] = 1024
    patch_size: int | list[int] | tuple[int, int] = 16
    mask_input_channels: int = 16
    num_point_embeddings: int = 4
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6

    def __post_init__(self, **kwargs):
        self.image_embedding_size = self.image_size // self.patch_size
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="facebook/sam-vit-huge")
@strict
class SamMaskDecoderConfig(PreTrainedConfig):
    r"""
    mlp_dim (`int`, *optional*, defaults to 2048):
        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    attention_downsample_rate (`int`, *optional*, defaults to 2):
        The downsampling rate of the attention layer.
    num_multimask_outputs (`int`, *optional*, defaults to 3):
        The number of outputs from the `SamMaskDecoder` module. In the Segment Anything paper, this is set to 3.
    iou_head_depth (`int`, *optional*, defaults to 3):
        The number of layers in the IoU head module.
    iou_head_hidden_dim (`int`, *optional*, defaults to 256):
        The dimensionality of the hidden states in the IoU head module.
    """

    base_config_key = "mask_decoder_config"

    hidden_size: int = 256
    hidden_act: str = "relu"
    mlp_dim: int = 2048
    num_hidden_layers: int = 2
    num_attention_heads: int = 8
    attention_downsample_rate: int = 2
    num_multimask_outputs: int = 3
    iou_head_depth: int = 3
    iou_head_hidden_dim: int = 256
    layer_norm_eps: float = 1e-6


@auto_docstring(checkpoint="facebook/sam-vit-huge")
@strict
class SamVisionConfig(PreTrainedConfig):
    r"""
    output_channels (`int`, *optional*, defaults to 256):
        Dimensionality of the output channels in the Patch Encoder.
    use_rel_pos (`bool`, *optional*, defaults to `True`):
        Whether to use relative position embedding.
    window_size (`int`, *optional*, defaults to 14):
        Window size for relative position.
    global_attn_indexes (`list[int]`, *optional*, defaults to `[2, 5, 8, 11]`):
        The indexes of the global attention layers.
    num_pos_feats (`int`, *optional*, defaults to 128):
        The dimensionality of the position embedding.
    mlp_dim (`int`, *optional*):
        The dimensionality of the MLP layer in the Transformer encoder. If `None`, defaults to `mlp_ratio *
        hidden_size`.

    Example:

    ```python
    >>> from transformers import (
    ...     SamVisionConfig,
    ...     SamVisionModel,
    ... )

    >>> # Initializing a SamVisionConfig with `"facebook/sam-vit-huge"` style configuration
    >>> configuration = SamVisionConfig()

    >>> # Initializing a SamVisionModel (with random weights) from the `"facebook/sam-vit-huge"` style configuration
    >>> model = SamVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    base_config_key = "vision_config"
    model_type = "sam_vision_model"

    hidden_size: int = 768
    output_channels: int = 256
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 1024
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-06
    attention_dropout: float | int = 0.0
    initializer_range: float = 1e-10
    qkv_bias: bool = True
    mlp_ratio: float = 4.0
    use_abs_pos: bool = True
    use_rel_pos: bool = True
    window_size: int = 14
    global_attn_indexes: list[int] | tuple[int, ...] = (2, 5, 8, 11)
    num_pos_feats: int = 128
    mlp_dim: int | None = None

    def __post_init__(self, **kwargs):
        self.mlp_dim = int(self.hidden_size * self.mlp_ratio) if self.mlp_dim is None else self.mlp_dim
        self.scale = self.hidden_size // 2
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="facebook/sam-vit-huge")
@strict
class SamConfig(PreTrainedConfig):
    r"""
    prompt_encoder_config (Union[`dict`, `SamPromptEncoderConfig`], *optional*):
        Dictionary of configuration options used to initialize [`SamPromptEncoderConfig`].
    mask_decoder_config (Union[`dict`, `SamMaskDecoderConfig`], *optional*):
        Dictionary of configuration options used to initialize [`SamMaskDecoderConfig`].

    Example:

    ```python
    >>> from transformers import (
    ...     SamVisionConfig,
    ...     SamPromptEncoderConfig,
    ...     SamMaskDecoderConfig,
    ...     SamModel,
    ... )

    >>> # Initializing a SamConfig with `"facebook/sam-vit-huge"` style configuration
    >>> configuration = SamConfig()

    >>> # Initializing a SamModel (with random weights) from the `"facebook/sam-vit-huge"` style configuration
    >>> model = SamModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a SamConfig from a SamVisionConfig, SamPromptEncoderConfig, and SamMaskDecoderConfig

    >>> # Initializing SAM vision, SAM Q-Former and language model configurations
    >>> vision_config = SamVisionConfig()
    >>> prompt_encoder_config = SamPromptEncoderConfig()
    >>> mask_decoder_config = SamMaskDecoderConfig()

    >>> config = SamConfig(vision_config, prompt_encoder_config, mask_decoder_config)
    ```"""

    model_type = "sam"
    sub_configs = {
        "prompt_encoder_config": SamPromptEncoderConfig,
        "mask_decoder_config": SamMaskDecoderConfig,
        "vision_config": SamVisionConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    prompt_encoder_config: dict | PreTrainedConfig | None = None
    mask_decoder_config: dict | PreTrainedConfig | None = None
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = SamVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = SamVisionConfig()

        if isinstance(self.prompt_encoder_config, dict):
            self.prompt_encoder_config = SamPromptEncoderConfig(**self.prompt_encoder_config)
        elif self.prompt_encoder_config is None:
            self.prompt_encoder_config = SamPromptEncoderConfig()

        if isinstance(self.mask_decoder_config, dict):
            self.mask_decoder_config = SamMaskDecoderConfig(**self.mask_decoder_config)
        elif self.mask_decoder_config is None:
            self.mask_decoder_config = SamMaskDecoderConfig()

        super().__post_init__(**kwargs)


__all__ = ["SamConfig", "SamMaskDecoderConfig", "SamPromptEncoderConfig", "SamVisionConfig"]
