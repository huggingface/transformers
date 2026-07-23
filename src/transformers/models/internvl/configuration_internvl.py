# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="OpenGVLab/InternVL3-1B-hf")
@strict
class InternVLVisionConfig(PreTrainedConfig):
    r"""
    projection_dropout (`float`, *optional*, defaults to 0.0):
        Dropout probability for the projection layer.
    norm_type (`str`, *optional*, defaults to `"layer_norm"`):
        The type of normalization to use in the encoder. Can be `"layer_norm"` or `"rms_norm"`.
    use_mask_token (`bool`, *optional*, defaults to `False`):
        Whether to use a mask token for masked image modeling
    use_mean_pooling (`bool`, *optional*, defaults to `True`):
        Whether to mean pool the final hidden states of the patches instead of using the final hidden state of the
        CLS token, before applying the classification head.

    Example:

    ```python
    >>> from transformers import InternVLVisionConfig, InternVLVisionModel

    >>> # Initializing a InternVLVisionModel OpenGVLab/InternVL3-1B-hf style configuration
    >>> configuration = InternVLVisionConfig()

    >>> # Initializing a model (with random weights) from the OpenGVLab/InternVL3-1B-hf configuration
    >>> model = InternVLVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "internvl_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    attention_bias: bool = False
    use_qk_norm: bool = False
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_dropout: float | int = 0.0
    projection_dropout: float | int = 0.0
    initializer_range: float = 0.02
    norm_type: str = "layer_norm"
    layer_norm_eps: float = 1e-06
    image_size: int | list[int] | tuple[int, ...] = (448, 448)
    patch_size: int | list[int] | tuple[int, ...] = (14, 14)
    num_channels: int = 3
    use_mask_token: bool = False
    use_absolute_position_embeddings: bool = True
    layer_scale_init_value: float = 0.1
    use_mean_pooling: bool = True

    def __post_init__(self, **kwargs):
        self.image_size = (
            self.image_size if isinstance(self.image_size, (list, tuple)) else (self.image_size, self.image_size)
        )
        self.patch_size = (
            self.patch_size if isinstance(self.patch_size, (list, tuple)) else (self.patch_size, self.patch_size)
        )
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="OpenGVLab/InternVL3-1B-hf")
@strict
class InternVLConfig(PreTrainedConfig):
    r"""
    downsample_ratio (`float`, *optional*, defaults to 0.5):
        Factor by which to downsample the image.

    Example:

    ```python
    >>> from transformers import InternVLForConditionalGeneration, InternVLConfig

    >>> # Initializing a InternVL style configuration
    >>> configuration = InternVLConfig()

    >>> # Initializing a model (with random weights) from the OpenGVLab/InternVL3-1B-hf configuration
    >>> model = InternVLForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "internvl"
    sub_configs = {"text_config": AutoConfig, "vision_config": InternVLVisionConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151667
    image_seq_length: int = 256
    downsample_ratio: float = 0.5
    projector_hidden_act: str = "gelu"
    vision_feature_layer: int | list[int] = -1
    vision_feature_select_strategy: str = "default"
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = InternVLVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = InternVLVisionConfig()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()

        super().__post_init__(**kwargs)


__all__ = ["InternVLVisionConfig", "InternVLConfig"]
