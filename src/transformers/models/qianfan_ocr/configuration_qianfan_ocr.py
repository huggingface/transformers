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


@auto_docstring(checkpoint="baidu/Qianfan-OCR")
@strict
class QianfanOCRVisionConfig(PreTrainedConfig):
    r"""
    drop_path_rate (`float`, *optional*, defaults to 0.1):
        Dropout rate for stochastic depth.
    projection_dropout (`float`, *optional*, defaults to 0.0):
        Dropout probability for the projection layer.
    norm_type (`str`, *optional*, defaults to `"layer_norm"`):
        The type of normalization to use in the encoder. Can be `"layer_norm"` or `"rms_norm"`.
    use_mean_pooling (`bool`, *optional*, defaults to `True`):
        Whether to mean pool the final hidden states of the patches instead of using the final hidden state of the
        CLS token, before applying the classification head.

    Example:

    ```python
    >>> from transformers import QianfanOCRVisionConfig

    >>> configuration = QianfanOCRVisionConfig()
    >>> configuration.hidden_size
    1024
    ```"""

    model_type = "qianfan_ocr_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    attention_bias: bool = True
    use_qk_norm: bool = False
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_dropout: float | int = 0.0
    projection_dropout: float | int = 0.0
    initializer_range: float = 0.02
    norm_type: str = "layer_norm"
    layer_norm_eps: float = 1e-06
    image_size: int | list[int] | tuple[int, ...] = (448, 448)
    patch_size: int | list[int] | tuple[int, ...] = (14, 14)
    num_channels: int = 3
    use_absolute_position_embeddings: bool = True
    layer_scale_init_value: float = 0.1
    use_mean_pooling: bool = True
    drop_path_rate: float = 0.1

    def __post_init__(self, **kwargs):
        self.image_size = (
            self.image_size if isinstance(self.image_size, (list, tuple)) else (self.image_size, self.image_size)
        )
        self.patch_size = (
            self.patch_size if isinstance(self.patch_size, (list, tuple)) else (self.patch_size, self.patch_size)
        )
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="baidu/Qianfan-OCR")
@strict
class QianfanOCRConfig(PreTrainedConfig):
    r"""
    force_image_size (`int`, *optional*, defaults to 448):
        Force all images to this resolution before patching.
    dynamic_image_size (`bool`, *optional*, defaults to `True`):
        Whether to dynamically tile images into multiple patches.
    ps_version (`str`, *optional*, defaults to `"v2"`):
        Pixel shuffle version. Can be `"v1"` or `"v2"`.
    downsample_ratio (`float`, *optional*, defaults to 0.5):
        Factor by which to downsample the image.

    Example:

    ```python
    >>> from transformers import QianfanOCRConfig

    >>> configuration = QianfanOCRConfig()
    >>> configuration.downsample_ratio
    0.5
    ```"""

    model_type = "qianfan_ocr"
    sub_configs = {"text_config": AutoConfig, "vision_config": QianfanOCRVisionConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151667
    image_seq_length: int = 256
    downsample_ratio: float = 0.5
    projector_hidden_act: str = "gelu"
    vision_feature_layer: int | list[int] = -1
    vision_feature_select_strategy: str = "default"
    tie_word_embeddings: bool = False
    force_image_size: int | None = 448
    dynamic_image_size: bool = True
    ps_version: str = "v2"

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = QianfanOCRVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = QianfanOCRVisionConfig()

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen3")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"]()

        super().__post_init__(**kwargs)


__all__ = ["QianfanOCRVisionConfig", "QianfanOCRConfig"]
