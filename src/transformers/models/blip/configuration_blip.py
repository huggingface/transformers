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
"""Blip model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Salesforce/blip-vqa-base")
@strict
class BlipTextConfig(PreTrainedConfig):
    r"""
    label_smoothing (float, *optional*):
        A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets
        become a mixture of the original ground truth and a uniform distribution as described in
        `Rethinking the Inception Architecture for Computer Vision <https://huggingface.co/papers/1512.00567>`__. Default: :math:`0.0`.

    Example:

    ```python
    >>> from transformers import BlipTextConfig, BlipTextModel

    >>> # Initializing a BlipTextConfig with Salesforce/blip-vqa-base style configuration
    >>> configuration = BlipTextConfig()

    >>> # Initializing a BlipTextModel (with random weights) from the Salesforce/blip-vqa-base style configuration
    >>> model = BlipTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "blip_text_model"
    base_config_key = "text_config"

    vocab_size: int = 30524
    hidden_size: int = 768
    encoder_hidden_size: int = 768
    intermediate_size: int = 3072
    projection_dim: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 512
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    bos_token_id: int | None = 30522
    eos_token_id: int | list[int] | None = 2
    pad_token_id: int | None = 0
    sep_token_id: int | None = 102
    is_decoder: bool = True
    use_cache: bool = True
    label_smoothing: float = 0.0


@auto_docstring(checkpoint="Salesforce/blip-vqa-base")
@strict
class BlipVisionConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import BlipVisionConfig, BlipVisionModel

    >>> # Initializing a BlipVisionConfig with Salesforce/blip-vqa-base style configuration
    >>> configuration = BlipVisionConfig()

    >>> # Initializing a BlipVisionModel (with random weights) from the Salesforce/blip-vqa-base style configuration
    >>> model = BlipVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "blip_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 768
    intermediate_size: int = 3072
    projection_dim: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    image_size: int | list[int] | tuple[int, int] = 384
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    initializer_range: float = 1e-10


@auto_docstring(checkpoint="Salesforce/blip-vqa-base")
@strict
class BlipConfig(PreTrainedConfig):
    r"""
    image_text_hidden_size (`int`, *optional*, defaults to 256):
        Dimensionality of the hidden state of the image-text fusion layer.
    label_smoothing (float, *optional*):
        A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets
        become a mixture of the original ground truth and a uniform distribution as described in
        `Rethinking the Inception Architecture for Computer Vision <https://huggingface.co/papers/1512.00567>`__. Default: :math:`0.0`.

    Example:

    ```python
    >>> from transformers import BlipConfig, BlipModel

    >>> # Initializing a BlipConfig with Salesforce/blip-vqa-base style configuration
    >>> configuration = BlipConfig()

    >>> # Initializing a BlipPModel (with random weights) from the Salesforce/blip-vqa-base style configuration
    >>> model = BlipModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a BlipConfig from a BlipTextConfig and a BlipVisionConfig

    >>> # Initializing a BLIPText and BLIPVision configuration
    >>> config_text = BlipTextConfig()
    >>> config_vision = BlipVisionConfig()

    >>> config = BlipConfig(text_config=config_text, vision_config=config_vision)
    ```"""

    model_type = "blip"
    sub_configs = {"text_config": BlipTextConfig, "vision_config": BlipVisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    projection_dim: int = 512
    logit_scale_init_value: float = 2.6592
    image_text_hidden_size: int = 256
    label_smoothing: float = 0.0
    tie_word_embeddings: bool = True
    initializer_factor: float = 1.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = BlipTextConfig()
            logger.info("`text_config` is `None`. Initializing the `BlipTextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = BlipTextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = BlipVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `BlipVisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = BlipVisionConfig(**self.vision_config)

        self.text_config.encoder_hidden_size = self.vision_config.hidden_size

        super().__post_init__(**kwargs)


__all__ = ["BlipConfig", "BlipTextConfig", "BlipVisionConfig"]
