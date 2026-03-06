# Copyright 2022 WenXiang ZhongzhiCheng LedellWu LiuGuang BoWenZhang and The HuggingFace Inc. team. All rights reserved.
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
"""AltCLIP model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="BAAI/AltCLIP")
class AltCLIPTextConfig(PreTrainedConfig):
    r"""
    project_dim (`int`, *optional*, defaults to 768):
        The dimensions of the teacher model before the mapping layer.

    Examples:

    ```python
    >>> from transformers import AltCLIPTextModel, AltCLIPTextConfig

    >>> # Initializing a AltCLIPTextConfig with BAAI/AltCLIP style configuration
    >>> configuration = AltCLIPTextConfig()

    >>> # Initializing a AltCLIPTextModel (with random weights) from the BAAI/AltCLIP style configuration
    >>> model = AltCLIPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "altclip_text_model"

    def __init__(
        self,
        vocab_size=250002,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        initializer_range=0.02,
        initializer_factor=0.02,
        layer_norm_eps=1e-05,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        project_dim=768,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.layer_norm_eps = layer_norm_eps
        self.project_dim = project_dim


@auto_docstring(checkpoint="BAAI/AltCLIP")
class AltCLIPVisionConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import AltCLIPVisionConfig, AltCLIPVisionModel

    >>> # Initializing a AltCLIPVisionConfig with BAAI/AltCLIP style configuration
    >>> configuration = AltCLIPVisionConfig()

    >>> # Initializing a AltCLIPVisionModel (with random weights) from the BAAI/AltCLIP style configuration
    >>> model = AltCLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "altclip_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act


@auto_docstring(checkpoint="BAAI/AltCLIP")
class AltCLIPConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import AltCLIPConfig, AltCLIPModel

    >>> # Initializing a AltCLIPConfig with BAAI/AltCLIP style configuration
    >>> configuration = AltCLIPConfig()

    >>> # Initializing a AltCLIPModel (with random weights) from the BAAI/AltCLIP style configuration
    >>> model = AltCLIPModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a AltCLIPConfig from a AltCLIPTextConfig and a AltCLIPVisionConfig

    >>> # Initializing a AltCLIPText and AltCLIPVision configuration
    >>> config_text = AltCLIPTextConfig()
    >>> config_vision = AltCLIPVisionConfig()

    >>> config = AltCLIPConfig(text_config=config_text, vision_config=config_vision)
    ```"""

    model_type = "altclip"
    sub_configs = {"text_config": AltCLIPTextConfig, "vision_config": AltCLIPVisionConfig}

    def __init__(
        self, text_config=None, vision_config=None, projection_dim=768, logit_scale_init_value=2.6592, **kwargs
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)

        # Instead of simply assigning `[text|vision]_config_dict` to `[text|vision]_config`, we use the values in
        # `[text|vision]_config_dict` to update the values in `[text|vision]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        if text_config_dict is not None:
            if text_config is None:
                text_config = {}

            # This is the complete result when using `text_config_dict`.
            _text_config_dict = AltCLIPTextConfig(**text_config_dict).to_dict()

            # Give a warning if the values exist in both `_text_config_dict` and `text_config` but being different.
            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and key != "transformers_version":
                    # If specified in `text_config_dict`
                    if key in text_config_dict:
                        message = (
                            f"`{key}` is found in both `text_config_dict` and `text_config` but with different values. "
                            f'The value `text_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`text_config_dict` is provided which will be used to initialize `AltCLIPTextConfig`. The "
                            f'value `text_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}

            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = AltCLIPVisionConfig(**vision_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _vision_config_dict:
                _vision_config_dict["id2label"] = {
                    str(key): value for key, value in _vision_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_vision_config_dict` and `vision_config` but being different.
            for key, value in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and key != "transformers_version":
                    # If specified in `vision_config_dict`
                    if key in vision_config_dict:
                        message = (
                            f"`{key}` is found in both `vision_config_dict` and `vision_config` but with different "
                            f'values. The value `vision_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`vision_config_dict` is provided which will be used to initialize `AltCLIPVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `vision_config` with the ones in `_vision_config_dict`.
            vision_config.update(_vision_config_dict)

        if text_config is None:
            text_config = AltCLIPTextConfig()
            logger.info("`text_config` is `None`. Initializing the `AltCLIPTextConfig` with default values.")
        elif isinstance(text_config, dict):
            text_config = AltCLIPTextConfig(**text_config)

        if vision_config is None:
            vision_config = AltCLIPVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `AltCLIPVisionConfig` with default values.")
        elif isinstance(vision_config, dict):
            vision_config = AltCLIPVisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
        super().__init__(**kwargs)


__all__ = ["AltCLIPTextConfig", "AltCLIPVisionConfig", "AltCLIPConfig"]
