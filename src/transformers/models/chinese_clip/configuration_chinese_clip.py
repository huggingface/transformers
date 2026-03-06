# Copyright 2022 The OFA-Sys Team Authors and The HuggingFace Team. All rights reserved.
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
"""Chinese-CLIP model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="OFA-Sys/chinese-clip-vit-base-patch16")
class ChineseCLIPTextConfig(PreTrainedConfig):
    r"""
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling [`ChineseCLIPModel`].

    Example:

    ```python
    >>> from transformers import ChineseCLIPTextConfig, ChineseCLIPTextModel

    >>> # Initializing a ChineseCLIPTextConfig with OFA-Sys/chinese-clip-vit-base-patch16 style configuration
    >>> configuration = ChineseCLIPTextConfig()

    >>> # Initializing a ChineseCLIPTextModel (with random weights) from the OFA-Sys/chinese-clip-vit-base-patch16 style configuration
    >>> model = ChineseCLIPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "chinese_clip_text_model"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=30522,
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
        initializer_factor=1.0,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
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


@auto_docstring(checkpoint="OFA-Sys/chinese-clip-vit-base-patch16")
class ChineseCLIPVisionConfig(PreTrainedConfig):
    r"""
    Example:
    ```python
    >>> from transformers import ChineseCLIPVisionConfig, ChineseCLIPVisionModel

    >>> # Initializing a ChineseCLIPVisionConfig with OFA-Sys/chinese-clip-vit-base-patch16 style configuration
    >>> configuration = ChineseCLIPVisionConfig()

    >>> # Initializing a ChineseCLIPVisionModel (with random weights) from the OFA-Sys/chinese-clip-vit-base-patch16 style configuration
    >>> model = ChineseCLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "chinese_clip_vision_model"
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


@auto_docstring(checkpoint="OFA-Sys/chinese-clip-vit-base-patch16")
class ChineseCLIPConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import ChineseCLIPConfig, ChineseCLIPModel

    >>> # Initializing a ChineseCLIPConfig with OFA-Sys/chinese-clip-vit-base-patch16 style configuration
    >>> configuration = ChineseCLIPConfig()

    >>> # Initializing a ChineseCLIPModel (with random weights) from the OFA-Sys/chinese-clip-vit-base-patch16 style configuration
    >>> model = ChineseCLIPModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a ChineseCLIPConfig from a ChineseCLIPTextConfig and a ChineseCLIPVisionConfig

    >>> # Initializing a ChineseCLIPTextConfig and ChineseCLIPVisionConfig configuration
    >>> config_text = ChineseCLIPTextConfig()
    >>> config_vision = ChineseCLIPVisionConfig()

    >>> config = ChineseCLIPConfig(text_config=config_text, vision_config=config_vision)
    ```"""

    model_type = "chinese_clip"
    sub_configs = {"text_config": ChineseCLIPTextConfig, "vision_config": ChineseCLIPVisionConfig}

    def __init__(
        self, text_config=None, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs
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
            _text_config_dict = ChineseCLIPTextConfig(**text_config_dict).to_dict()

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
                            f"`text_config_dict` is provided which will be used to initialize `ChineseCLIPTextConfig`. "
                            f'The value `text_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}

            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = ChineseCLIPVisionConfig(**vision_config_dict).to_dict()
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
                            f"`vision_config_dict` is provided which will be used to initialize "
                            f'`ChineseCLIPVisionConfig`. The value `vision_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `vision_config` with the ones in `_vision_config_dict`.
            vision_config.update(_vision_config_dict)

        if text_config is None:
            text_config = ChineseCLIPTextConfig()
            logger.info("`text_config` is `None`. initializing the `ChineseCLIPTextConfig` with default values.")
        elif isinstance(text_config, dict):
            text_config = ChineseCLIPTextConfig(**text_config)

        if vision_config is None:
            vision_config = ChineseCLIPVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `ChineseCLIPVisionConfig` with default values.")
        elif isinstance(vision_config, dict):
            vision_config = ChineseCLIPVisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
        self.initializer_range = 0.02
        super().__init__(**kwargs)


__all__ = ["ChineseCLIPConfig", "ChineseCLIPTextConfig", "ChineseCLIPVisionConfig"]
