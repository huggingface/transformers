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
"""X-CLIP model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/xclip-base-patch32")
@strict
class XCLIPTextConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import XCLIPTextModel, XCLIPTextConfig

    >>> # Initializing a XCLIPTextModel with microsoft/xclip-base-patch32 style configuration
    >>> configuration = XCLIPTextConfig()

    >>> # Initializing a XCLIPTextConfig from the microsoft/xclip-base-patch32 style configuration
    >>> model = XCLIPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xclip_text_model"
    base_config_key = "text_config"

    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 77
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2


@auto_docstring(checkpoint="microsoft/xclip-base-patch32")
@strict
class XCLIPVisionConfig(PreTrainedConfig):
    r"""
    mit_hidden_size (`int`, *optional*, defaults to 512):
        Dimensionality of the encoder layers of the Multiframe Integration Transformer (MIT).
    mit_intermediate_size (`int`, *optional*, defaults to 2048):
        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Multiframe Integration Transformer
        (MIT).
    mit_num_hidden_layers (`int`, *optional*, defaults to 1):
        Number of hidden layers in the Multiframe Integration Transformer (MIT).
    mit_num_attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads for each attention layer in the Multiframe Integration Transformer (MIT).
    num_frames (`int`, *optional*, defaults to 8):
        The number of frames in each video.

    Example:

    ```python
    >>> from transformers import XCLIPVisionModel, XCLIPVisionConfig

    >>> # Initializing a XCLIPVisionModel with microsoft/xclip-base-patch32 style configuration
    >>> configuration = XCLIPVisionConfig()

    >>> # Initializing a XCLIPVisionModel model from the microsoft/xclip-base-patch32 style configuration
    >>> model = XCLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xclip_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mit_hidden_size: int = 512
    mit_intermediate_size: int = 2048
    mit_num_hidden_layers: int = 1
    mit_num_attention_heads: int = 8
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 32
    num_frames: int = 8
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    drop_path_rate: float | int = 0.0


@auto_docstring(checkpoint="microsoft/xclip-base-patch32")
@strict
class XCLIPConfig(PreTrainedConfig):
    r"""
    prompt_layers (`int`, *optional*, defaults to 2):
        Number of layers in the video specific prompt generator.
    prompt_alpha (`float`, *optional*, defaults to 0.1):
        Alpha value to use in the video specific prompt generator.
    prompt_hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
        The non-linear activation function (function or string) in the video specific prompt generator. If string,
        `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
    prompt_num_attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads in the cross-attention of the video specific prompt generator.
    prompt_attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability for the attention layers in the video specific prompt generator.
    prompt_projection_dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability for the projection layers in the video specific prompt generator.
    """

    model_type = "xclip"
    sub_configs = {"text_config": XCLIPTextConfig, "vision_config": XCLIPVisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    projection_dim: int = 512
    prompt_layers: int = 2
    prompt_alpha: float = 0.1
    prompt_hidden_act: str = "quick_gelu"
    prompt_num_attention_heads: int = 8
    prompt_attention_dropout: float | int = 0.0
    prompt_projection_dropout: float | int = 0.0
    logit_scale_init_value: float = 2.6592
    initializer_factor: float = 1.0

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `XCLIPTextConfig` with default values.")
        elif isinstance(self.text_config, XCLIPTextConfig):
            text_config = self.text_config.to_dict()
        else:
            text_config = self.text_config

        if self.vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `XCLIPVisionConfig` with default values.")
        elif isinstance(self.vision_config, XCLIPVisionConfig):
            vision_config = self.vision_config.to_dict()
        else:
            vision_config = self.vision_config

        # For backward compatibility check keyword args
        # Instead of simply assigning `[text|vision]_config_dict` to `[text|vision]_config`, we use the values in
        # `[text|vision]_config_dict` to update the values in `[text|vision]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)

        if text_config_dict is not None:
            # This is the complete result when using `text_config_dict`.
            _text_config_dict = XCLIPTextConfig(**text_config_dict).to_dict()

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
                            f"`text_config_dict` is provided which will be used to initialize `XCLIPTextConfig`. The "
                            f'value `text_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = XCLIPVisionConfig(**vision_config_dict).to_dict()
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
                            f"`vision_config_dict` is provided which will be used to initialize `XCLIPVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `vision_config` with the ones in `_vision_config_dict`.
            vision_config.update(_vision_config_dict)

        # Finally we can convert back our unified text/vision configs to `PretrainedConfig`
        self.text_config = XCLIPTextConfig(**text_config)
        self.vision_config = XCLIPVisionConfig(**vision_config)

        super().__post_init__(**kwargs)


__all__ = ["XCLIPConfig", "XCLIPTextConfig", "XCLIPVisionConfig"]
