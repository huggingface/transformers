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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/xclip-base-patch32")
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

    def __init__(
        self,
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout


@auto_docstring(checkpoint="microsoft/xclip-base-patch32")
class XCLIPVisionConfig(PreTrainedConfig):
    r"""
    num_frames (`int`, *optional*, defaults to 8):
        The number of frames in each video.
    mit_hidden_size (`int`, *optional*, defaults to 512):
        Dimensionality of the encoder layers of the Multiframe Integration Transformer (MIT).
    mit_intermediate_size (`int`, *optional*, defaults to 2048):
        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Multiframe Integration Transformer
        (MIT).
    mit_num_hidden_layers (`int`, *optional*, defaults to 1):
        Number of hidden layers in the Multiframe Integration Transformer (MIT).
    mit_num_attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads for each attention layer in the Multiframe Integration Transformer (MIT).

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

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        mit_hidden_size=512,
        mit_intermediate_size=2048,
        mit_num_hidden_layers=1,
        mit_num_attention_heads=8,
        num_channels=3,
        image_size=224,
        patch_size=32,
        num_frames=8,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        drop_path_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mit_hidden_size = mit_hidden_size
        self.mit_intermediate_size = mit_intermediate_size
        self.mit_num_hidden_layers = mit_num_hidden_layers
        self.mit_num_attention_heads = mit_num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.drop_path_rate = drop_path_rate


@auto_docstring(checkpoint="microsoft/xclip-base-patch32")
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

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=512,
        prompt_layers=2,
        prompt_alpha=0.1,
        prompt_hidden_act="quick_gelu",
        prompt_num_attention_heads=8,
        prompt_attention_dropout=0.0,
        prompt_projection_dropout=0.0,
        logit_scale_init_value=2.6592,
        **kwargs,
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
            if vision_config is None:
                vision_config = {}

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

        if text_config is None:
            text_config = XCLIPTextConfig()
            logger.info("`text_config` is `None`. initializing the `XCLIPTextConfig` with default values.")
        elif isinstance(text_config, dict):
            text_config = XCLIPTextConfig(**text_config)

        if vision_config is None:
            vision_config = XCLIPVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `XCLIPVisionConfig` with default values.")
        elif isinstance(vision_config, dict):
            vision_config = XCLIPVisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config

        self.projection_dim = projection_dim
        self.prompt_layers = prompt_layers
        self.prompt_alpha = prompt_alpha
        self.prompt_hidden_act = prompt_hidden_act
        self.prompt_num_attention_heads = prompt_num_attention_heads
        self.prompt_attention_dropout = prompt_attention_dropout
        self.prompt_projection_dropout = prompt_projection_dropout
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

        super().__init__(**kwargs)


__all__ = ["XCLIPConfig", "XCLIPTextConfig", "XCLIPVisionConfig"]
