# Copyright 2022 Meta Platforms authors and The HuggingFace Team. All rights reserved.
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
"""FLAVA model configurations"""

from typing import Any

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/flava-full")
@strict
class FlavaImageConfig(PreTrainedConfig):
    r"""
    mask_token (`bool`, *optional*, defaults to `True`):
        Whether to use a mask token or not. Used in MIM (Masked Image Modeling) loss for FLAVA.

    Example:

    ```python
    >>> from transformers import FlavaImageConfig, FlavaImageModel

    >>> # Initializing a FlavaImageModel with  style configuration
    >>> configuration = FlavaImageConfig()

    >>> # Initializing a FlavaImageModel model (with random weights) from the style configuration
    >>> model = FlavaImageModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "flava_image_model"
    base_config_key = "image_config"

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    qkv_bias: bool = True
    mask_token: bool = True
    vocab_size: int = 8192


@auto_docstring(checkpoint="facebook/flava-full")
@strict
class FlavaTextConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import FlavaTextConfig, FlavaTextModel

    >>> # Initializing a FlavaTextModel with  style configuration
    >>> configuration = FlavaTextConfig()

    >>> # Initializing a FlavaTextModel model (with random weights) from the style configuration
    >>> model = FlavaTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "flava_text_model"
    base_config_key = "text_config"

    vocab_size: int = 30522
    type_vocab_size: int = 2
    max_position_embeddings: int = 512
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    qkv_bias: bool = True


@auto_docstring(checkpoint="facebook/flava-full")
@strict
class FlavaMultimodalConfig(PreTrainedConfig):
    r"""
    use_cls_token (`bool`, *optional*, defaults to `True`):
        Whether to use an extra CLS token for multimodal settings. Usually needed by the FLAVA model.

    Example:

    ```python
    >>> from transformers import FlavaMultimodalConfig, FlavaMultimodalModel

    >>> # Initializing a FlavaMultimodalModel with  style configuration
    >>> configuration = FlavaMultimodalConfig()

    >>> # Initializing a FlavaMultimodalModel model (with random weights) from the style configuration
    >>> model = FlavaMultimodalModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "flava_multimodal_model"
    base_config_key = "multimodal_config"

    hidden_size: int = 768
    num_hidden_layers: int = 6
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    qkv_bias: bool = True
    use_cls_token: bool = True


@auto_docstring(checkpoint="facebook/flava-full")
@strict
class FlavaImageCodebookConfig(PreTrainedConfig):
    r"""
    num_groups (`int`, *optional*, defaults to 4):
        Number of groups to be created. This parameter as of now doesn't affect the model and is used for some
        internal calculation and estimations.
    num_blocks_per_group (`int`, *optional*, defaults to 2):
        Number of conv-based blocks per group.
    freeze (`bool`, defaults to `True`):
        Whether to freeze the weights of the model.

    Example:

    ```python
    >>> from transformers import FlavaImageCodebookConfig, FlavaImageCodebook

    >>> # Initializing a FlavaImageCodebook with style configuration
    >>> configuration = FlavaImageCodebookConfig()

    >>> # Initializing a FlavaImageCodebook model (with random weights) from the style configuration
    >>> model = FlavaImageCodebook(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    num_groups: int = 4
    input_channels: int = 3
    num_blocks_per_group: int = 2
    hidden_size: int = 256
    vocab_size: int = 8192
    freeze: bool = True
    initializer_range: float = 0.02


@auto_docstring(checkpoint="facebook/flava-full")
@strict
class FlavaConfig(PreTrainedConfig):
    r"""
    image_config (`dict`, *optional*):
        Dictionary of configuration options used to initialize [`FlavaImageConfig`].
    multimodal_config (`dict`, *optional*):
        Dictionary of configuration options used to initialize [`FlavaMultimodalConfig`].
    image_codebook_config (`dict`, *optional*):
        Dictionary of configuration options used to initialize [`FlavaCodebookConfig`].
    init_codebook (`bool`, *optional*, defaults to `True`):
        Whether to initialize the codebook
    logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
        The initial value of the *logit_scale* parameter. Default is used as per the original FLAVA/CLIP
        implementation.
    ce_ignore_index (`int`, *optional*, defaults to -100):
        Cross entropy index to ignore.
    mim_weight (`float`, *optional*, defaults to 1.0):
        Weight to be assigned to MIM (Masked Image Modeling) unimodal loss
    mlm_weight (`float`, *optional*, defaults to 1.0):
        Weight to be assigned to MLM (Masked Language Modeling) unimodal loss
    global_contrastive_weight (`float`, *optional*, defaults to 1.0):
        Weight to be assigned to global contrastive cross-alignment loss.
    itm_weight (`float`, *optional*, defaults to 1.0):
        Weight to be assigned to image-text matching multimodal loss.
    mmm_image_weight (`float`, *optional*, defaults to 1.0):
        Weight to be assigned to MMM loss's image part.
    mmm_text_weight (`float`, *optional*, defaults to 1.0):
        Weight to be assigned to MMM loss's text part.
    global_backprop_contrastive (`bool`, *optional*, defaults to `True`):
        Whether to use global backpropgation through all workers in contrastive loss.
    skip_unmasked_multimodal_encoder (`bool`, *optional*, defaults to `True`):
        Whether to skip running unmasked multimodal encoder whose outputs are not used by FLAVA losses.
    return_loss (`bool`, *optional*, defaults to `True`):
        Whether to return loss or not

    Example:

    ```python
    >>> from transformers import FlavaConfig, FlavaModel, FlavaForPreTraining

    >>> # Initializing a FlavaConfig with style configuration
    >>> configuration = FlavaConfig()

    >>> # Initializing a FlavaModel and FlavaForPreTraining model (with random weights) from the style configuration
    >>> model = FlavaModel(configuration)
    >>> model_pre = FlavaForPreTraining(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    >>> configuration_pre = model_pre.config
    ```
    """

    model_type = "flava"
    sub_configs = {
        "text_config": FlavaTextConfig,
        "image_config": FlavaImageConfig,
        "multimodal_config": FlavaMultimodalConfig,
        "image_codebook_config": FlavaImageCodebookConfig,
    }

    image_config: dict[str, Any] | PreTrainedConfig | None = None
    text_config: dict[str, Any] | PreTrainedConfig | None = None
    multimodal_config: dict[str, Any] | PreTrainedConfig | None = None
    image_codebook_config: dict[str, Any] | PreTrainedConfig | None = None
    hidden_size: int = 768
    layer_norm_eps: float = 1e-12
    projection_dim: int = 768
    init_codebook: bool = True
    logit_scale_init_value: float = 2.6592
    initializer_range: float = 0.02
    ce_ignore_index: int = -100
    mim_weight: float = 1.0
    mlm_weight: float = 1.0
    global_contrastive_weight: float = 1.0
    itm_weight: float = 1.0
    mmm_image_weight: float = 1.0
    mmm_text_weight: float = 1.0
    global_backprop_contrastive: bool = True
    skip_unmasked_multimodal_encoder: bool = True
    return_loss: bool = True
    tie_word_embeddings: bool = True
    initializer_factor: float = 1.0

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `FlavaTextConfig` with default values.")
        elif isinstance(self.text_config, FlavaTextConfig):
            text_config = self.text_config.to_dict()
        else:
            text_config = self.text_config

        if self.image_config is None:
            image_config = {}
            logger.info("`image_config` is `None`. initializing the `FlavaImageConfig` with default values.")
        elif isinstance(self.image_config, FlavaImageConfig):
            image_config = self.image_config.to_dict()
        else:
            image_config = self.image_config

        if self.multimodal_config is None:
            multimodal_config = {}
            logger.info("`multimodal_config` is `None`. Initializing the `FlavaMultimodalConfig` with default values.")
        elif isinstance(self.multimodal_config, FlavaMultimodalConfig):
            multimodal_config = self.multimodal_config.to_dict()
        else:
            multimodal_config = self.multimodal_config

        if self.image_codebook_config is None:
            image_codebook_config = {}
            logger.info(
                "`image_codebook_config` is `None`. initializing the `FlavaImageCodebookConfig` with default values."
            )
        elif isinstance(self.image_codebook_config, FlavaImageCodebookConfig):
            image_codebook_config = self.image_codebook_config.to_dict()
        else:
            image_codebook_config = self.image_codebook_config

        # If `_config_dict` exist, we use them for the backward compatibility.
        text_config_dict = kwargs.pop("text_config_dict", None)
        image_config_dict = kwargs.pop("image_config_dict", None)
        multimodal_config_dict = kwargs.pop("multimodal_config_dict", None)
        image_codebook_config_dict = kwargs.pop("image_codebook_config_dict", None)

        # Instead of simply assigning `[text|vision]_config_dict` to `[text|vision]_config`, we use the values in
        # `[text|vision]_config_dict` to update the values in `[text|vision]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        if text_config_dict is not None:
            # This is the complete result when using `text_config_dict`.
            _text_config_dict = FlavaTextConfig(**text_config_dict).to_dict()

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
                            f"`text_config_dict` is provided which will be used to initialize `FlavaTextConfig`. The "
                            f'value `text_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if image_config_dict is not None:
            # This is the complete result when using `image_config_dict`.
            _image_config_dict = FlavaImageConfig(**image_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _image_config_dict:
                _image_config_dict["id2label"] = {
                    str(key): value for key, value in _image_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_image_config_dict` and `image_config` but being different.
            for key, value in _image_config_dict.items():
                if key in image_config and value != image_config[key] and key != "transformers_version":
                    # If specified in `image_config_dict`
                    if key in image_config_dict:
                        message = (
                            f"`{key}` is found in both `image_config_dict` and `image_config` but with different "
                            f'values. The value `image_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`image_config_dict` is provided which will be used to initialize `FlavaImageConfig`. "
                            f'The value `image_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `image_config` with the ones in `_image_config_dict`.
            image_config.update(_image_config_dict)

        if multimodal_config_dict is not None:
            # This is the complete result when using `multimodal_config_dict`.
            _multimodal_config_dict = FlavaMultimodalConfig(**multimodal_config_dict).to_dict()

            # Give a warning if the values exist in both `_multimodal_config_dict` and `multimodal_config` but being
            # different.
            for key, value in _multimodal_config_dict.items():
                if key in multimodal_config and value != multimodal_config[key] and key != "transformers_version":
                    # If specified in `multimodal_config_dict`
                    if key in multimodal_config_dict:
                        message = (
                            f"`{key}` is found in both `multimodal_config_dict` and `multimodal_config` but with "
                            f'different values. The value `multimodal_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`multimodal_config_dict` is provided which will be used to initialize "
                            f'`FlavaMultimodalConfig`. The value `multimodal_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `multimodal_config` with the ones in `_multimodal_config_dict`.
            multimodal_config.update(_multimodal_config_dict)

        if image_codebook_config_dict is not None:
            # This is the complete result when using `image_codebook_config_dict`.
            _image_codebook_config_dict = FlavaImageCodebookConfig(**image_codebook_config_dict).to_dict()

            # Give a warning if the values exist in both `_image_codebook_config_dict` and `image_codebook_config` but
            # being different.
            for key, value in _image_codebook_config_dict.items():
                if (
                    key in image_codebook_config
                    and value != image_codebook_config[key]
                    and key != "transformers_version"
                ):
                    # If specified in `image_codebook_config_dict`
                    if key in image_codebook_config_dict:
                        message = (
                            f"`{key}` is found in both `image_codebook_config_dict` and `image_codebook_config` but "
                            f'with different values. The value `image_codebook_config_dict["{key}"]` will be used '
                            "instead."
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`image_codebook_config_dict` is provided which will be used to initialize "
                            f'`FlavaImageCodebookConfig`. The value `image_codebook_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `image_codebook_config` with the ones in `_image_codebook_config_dict`.
            image_codebook_config.update(_image_codebook_config_dict)

        # Finally we can convert back our unified text/vision configs to `PretrainedConfig`
        self.text_config = FlavaTextConfig(**text_config)
        self.image_config = FlavaImageConfig(**image_config)
        self.multimodal_config = FlavaMultimodalConfig(**multimodal_config)
        self.image_codebook_config = FlavaImageCodebookConfig(**image_codebook_config)

        super().__post_init__(**kwargs)


__all__ = ["FlavaConfig", "FlavaImageCodebookConfig", "FlavaImageConfig", "FlavaMultimodalConfig", "FlavaTextConfig"]
