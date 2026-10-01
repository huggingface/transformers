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
"""ALIGN model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="kakaobrain/align-base")
@strict
class AlignTextConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import AlignTextConfig, AlignTextModel

    >>> # Initializing a AlignTextConfig with kakaobrain/align-base style configuration
    >>> configuration = AlignTextConfig()

    >>> # Initializing a AlignTextModel (with random weights) from the kakaobrain/align-base style configuration
    >>> model = AlignTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "align_text_model"
    base_config_key = "text_config"

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None


@auto_docstring(checkpoint="kakaobrain/align-base")
@strict
class AlignVisionConfig(PreTrainedConfig):
    r"""
    width_coefficient (`float`, *optional*, defaults to 2.0):
        Scaling coefficient for network width at each stage.
    depth_coefficient (`float`, *optional*, defaults to 3.1):
        Scaling coefficient for network depth at each stage.
    depth_divisor (`int`, *optional*, defaults to 8):
        A unit of network width.
    kernel_sizes (`list[int]`, *optional*, defaults to `[3, 3, 5, 3, 5, 5, 3]`):
        List of kernel sizes to be used in each block.
    in_channels (`list[int]`, *optional*, defaults to `[32, 16, 24, 40, 80, 112, 192]`):
        List of input channel sizes to be used in each block for convolutional layers.
    out_channels (`list[int]`, *optional*, defaults to `[16, 24, 40, 80, 112, 192, 320]`):
        List of output channel sizes to be used in each block for convolutional layers.
    depthwise_padding (`list[int]`, *optional*, defaults to `[]`):
        List of block indices with square padding.
    strides (`list[int]`, *optional*, defaults to `[1, 2, 2, 2, 1, 2, 1]`):
        List of stride sizes to be used in each block for convolutional layers.
    num_block_repeats (`list[int]`, *optional*, defaults to `[1, 2, 2, 3, 3, 4, 1]`):
        List of the number of times each block is to repeated.
    expand_ratios (`list[int]`, *optional*, defaults to `[1, 6, 6, 6, 6, 6, 6]`):
        List of scaling coefficient of each block.
    squeeze_expansion_ratio (`float`, *optional*, defaults to 0.25):
        Squeeze expansion ratio.
    hidden_dim (`int`, *optional*, defaults to 1280):
        The hidden dimension of the layer before the classification head.
    pooling_type (`str` or `function`, *optional*, defaults to `"mean"`):
        Type of final pooling to be applied before the dense classification head. Available options are [`"mean"`,
        `"max"`]
    batch_norm_momentum (`float`, *optional*, defaults to 0.99):
        The momentum used by the batch normalization layers.
    drop_connect_rate (`float`, *optional*, defaults to 0.2):
        The drop rate for skip connections.

    Example:

    ```python
    >>> from transformers import AlignVisionConfig, AlignVisionModel

    >>> # Initializing a AlignVisionConfig with kakaobrain/align-base style configuration
    >>> configuration = AlignVisionConfig()

    >>> # Initializing a AlignVisionModel (with random weights) from the kakaobrain/align-base style configuration
    >>> model = AlignVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "align_vision_model"
    base_config_key = "vision_config"

    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 600
    width_coefficient: float = 2.0
    depth_coefficient: float = 3.1
    depth_divisor: int = 8
    kernel_sizes: list[int] | tuple[int, ...] = (3, 3, 5, 3, 5, 5, 3)
    in_channels: list[int] | tuple[int, ...] = (32, 16, 24, 40, 80, 112, 192)
    out_channels: list[int] | tuple[int, ...] = (16, 24, 40, 80, 112, 192, 320)
    depthwise_padding: list | tuple[int, ...] = ()
    strides: list[int] | tuple[int, ...] = (1, 2, 2, 2, 1, 2, 1)
    num_block_repeats: list[int] | tuple[int, ...] = (1, 2, 2, 3, 3, 4, 1)
    expand_ratios: list[int] | tuple[int, ...] = (1, 6, 6, 6, 6, 6, 6)
    squeeze_expansion_ratio: float = 0.25
    hidden_act: str = "swish"
    hidden_dim: int = 2560
    pooling_type: str = "mean"
    initializer_range: float = 0.02
    batch_norm_eps: float = 0.001
    batch_norm_momentum: float = 0.99
    drop_connect_rate: float | int = 0.2

    def __post_init__(self, **kwargs):
        self.num_hidden_layers = sum(self.num_block_repeats) * 4
        for attr in [
            "kernel_sizes",
            "in_channels",
            "out_channels",
            "depthwise_padding",
            "strides",
            "num_block_repeats",
            "expand_ratios",
        ]:
            # cast tuple so it can be JSON-ized when saving
            setattr(self, attr, list(getattr(self, attr)))
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="kakaobrain/align-base")
@strict
class AlignConfig(PreTrainedConfig):
    r"""
    temperature_init_value (`float`, *optional*, defaults to 1.0):
        The initial value of the *temperature* parameter. Default is used as per the original ALIGN implementation.

    Example:

    ```python
    >>> from transformers import AlignConfig, AlignModel

    >>> # Initializing a AlignConfig with kakaobrain/align-base style configuration
    >>> configuration = AlignConfig()

    >>> # Initializing a AlignModel (with random weights) from the kakaobrain/align-base style configuration
    >>> model = AlignModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a AlignConfig from a AlignTextConfig and a AlignVisionConfig
    >>> from transformers import AlignTextConfig, AlignVisionConfig

    >>> # Initializing ALIGN Text and Vision configurations
    >>> config_text = AlignTextConfig()
    >>> config_vision = AlignVisionConfig()

    >>> config = AlignConfig(text_config=config_text, vision_config=config_vision)
    ```"""

    model_type = "align"
    sub_configs = {"text_config": AlignTextConfig, "vision_config": AlignVisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    projection_dim: int = 640
    temperature_init_value: float = 1.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = AlignTextConfig()
            logger.info("`text_config` is `None`. Initializing the `AlignTextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = AlignTextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = AlignVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `AlignVisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = AlignVisionConfig(**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = ["AlignTextConfig", "AlignVisionConfig", "AlignConfig"]
