# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Fast model configuration"""

from transformers import CONFIG_MAPPING, PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class FastConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastForSceneTextRecognition`]. It is used to
    instantiate a FastForSceneTextRecognition model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastForSceneTextRecognition.
    [jadechoghari/fast-tiny](https://huggingface.co/jadechoghari/fast-tiny)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*):
            The configuration of the backbone model. Only used in case `use_timm_backbone` is set to `False` in which
            case it will default to `ResNetConfig()`.
        neck_in_channels (`List[int]`, *optional*, defaults to `[64, 128, 256, 512]`):
            Denotes the in channels of FastRepConvLayer in neck module.
        neck_out_channels (`List[int]`, *optional*, defaults to `[128, 128, 128, 128]`):
            Denotes the number of output channels for each FastRepConvLayer in the neck module.
            This list should be of the same length as `neck_in_channels`.
        neck_kernel_size (`List[Tuple[int, int]]`, *optional*, defaults to `[(3, 3), (3, 3), (3, 3), (3, 3)]`):
            Denotes the kernel size for each FastRepConvLayer in the neck module.
            This list should be of the same length as `neck_in_channels`.
            Each element is a tuple of two integers, specifying the height and width of the kernel.
        neck_stride (`List[Tuple[int, int]]`, *optional*, defaults to `[(1, 1), (1, 1), (1, 1), (1, 1)]`):
            Denotes the stride for each FastRepConvLayer in the neck module.
            This list should be of the same length as `neck_in_channels`.
            Each element is a tuple of two integers, specifying the stride along the height and width.
        head_pooling_size (`int`, *optional*, defaults to 9):
            Denotes the pooling size for the head layer. This integer specifies the size of the pooling window.
        head_dropout_ratio (`float`, *optional*, defaults to `0.0`):
            Denotes the dropout ratio used in the dropout layer of the head. Should be a float between 0 and 1, where 0 means no dropout and 1 means full dropout.
        head_conv_in_channels (`int`, *optional*, defaults to 512):
            Denotes the in channels of first conv layer in head layer.
        head_conv_out_channels (`int`, *optional*, defaults to 128):
            Denotes the out channels of first conv layer in head layer.
        head_conv_kernel_size (`List[int]`, *optional*, defaults to `[3, 3]`):
            Denotes the conv kernel size of first conv layer in head layer.
        head_conv_stride (`int`, *optional*, defaults to 1):
            Denotes the conv stride of first conv layer in head layer.
        head_final_kernel_size (`int`, *optional*, defaults to 1):
            Denotes the conv kernel size of final conv layer in head layer.
        head_final_stride (`int`, *optional*, defaults to 1):
            Denotes the conv stride of final conv layer in head layer.
        head_final_bias (`bool`, *optional*, defaults to `False`):
            Denotes the conv bais of final conv layer in head layer.
        head_final_in_channels (`int`, *optional*, defaults to 128):
            Denotes the in channels of final conv layer in head layer.
        head_final_out_channels (`int`, *optional*, defaults to 5):
            Denotes the out channels of final conv layer in head layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for normal distribution weight initialization, with a mean of 0.0.


    Examples:

    ```python
    >>> from transformers import FastConfig, FastForSceneTextRecognition

    >>> # Initializing a Fast Config
    >>> configuration = FastConfig()

    >>> # Initializing a model (with random weights)
    >>> model = FastForSceneTextRecognition(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        backbone_config=None,
        neck_in_channels=None,
        neck_out_channels=None,
        neck_kernel_size=None,
        neck_stride=None,
        head_pooling_size=9,
        head_dropout_ratio=0,
        head_conv_in_channels=512,
        head_conv_out_channels=128,
        head_conv_kernel_size=[3, 3],
        head_conv_stride=1,
        head_final_kernel_size=1,
        head_final_stride=1,
        head_final_bias=False,
        head_final_in_channels=128,
        head_final_out_channels=5,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if neck_in_channels is None:
            neck_in_channels = [64, 128, 256, 512]
        if neck_out_channels is None:
            neck_out_channels = [128, 128, 128, 128]
        if neck_kernel_size is None:
            neck_kernel_size = [[3, 3], [3, 3], [3, 3], [3, 3]]
        if neck_stride is None:
            neck_stride = [1, 1, 1, 1]

        if backbone_config is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `TextNet` backbone.")
            backbone_config = CONFIG_MAPPING["textnet"](
                out_features=["stage1", "stage2", "stage3", "stage4"],
                attn_implementation="eager",
            )

        elif isinstance(backbone_config, dict):
            backbone_config.setdefault("attn_implementation", "eager")
            backbone_model_type = backbone_config.get("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        else:
            if hasattr(backbone_config, "attn_implementation") and backbone_config.attn_implementation is None:
                backbone_config.attn_implementation = "eager"

        self.backbone_config = backbone_config

        self._attn_implementation = "eager"

        if isinstance(backbone_config, PretrainedConfig):
            if not hasattr(backbone_config, "_attn_implementation") or backbone_config._attn_implementation is None:
                backbone_config._attn_implementation = "eager"

        self.neck_in_channels = neck_in_channels
        self.neck_out_channels = neck_out_channels
        self.neck_kernel_size = neck_kernel_size
        self.neck_stride = neck_stride

        self.head_pooling_size = head_pooling_size
        self.head_dropout_ratio = head_dropout_ratio

        self.head_conv_in_channels = head_conv_in_channels
        self.head_conv_out_channels = head_conv_out_channels
        self.head_conv_kernel_size = head_conv_kernel_size
        self.head_conv_stride = head_conv_stride

        self.head_final_kernel_size = head_final_kernel_size
        self.head_final_stride = head_final_stride
        self.head_final_bias = head_final_bias
        self.head_final_in_channels = head_final_in_channels
        self.head_final_out_channels = head_final_out_channels

        self.initializer_range = initializer_range

    def __setattr__(self, name, value):
        if name == "_attn_implementation":
            object.__setattr__(self, name, value)
            for v in self.__dict__.values():
                if isinstance(v, PretrainedConfig):
                    setattr(v, "_attn_implementation", value)
        else:
            super().__setattr__(name, value)

    @classmethod
    def from_backbone_config(cls, backbone_config: PretrainedConfig, **kwargs):
        """Instantiate a [`FastConfig`] (or a derived class) from a pre-trained backbone model configuration.

        Args:
            backbone_config ([`PretrainedConfig`]):
                The backbone configuration.
        Returns:
            [`FastConfig`]: An instance of a configuration object
        """
        return cls(backbone_config=backbone_config, **kwargs)


__all__ = ["FastConfig"]
