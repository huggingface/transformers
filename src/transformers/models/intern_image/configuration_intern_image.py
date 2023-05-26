# coding=utf-8
# Copyright 2022 OpenGVLab and The HuggingFace Inc. team. All rights reserved.
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
""" internimage model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

INTERN_IMAGE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "OpenGVLab/internimage": "https://huggingface.co/OpenGVLab/internimage/resolve/main/config.json",
    # See all internimage models at https://huggingface.co/models?filter=intern_image
}


class InternImageConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~InternImageModel`].
    It is used to instantiate an internimage model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the internimage [OpenGVLab/internimage](https://huggingface.co/OpenGVLab/internimage) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        core_op (`str`, *optional*, defaults to `"DCNv3_pytorch"`):
            Core operation used in the InternImageModel.
        channels (`int`, *optional*, defaults to 64):
            Number of channels in the InternImageModel.
        depths (`tuple`, *optional*, defaults to `(4, 4, 18, 4)`):
            Tuple specifying the depth of layers in the InternImageModel.
        groups (`tuple`, *optional*, defaults to `(4, 8, 16, 32)`):
            Tuple specifying the group of layers in the InternImageModel.
        num_classes (`int`, *optional*, defaults to 1000):
            Number of classes for the model output.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of mlp layers in the InternImageModel.
        drop_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate in the model.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            Dropout path rate in the model.
        drop_path_type (`str`, *optional*, defaults to `"linear"`):
            Type of dropout path used in the model.
        act_layer (`str`, *optional*, defaults to `"GELU"`):
            Activation function used in the model.
        norm_layer (`str`, *optional*, defaults to `"LN"`):
            Normalization layer used in the model.
        layer_scale (`str`, *optional*, defaults to `None`):
            Scale of the layers in the model.
        offset_scale (`float`, *optional*, defaults to 1.0):
            Offset scale in the model.
        post_norm (`bool`, *optional*, defaults to `False`):
            Whether to use post normalization in the model.
        cls_scale (`float`, *optional*, defaults to 1.5):
            Scale of the classification layer in the model.
        with_cp (`bool`, *optional*, defaults to `False`):
            Whether to use checkpointing in the model.
        Example:

    ```python
    >>> from transformers import InternImageModel, InternImageConfig

    >>> # Initializing a internimage OpenGVLab/internimage style configuration
    >>> configuration = InternImageConfig()

    >>> # Initializing a model from the OpenGVLab/internimage style configuration
    >>> model = InternImageModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "intern_image"

    def __init__(
        self,
        core_op="DCNv3_pytorch",
        channels=64,
        depths=(4, 4, 18, 4),
        groups=(4, 8, 16, 32),
        num_classes=1000,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_path_type="linear",
        act_layer="GELU",
        norm_layer="LN",
        layer_scale=None,
        offset_scale=1.0,
        post_norm=False,
        cls_scale=1.5,
        with_cp=False,
        **kwargs,
    ):
        self.core_op = core_op
        self.channels = channels
        self.depths = depths
        self.groups = groups
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.drop_path_type = drop_path_type
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.layer_scale = layer_scale
        self.offset_scale = offset_scale
        self.post_norm = post_norm
        self.cls_scale = cls_scale
        self.with_cp = with_cp
        super().__init__(**kwargs)
