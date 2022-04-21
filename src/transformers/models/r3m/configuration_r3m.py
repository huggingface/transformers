# coding=utf-8
# Copyright 2022 Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, Abhinav Gupta and The HuggingFace Inc. team. All rights reserved.
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
""" R3M model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

R3M_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "surajnair/r3m-50": "https://huggingface.co/surajnair/r3m-50/resolve/main/config.json",
    # See all R3M models at https://huggingface.co/models?filter=r3m
}


class R3MConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~R3MModel`].
    It is used to instantiate an R3M model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the R3M [surajnair/r3m-50](https://huggingface.co/surajnair/r3m-50) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        resnet_size (`int`, *optional*, defaults to 50):
            Size of the ResNet for the  R3M model. Either ResNet18, 34, or 50. 
        Example:

    ```python
    >>> from transformers import R3MModel, R3MConfig

    >>> # Initializing a R3M surajnair/r3m-50 style configuration
    >>> configuration = R3MConfig()

    >>> # Initializing a model from the surajnair/r3m-50 style configuration
    >>> model = R3MModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
"""
    model_type = "r3m"
    

    def __init__(
        self,
        resnet_size=50,
        **kwargs
    ):
        self.resnet_size = resnet_size
        super().__init__(
            **kwargs
        )

    