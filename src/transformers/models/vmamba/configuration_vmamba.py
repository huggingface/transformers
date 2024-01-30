# coding=utf-8
# Copyright 2022 Derk Mus and The HuggingFace Inc. team. All rights reserved.
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
""" VMamba model configuration """


from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

VMAMBA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "vmamba": "https://huggingface.co/vmamba/resolve/main/config.json",
    # See all VMamba models at https://huggingface.co/models?filter=vmamba
}


class VMambaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~VMambaModel`].
    It is used to instantiate an VMamba model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the VMamba [vmamba](https://huggingface.co/vmamba) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        depths (`list`, *optional*, defaults to [2, 2, 9, 2]):
            The number of VSS blocks per state. Default is the configuration for VMamba-Tiny.
        dims (`list`, *optional*, defaults to [96, 192, 384, 768]):
            Dimensionality for each layer. Default is the configuration for VMamba-Tiny.
        d_state (`int`, *optional*, defaults to 16):
            Dimensionality of the patch embeddings.
        drop_rate (`int`, *optional*, defaults to 0):
            Dropout rate.
        drop_path_rate (`int`, *optional*, defaults to 0.1):
            Stochastic depth decay
        use_checkpoint (`bool`, *optional*, defaults to `False`):
            Whether to use gradient checkpointing.
        Example:

    ```python
    >>> from transformers import VMambaModel, VMambaConfig

    >>> # Initializing a VMamba vmamba style configuration
    >>> configuration = VMambaConfig()

    >>> # Initializing a model from the vmamba style configuration
    >>> model = VMambaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vmamba"

    def __init__(
        self,
        patch_size=4,
        in_channels=3,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        d_state=16,
        drop_rate=0,
        drop_path_rate=0.1,
        use_checkpoint=False,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.depths = depths
        self.dims = dims
        self.d_state = d_state
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.use_checkpoint = use_checkpoint
        super().__init__(**kwargs)
