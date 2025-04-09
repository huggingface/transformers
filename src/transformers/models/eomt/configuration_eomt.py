# coding=utf-8
# Copyright 2025 Mobile Perception Systems Lab at TU/e and The HuggingFace Inc. team. All rights reserved.
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
"""EoMT model configuration"""

from typing import Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class EoMTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`EoMTModel`]. It is used to instantiate a
    EoMT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the EoMT
    [tue-mps/coco_panoptic_eomt_large_640](https://huggingface.co/tue-mps/coco_panoptic_eomt_large_640)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone (`str`, *optional*, defaults to `"vit_large_patch14_reg4_dinov2"`):
            Name of backbone to use, this will load the corresponding pretrained weights from the timm library.
        image_size (`int`, *optional*, defaults to 640):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_blocks (`int`, *optional*, defaults to 4):
            Number of blocks that process queries.
        num_queries (`int`, *optional*, defaults to 200):
            Number of queries.

    Examples:

    ```python
    >>> from transformers import EoMTConfig, EoMTModel

    >>> # Initializing a EoMT tue-mps/coco_panoptic_eomt_large_640 configuration
    >>> configuration = EoMTConfig()

    >>> # Initializing a model (with random weights) from the tue-mps/coco_panoptic_eomt_large_640 style configuration
    >>> model = EoMTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    """

    model_type = "eomt"
    backbones_supported = ["vit"]

    def __init__(
        self,
        backbone: Optional[str] = "vit_large_patch14_reg4_dinov2",
        image_size: int = 640,
        patch_size: int = 16,
        num_blocks: int = 4,
        num_queries: int = 200,
        **kwargs,
    ):
        self.num_hidden_layers = num_blocks + 1  # + 1 for the initial query tokens
        self.num_blocks = num_blocks
        self.num_queries = num_queries
        self.backbone = backbone
        self.patch_size = patch_size
        self.use_timm_backbone = True
        self.use_pretrained_backbone = True
        self.backbone_kwargs = {
            "img_size": image_size,
            "patch_size": patch_size,
            "features_only": False,
            "num_classes": 0,
        }

        super().__init__(**kwargs)


__all__ = ["EoMTConfig"]
