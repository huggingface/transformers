# coding=utf-8
# Copyright 2022 Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, Abhinav Gupta The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch R3M model. """




import math
import os

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
import torchvision
from torch.nn.modules.linear import Identity
from torchvision import transforms
import torchvision.transforms as T
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Optional, Tuple, Union

from ...activations import ACT2FN
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_utils import (
    PreTrainedModel,
)
from ...utils import logging
from .configuration_r3m import R3MConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "surajnair/r3m-50"
_CONFIG_FOR_DOC = "R3MConfig"

R3M_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "surajnair/r3m-50",
    # See all R3M models at https://huggingface.co/models?filter=r3m
]


class R3MPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = R3MConfig
    base_model_prefix = "r3m"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, R3MModel):
            module.gradient_checkpointing = value


R3M_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config ([`~R3MConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

R3M_INPUTS_DOCSTRING = r"""
    Args:
        image  (`torch.FloatTensor` of shape `(batch_size, 3, height, width)`, *required*):
            Input batch of images
        image_shape (list of image [channels, height, width], *optional*):
            Input image shape.
"""


@add_start_docstrings(
    "The bare R3M Model outputting raw hidden-states without any specific head on top.",
    R3M_START_DOCSTRING,
)
class R3MModel(R3MPreTrainedModel):
    """

    The model can behaves as an image encoder.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        size = self.config.resnet_size
        
        ## Visual Encoder
        if size == 18:
            self.outdim = 512
            self.convnet = torchvision.models.resnet18(pretrained=False)
        elif size == 34:
            self.outdim = 512
            self.convnet = torchvision.models.resnet34(pretrained=False)
        elif size == 50:
            self.outdim = 2048
            self.convnet = torchvision.models.resnet50(pretrained=False)
            
        self.convnet.fc = Identity()
        self.normlayer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Initialize weights and apply final processing
        self.post_init()

    @add_code_sample_docstrings(
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        image, 
        image_shape = [3, 224, 224]):
        r"""
        image  (`torch.FloatTensor` of shape `(batch_size, 3, height, width)`, *required*):
            Input batch of images
        image_shape (list of image [channels, height, width], *optional*):
            Input image shape.
        """
        if image_shape != [3, 224, 224]:
            preprocess = nn.Sequential(
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        self.normlayer,
                )
        else:
            preprocess = nn.Sequential(
                        self.normlayer,
                )

        ## Input must be [0, 255], [3,244,244]
        obs = image.float() /  255.0
        obs_p = preprocess(obs)
        h = self.convnet(obs_p)
        return h