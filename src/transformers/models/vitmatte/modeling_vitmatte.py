# coding=utf-8
# Copyright 2023 HUST-VL and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ViTMatte backbone."""

from typing import Optional

import torch
from torch import nn

from ... import AutoBackbone
from ...modeling_utils import PreTrainedModel
from ...utils.backbone_utils import BackboneMixin
from .configuration_vitmatte import VitMatteConfig


VITMATTE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "hustvl/vitmatte-small-composition-1k",
    # See all VitMatte models at https://huggingface.co/models?filter=vitmatte
]


class VitMattePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VitMatteConfig
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, VitMattePreTrainedModel):
            module.backbone.init_weights()

    def init_weights(self):
        """Initialize the weights"""
        self.backbone.init_weights()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BackboneMixin):
            module.gradient_checkpointing = value


class VitMatteForImageMatting(VitMattePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.backbone = AutoBackbone.from_config(config.backbone_config)
        self.decoder = nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, VitMatteForImageMatting
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
        >>> model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )

        features = outputs.feature_maps

        # TODO: permute + apply decoder
        # features = features.permute(0, 3, 1, 2)

        if labels is not None:
            raise NotImplementedError("Training is not yet supported")

        return features
