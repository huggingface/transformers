# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import os
from typing import Optional, Tuple, Union

import torch

from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel, load_state_dict, SAFE_WEIGHTS_NAME
from ...utils import (
    add_start_docstrings_to_model_forward,
    is_timm_available,
    is_torch_available,
    requires_backends,
    is_timm_checkpoint,
    is_timm_hub_checkpoint,
)
from .configuration_timm_wrapper import TimmWrapperConfig


if is_timm_available():
    import timm


if is_torch_available():
    from torch import Tensor


TIMM_WRAPPER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. Not compatible with timm wrapped models.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. Not compatible with timm wrapped models.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class TimmWrapperModel(PreTrainedModel):
    """
    Wrapper class for timm models to be used in transformers.
    """

    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False
    config_class = TimmWrapperConfig

    def __init__(self, config, **kwargs):
        requires_backends(self, "timm")
        super().__init__(config)
        self.config = config

        pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path", None)

        # model_name passed into kwargs takes precedence
        model_name = kwargs.pop("model_name", None)
        if model_name is None and hasattr(config, "model_name"):
            model_name = config.model_name
        elif model_name is None and pretrained_model_name_or_path is not None:
            model_name = pretrained_model_name_or_path
        elif model_name is None:
            raise ValueError("model_name must be specified in either the config or kwargs")

        pretrained = kwargs.pop("pretrained", False)

        # If the pretrained_model_name_or_path is a timm checkpoint, and a local file, we load the checkpoint safetensors file as the model
        if is_timm_checkpoint(pretrained_model_name_or_path) and not is_timm_hub_checkpoint(pretrained_model_name_or_path):
            model = timm.create_model(model_name=model_name, pretrained=False)
            if pretrained:
                weights_path = os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME)
                state_dict = load_state_dict(weights_path)
                # Remove the prefix "model." from the keys
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
        # If the pretrained_model_name_or_path is a timm checkpoint and matches a checkpoint on the hub, we use timm.create_model directly
        else:
            model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
            )
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        requires_backends(cls, ["vision", "timm"])
        from ...models.timm_wrapper import TimmWrapperConfig

        config = kwargs.pop("config", TimmWrapperConfig.from_pretrained(pretrained_model_name_or_path))
        pretrained = kwargs.pop("pretrained", True)
        model_name = kwargs.pop("model_name", None)

        if kwargs:
            raise ValueError(f"Unknown arguments: {', '.join(kwargs.keys())}")

        kwargs["pretrained_model_name_or_path"] = pretrained_model_name_or_path
        kwargs["pretrained"] = kwargs.pop("pretrained", True)

        # FIXME - use super from_pretrained, or at least _from_pretrained if possible

        return super()._from_config(config, **kwargs)

    def _init_weights(self, module):
        """
        Empty init weights function to ensure compatibility of the class in the library.
        """
        pass

    @add_start_docstrings_to_model_forward(TIMM_WRAPPER_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[BaseModelOutput, Tuple[Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if output_hidden_states is not None or output_attentions is not None:
            raise ValueError("Cannot set output_attentions or output_hidden_states for timm models")

        prediction = self.model(pixel_values, **kwargs)

        if not return_dict:
            return (prediction,)

        return BaseModelOutput(last_hidden_state=prediction)


class TimmWrapperForImageClassification(TimmWrapperModel):
    """
    Wrapper class for timm models to be used in transformers for image classification.
    """

    @add_start_docstrings_to_model_forward(TIMM_WRAPPER_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[BaseModelOutput, Tuple[Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if output_hidden_states is not None or output_attentions is not None:
            raise ValueError("Cannot set return_dict, output_attentions or output_hidden_states for timm models")

        logits = self.model(pixel_values, **kwargs)

        loss = None
        if labels is not None:
            raise ValueError("It is not possible to train timm models for image classification yet.")

        if not return_dict:
            return (loss, logits)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
