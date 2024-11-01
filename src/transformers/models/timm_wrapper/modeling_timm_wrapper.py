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

from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings_to_model_forward,
    is_timm_available,
    requires_backends,
)
from .configuration_timm_wrapper import TimmWrapperConfig


if is_timm_available():
    import timm


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
        **kwargs:
            Additional keyword arguments passed along to the model forward.
"""


def _load_timm_model(
    config: TimmWrapperConfig,
    model_name: Optional[str] = None,
    pretrained_model_name_or_path: Optional[str] = None,
    add_classification_head: bool = False,
):
    # model_name passed into kwargs takes precedence
    if model_name is None and hasattr(config, "model_name"):
        model_name = config.model_name
    elif model_name is None and pretrained_model_name_or_path is not None:
        model_name = pretrained_model_name_or_path
    elif model_name is None:
        raise ValueError("model_name must be specified in either the config or kwargs")

    # timm model will not add classification head if num_classes = 0
    num_classes = config.num_labels if add_classification_head else 0

    model = timm.create_model(model_name=model_name, pretrained=False, num_classes=num_classes)

    return model


class TimmWrapperPreTrainedModel(PreTrainedModel):
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False
    config_class = TimmWrapperConfig
    base_model_prefix = "timm_model"
    _no_split_modules = []

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision", "timm"])
        super().__init__(*args, **kwargs)

    @staticmethod
    def _fix_state_dict_key(key):
        """
        Override original method which renames `gamma` and `beta` to `weight` and `bias`.
        We don't want this behavior for timm wrapped models.
        """
        return key

    def _init_weights(self, module):
        """
        Empty init weights function to ensure compatibility of the class in the library.
        """
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()


class TimmWrapperModel(TimmWrapperPreTrainedModel):
    """
    Wrapper class for timm models to be used in transformers.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.timm_model = _load_timm_model(config, add_classification_head=False, **kwargs)
        self.post_init()

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

        features = self.timm_model(pixel_values, **kwargs)

        if not return_dict:
            return (features,)

        return BaseModelOutput(last_hidden_state=features)


class TimmWrapperForImageClassification(TimmWrapperPreTrainedModel):
    """
    Wrapper class for timm models to be used in transformers for image classification.
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.timm_model = _load_timm_model(config, add_classification_head=True, **kwargs)
        self.num_labels = config.num_labels
        self.post_init()

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
            raise ValueError("Cannot set `output_attentions` or `output_hidden_states` for timm models")

        logits = self.timm_model(pixel_values, **kwargs)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
