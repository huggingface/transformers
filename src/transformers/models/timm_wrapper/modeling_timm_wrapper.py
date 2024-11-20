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

from typing import List, Optional, Tuple, Union

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


def _load_timm_model(config: TimmWrapperConfig, add_classification_head: bool = False):
    # timm model will not add classification head if num_classes = 0
    num_classes = config.num_labels if add_classification_head else 0
    model = timm.create_model(model_name=config.architecture, pretrained=False, num_classes=num_classes)
    return model


class TimmWrapperPreTrainedModel(PreTrainedModel):
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False
    config_class = TimmWrapperConfig
    _no_split_modules = []

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision", "timm"])
        super().__init__(*args, **kwargs)

    @staticmethod
    def _fix_state_dict_key_on_load(key):
        """
        Override original method which renames `gamma` and `beta` to `weight` and `bias`.
        We don't want this behavior for timm wrapped models. Instead, this method adds
        "timm_model." prefix to be able to load official timm Hub checkpoints.
        """
        if "timm_model." not in key:
            return f"timm_model.{key}"
        return key

    def _fix_state_dict_key_on_save(self, key):
        """
        Override original method to remove "timm_model." prefix from state_dict keys.
        Makes saved checkpoint compatible with `timm` library.
        """
        return key.replace("timm_model.", "")

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Override original method to fix state_dict keys on load for cases when weight are loaded
        without using `from_pretrained` method (e.g. in Trainer to resume from checkpoint).
        """
        state_dict = self._fix_state_dict_keys_on_load(state_dict)
        return super().load_state_dict(state_dict, *args, **kwargs)

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

    def __init__(self, config: TimmWrapperConfig, add_classification_head: bool = False):
        super().__init__(config)
        self.timm_model = _load_timm_model(config, add_classification_head=add_classification_head)
        self.post_init()

    @add_start_docstrings_to_model_forward(TIMM_WRAPPER_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[Union[bool, List[int]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[BaseModelOutput, Tuple[Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        if output_attentions:
            raise ValueError("Cannot set `output_attentions` for timm models")

        if output_hidden_states and not hasattr(self.timm_model, "forward_intermediates"):
            raise ValueError(
                "The 'output_hidden_states' option cannot be set for this timm model. "
                "To enable this feature, the 'forward_intermediates' method must be implemented "
                "in the timm model (available in timm versions > 1.*). Please consider using a "
                "different architecture or updating the timm package to a compatible version."
            )

        pixel_values = pixel_values.to(self.device, self.dtype)

        if output_hidden_states:
            # to enable hidden states selection
            if isinstance(output_hidden_states, (list, tuple)):
                kwargs["indices"] = output_hidden_states
            last_hidden_state, hidden_states = self.timm_model.forward_intermediates(pixel_values, **kwargs)
        else:
            last_hidden_state = self.timm_model.forward_features(pixel_values, **kwargs)
            hidden_states = None

        # logits in case add_classification_head=True, otherwise it would be pooled output
        head_output = self.timm_model.forward_head(last_hidden_state)

        if not return_dict:
            output = (head_output, hidden_states) if hidden_states is not None else (head_output,)
            return output

        return BaseModelOutput(last_hidden_state=head_output, hidden_states=hidden_states)


class TimmWrapperForImageClassification(TimmWrapperModel):
    """
    Wrapper class for timm models to be used in transformers for image classification.
    """

    def __init__(self, config: TimmWrapperConfig):
        super().__init__(config, add_classification_head=True)
        self.num_labels = config.num_labels
        self.post_init()

    @add_start_docstrings_to_model_forward(TIMM_WRAPPER_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[Union[bool, List[int]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[ImageClassifierOutput, Tuple[Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = super().forward(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        loss = None
        if labels is not None:
            logits = outputs[0]
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
            outputs = (loss,) + outputs
            outputs = tuple(out for out in outputs if out is not None)
            return outputs

        return ImageClassifierOutput(
            loss=loss,
            logits=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )
