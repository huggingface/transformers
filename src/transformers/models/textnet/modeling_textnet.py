# coding=utf-8
# Copyright 2021 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch TextNet model."""
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import PreTrainedModel, add_start_docstrings
from transformers.modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from transformers.models.textnet.configuration_textnet import TextNetConfig
from transformers.utils import add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers.utils.backbone_utils import BackboneMixin


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "BitConfig"

TEXTNET_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`BitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`BitImageProcessor.__call__`]
            for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
"""

BIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # "google/bit-50",
    # See all BiT models at https://huggingface.co/models?filter=bit
]


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    return kernel_size // 2


def build_activation(act_func, inplace=True):
    if act_func == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_func == "relu6":
        return nn.ReLU6(inplace=inplace)
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    elif act_func is None:
        return None
    else:
        raise ValueError("do not support: %s" % act_func)


class TextNetConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        has_shuffle=False,
        use_batch_norm=True,
        act_func="relu",
        dropout_rate=0,
        use_act=True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
        self.activation_function = act_func

        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.batch_norm = nn.Identity()
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)

        self.activation = nn.Identity()
        if use_act:
            act = build_activation(self.activation_function, True)
            if act is not None:
                self.activation = act

    def forward(self, hidden_states):
        if self.training:
            if hasattr(self, "fused_conv"):
                delattr(self, "fused_conv")
            hidden_states = self.conv(hidden_states)
            hidden_states = self.batch_norm(hidden_states)
            return self.activation(hidden_states)
        else:
            if not hasattr(self, "fused_conv"):
                setattr(self, "fused_conv", self.fuse_conv_batch_norm(self.conv, self.batch_norm))
            hidden_states = self.fused_conv(hidden_states)
            if self.activation is not None:
                hidden_states = self.activation(hidden_states)
            return hidden_states

    def fuse_conv_batch_norm(self, conv, batch_norm):
        """During inference, the functionary of batch norm layers is turned off but
        only the mean and var alone channels are used, which exposes the chance to fuse it with the preceding conv
        layers to save computations and simplify network structures."""
        if isinstance(batch_norm, nn.Identity):
            return conv
        conv_w = conv.weight
        conv_b = conv.bias if conv.bias is not None else torch.zeros_like(batch_norm.running_mean)

        factor = batch_norm.weight / torch.sqrt(batch_norm.running_var + batch_norm.eps)
        conv.weight = nn.Parameter(conv_w * factor.reshape([conv.out_channels, 1, 1, 1]))
        conv.bias = nn.Parameter((conv_b - batch_norm.running_mean) * factor + batch_norm.bias)
        return conv


class TestNetRepConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        padding = (int(((kernel_size[0] - 1) * dilation) / 2), int(((kernel_size[1] - 1) * dilation) / 2))

        self.nonlinearity = nn.ReLU(inplace=True)

        self.main_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.main_batch_norm = nn.BatchNorm2d(num_features=out_channels)

        ver_pad = (int(((kernel_size[0] - 1) * dilation) / 2), 0)
        hor_pad = (0, int(((kernel_size[1] - 1) * dilation) / 2))

        if kernel_size[1] != 1:
            self.vertical_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size[0], 1),
                stride=stride,
                padding=ver_pad,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.vertical_batch_norm = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.vertical_conv, self.vertical_batch_norm = None, None

        if kernel_size[0] != 1:  # 卷积核的高大于1 -> 有水平卷积
            self.horizontal_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, kernel_size[1]),
                stride=stride,
                padding=hor_pad,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.horizontal_batch_norm = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.horizontal_conv, self.horizontal_batch_norm = None, None

        self.rbr_identity = (
            nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        )

    def forward(self, hidden_states):
        if self.training:
            if hasattr(self, "fused_conv"):
                self.__delattr__("fused_conv")

            main_outputs = self.main_conv(hidden_states)
            main_outputs = self.main_batch_norm(main_outputs)
            if self.vertical_conv is not None:
                vertical_outputs = self.vertical_conv(hidden_states)
                vertical_outputs = self.vertical_batch_norm(vertical_outputs)
            else:
                vertical_outputs = 0

            if self.horizontal_conv is not None:
                horizontal_outputs = self.horizontal_conv(hidden_states)
                horizontal_outputs = self.horizontal_batch_norm(horizontal_outputs)
            else:
                horizontal_outputs = 0

            if self.rbr_identity is None:
                id_out = 0
            else:
                id_out = self.rbr_identity(hidden_states)

            return self.nonlinearity(main_outputs + vertical_outputs + horizontal_outputs + id_out)
        else:
            if not hasattr(self, "fused_conv"):
                self.prepare_for_eval()
            return self.nonlinearity(self.fused_conv(hidden_states))

    def _identity_to_conv(self, identity):
        if identity is None:
            return 0, 0
        if not hasattr(self, "id_tensor"):
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 1, 1), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 0, 0] = 1
            id_tensor = torch.from_numpy(kernel_value).to(identity.weight.device)
            self.id_tensor = self._pad_to_mxn_tensor(id_tensor)
        kernel = self.id_tensor
        running_mean = identity.running_mean
        running_var = identity.running_var
        gamma = identity.weight
        beta = identity.bias
        eps = identity.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _fuse_batch_norm_tensor(self, conv, batch_norm):
        kernel = conv.weight
        kernel = self._pad_to_mxn_tensor(kernel)
        running_mean = batch_norm.running_mean
        running_var = batch_norm.running_var
        gamma = batch_norm.weight
        beta = batch_norm.bias
        eps = batch_norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        kernel_mxn, bias_mxn = self._fuse_batch_norm_tensor(self.main_conv, self.main_batch_norm)
        if self.vertical_conv is not None:
            kernel_mx1, bias_mx1 = self._fuse_batch_norm_tensor(self.vertical_conv, self.vertical_batch_norm)
        else:
            kernel_mx1, bias_mx1 = 0, 0
        if self.horizontal_conv is not None:
            kernel_1xn, bias_1xn = self._fuse_batch_norm_tensor(self.horizontal_conv, self.horizontal_batch_norm)
        else:
            kernel_1xn, bias_1xn = 0, 0
        kernel_id, bias_id = self._identity_to_conv(self.rbr_identity)
        kernel_mxn = kernel_mxn + kernel_mx1 + kernel_1xn + kernel_id
        bias_mxn = bias_mxn + bias_mx1 + bias_1xn + bias_id
        return kernel_mxn, bias_mxn

    def _pad_to_mxn_tensor(self, kernel):
        kernel_height, kernel_width = self.kernel_size
        height, width = kernel.shape[2:]
        pad_left_right = (kernel_width - width) // 2
        pad_top_down = (kernel_height - height) // 2
        return torch.nn.functional.pad(kernel, [pad_left_right, pad_left_right, pad_top_down, pad_top_down])

    def prepare_for_eval(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(
            in_channels=self.main_conv.in_channels,
            out_channels=self.main_conv.out_channels,
            kernel_size=self.main_conv.kernel_size,
            stride=self.main_conv.stride,
            padding=self.main_conv.padding,
            dilation=self.main_conv.dilation,
            groups=self.main_conv.groups,
            bias=True,
        )
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias
        for para in self.fused_conv.parameters():
            para.detach_()


class TextNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TextNetConfig
    base_model_prefix = "textnet"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()


@add_start_docstrings(
    "The bare Textnet model outputting raw features without any specific head on top.",
    TEXTNET_START_DOCSTRING,
)
class TextNetModel(TextNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.first_conv = TextNetConvLayer(
            config.in_channels,
            config.out_channels,
            config.kernel_size,
            config.stride,
            config.dilation,
            config.groups,
            config.bias,
            config.has_shuffle,
            config.use_bn,
            config.act_func,
            config.dropout_rate,
            config.ops_order,
        )
        stage1 = []
        for stage_config in zip(
            config.stage1_in_channels,
            config.stage1_out_channels,
            config.stage1_kernel_size,
            config.stage1_stride,
            config.stage1_dilation,
            config.stage1_groups,
        ):
            stage1.append(TestNetRepConvLayer(*stage_config))
        self.stage1 = nn.ModuleList(stage1)

        stage2 = []
        for stage_config in zip(
            config.stage2_in_channels,
            config.stage2_out_channels,
            config.stage2_kernel_size,
            config.stage2_stride,
            config.stage2_dilation,
            config.stage2_groups,
        ):
            stage2.append(TestNetRepConvLayer(*stage_config))
        self.stage2 = nn.ModuleList(stage2)

        stage3 = []
        for stage_config in zip(
            config.stage3_in_channels,
            config.stage3_out_channels,
            config.stage3_kernel_size,
            config.stage3_stride,
            config.stage3_dilation,
            config.stage3_groups,
        ):
            stage3.append(TestNetRepConvLayer(*stage_config))
        self.stage3 = nn.ModuleList(stage3)

        stage4 = []
        for stage_config in zip(
            config.stage4_in_channels,
            config.stage4_out_channels,
            config.stage4_kernel_size,
            config.stage4_stride,
            config.stage4_dilation,
            config.stage4_groups,
        ):
            stage4.append(TestNetRepConvLayer(*stage_config))
        self.stage4 = nn.ModuleList(stage4)

        self.pooler = nn.AdaptiveAvgPool2d((2, 2))

        self.init_weights()

    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> Union[Tuple[Any, List[Any]], Tuple[Any], BaseModelOutputWithPoolingAndNoAttention]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_state = self.first_conv(pixel_values)
        hidden_states = [hidden_state]

        for block in self.stage1:
            hidden_state = block(hidden_state)
        hidden_states.append(hidden_state)

        for block in self.stage2:
            hidden_state = block(hidden_state)
        hidden_states.append(hidden_state)

        for block in self.stage3:
            hidden_state = block(hidden_state)
        hidden_states.append(hidden_state)

        for block in self.stage4:
            hidden_state = block(hidden_state)
        hidden_states.append(hidden_state)

        pooled_output = self.pooler(hidden_state)

        if not return_dict:
            output = (pooled_output, hidden_state)
            return output + (hidden_states,) if output_hidden_states else output

        return BaseModelOutputWithPoolingAndNoAttention(
            pooler_output=pooled_output,
            last_hidden_state=hidden_state,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
        )


@add_start_docstrings(
    """
    TextNet backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    TEXTNET_START_DOCSTRING,
)
class TextNetBackbone(TextNetPreTrainedModel, BackboneMixin):
    def __init__(self, config):
        super().__init__(config)
        super()._init_backbone(config)

        self.textnet = TextNetModel(config)
        self.num_features = [
            config.out_channels,
            config.stage1_out_channels[-1],
            config.stage2_out_channels[-1],
            config.stage3_out_channels[-1],
            config.stage4_out_channels[-1],
        ]

        # initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward("BIT_INPUTS_DOCSTRING")
    @replace_return_docstrings(output_type=BackboneOutput, config_class="")
    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        # >>> from transformers import AutoImageProcessor, AutoBackbone
        # >>> import torch
        # >>> from PIL import Image
        # >>> import requests
        #
        # >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        # >>> image = Image.open(requests.get(url, stream=True).raw)
        #
        # >>> processor = AutoImageProcessor.from_pretrained("google/resnetnv2-50")
        # >>> model = AutoBackbone.from_pretrained("google/resnetnv2-50")
        #
        # >>> inputs = processor(image, return_tensors="pt")
        # >>> outputs = model(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.textnet(pixel_values, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states

        feature_maps = ()
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )


@add_start_docstrings(
    """
    TextNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    TEXTNET_START_DOCSTRING,
)
class TextNetForImageClassification(TextNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.textnet = TextNetModel(config)
        # classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.hidden_sizes[-1] * 2 * 2, config.num_labels) if config.num_labels > 0 else nn.Identity(),
        )
        # initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward("BIT_INPUTS_DOCSTRING")
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ImageClassifierOutputWithNoAttention:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.textnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        last_hidden_state = outputs.last_hidden_state if return_dict else outputs[0]

        logits = self.classifier(last_hidden_state)

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
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
