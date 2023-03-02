# coding=utf-8
# Copyright 2022 The Google AI team and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch TrillssonEfficientNet model."""

from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPoolingAndNoAttention, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_trillsson_efficientnet import TrillssonEfficientNetConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "TrillssonEfficientNetConfig"
_FEAT_EXTRACTOR_FOR_DOC = "TrillssonEfficientNetFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "vumichien/nonsemantic-speech-trillsson3"
_EXPECTED_OUTPUT_SHAPE = [1, 1024]

# Audio class docstring
_SEQ_CLASS_CHECKPOINT = "vumichien/trillsson3-ft-keyword-spotting"
_SEQ_CLASS_EXPECTED_OUTPUT = "'_unknown_'"
_SEQ_CLASS_EXPECTED_LOSS = 9.88

TRILLSSON_EFFICIENTNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "vumichien/nonsemantic-speech-trillsson3",
]


def make_divisible(filters, divisor=8, min_value=None) -> int:
    """
    Ensure that all layers have a channel count that is divisible by `divisor`. This function is taken from the
    original TensorFlow repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_filters = max(min_value, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return new_filters


class Conv2DSamePadding(nn.Conv2d):
    """2D Convolutions same padding like TensorFlow,"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_features):
        in_height = int(input_features.shape[-2])
        in_width = int(input_features.shape[-1])
        stride_height, stride_width = self.stride
        kernel_height, kernel_width = self.kernel_size
        dilation_height, dilation_width = self.dilation

        if in_height % stride_height == 0:
            pad_along_height = max(kernel_height - stride_height, 0)
        else:
            pad_along_height = max(kernel_height - (in_height % stride_height), 0)

        if in_width % stride_width == 0:
            pad_along_width = max(kernel_width - stride_width, 0)
        else:
            pad_along_width = max(kernel_width - (in_width % stride_width), 0)

        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top

        padding = (
            pad_left * dilation_width,
            pad_right * dilation_width,
            pad_top * dilation_height,
            pad_bottom * dilation_height,
        )
        output_features = nn.functional.pad(input_features, padding, "constant", 0.0)
        return nn.functional.conv2d(
            output_features, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class TrillssonSqueezeExcitation(nn.Module):
    def __init__(
        self, config: TrillssonEfficientNetConfig, in_channels: int, out_channels: int, reduction: int = 4
    ) -> None:
        super().__init__()
        self.config = config
        self.se_squeeze = nn.AdaptiveAvgPool2d(1)
        self.se_reduce = Conv2DSamePadding(
            in_channels=out_channels,
            out_channels=make_divisible(in_channels // reduction, config.depth_divisible_by, config.min_depth),
            kernel_size=1,
        )
        self.se_expand = Conv2DSamePadding(
            in_channels=make_divisible(in_channels // reduction, config.depth_divisible_by, config.min_depth),
            out_channels=out_channels,
            kernel_size=1,
        )
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, input_features):
        hidden_states = self.se_squeeze(input_features)
        hidden_states = self.se_reduce(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.se_expand(hidden_states)
        hidden_states = torch.sigmoid(hidden_states) * input_features
        return hidden_states


class TrillssonStem(nn.Module):
    def __init__(
        self,
        config: TrillssonEfficientNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.stem_conv = Conv2DSamePadding(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=0, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(
            num_features=out_channels, eps=self.config.norm_eps, momentum=self.config.norm_momentum
        )
        self.activation = ACT2FN[config.hidden_act]
        self.gradient_checkpointing = False

    def forward(self, input_features):
        hidden_states = self.stem_conv(input_features)
        hidden_states = self.stem_bn(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TrillssonTop(nn.Module):
    def __init__(
        self,
        config: TrillssonEfficientNetConfig,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.top_conv = Conv2DSamePadding(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.top_bn = nn.BatchNorm2d(
            num_features=out_channels, eps=self.config.norm_eps, momentum=self.config.norm_momentum
        )
        self.activation = ACT2FN[config.hidden_act]
        self.gradient_checkpointing = False

    def forward(self, input_features):
        hidden_states = self.top_conv(input_features)
        hidden_states = self.top_bn(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TrillssonInvertedResidual(nn.Module):
    def __init__(
        self,
        config: TrillssonEfficientNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
        use_se: int,
        survival_probability: int,
    ) -> None:
        """TrillssonInvertedResidual block: Inverted Residual Bottleneck."""
        super().__init__()
        hidden_dim = round(in_channels * expand_ratio)
        self.config = config
        self.expand_ratio = expand_ratio
        self.use_se = use_se
        self.survival_probability = survival_probability
        self.identity = stride == 1 and in_channels == out_channels
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(self.survival_probability)
        if self.expand_ratio != 1:
            self.expand_conv = Conv2DSamePadding(
                in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.expand_bn = nn.BatchNorm2d(
                num_features=hidden_dim, eps=self.config.norm_eps, momentum=self.config.norm_momentum
            )
        self.dwconv2 = Conv2DSamePadding(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=stride,
            padding=0,
            groups=hidden_dim,
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(
            num_features=hidden_dim, eps=self.config.norm_eps, momentum=self.config.norm_momentum
        )
        if self.use_se:
            self.squeeze_excitation = TrillssonSqueezeExcitation(
                config=self.config, in_channels=in_channels, out_channels=hidden_dim
            )
            self.project_conv = Conv2DSamePadding(
                in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.project_bn = nn.BatchNorm2d(
                num_features=out_channels, eps=self.config.norm_eps, momentum=self.config.norm_momentum
            )
        self.gradient_checkpointing = False

    def forward(self, input_features):
        if self.expand_ratio != 1:
            hidden_states = self.expand_conv(input_features)
            hidden_states = self.expand_bn(hidden_states)
            hidden_states = self.activation(hidden_states)
        else:
            hidden_states = input_features
        hidden_states = self.dwconv2(hidden_states)
        hidden_states = self.normalization(hidden_states)
        hidden_states = self.activation(hidden_states)
        if self.use_se:
            hidden_states = self.squeeze_excitation(hidden_states)
            hidden_states = self.project_conv(hidden_states)
            hidden_states = self.project_bn(hidden_states)
            if self.identity:
                if self.survival_probability:
                    hidden_states = self.dropout(hidden_states)
                hidden_states = input_features + hidden_states
        return hidden_states


class TrillssonFusedInvertedResiduals(nn.Module):
    def __init__(
        self,
        config: TrillssonEfficientNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
        use_se: int,
        survival_probability: float,
    ) -> None:
        """TrillssonFusedInvertedResiduals: Fusing the proj conv1x1 and depthwise_conv into a conv2d."""
        super().__init__()
        hidden_dim = round(in_channels * expand_ratio)
        self.config = config
        self.expand_ratio = expand_ratio
        self.use_se = use_se
        self.survival_probability = survival_probability
        self.identity = stride == 1 and in_channels == out_channels
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(self.survival_probability)

        if self.expand_ratio != 1:
            self.expand_conv = Conv2DSamePadding(
                in_channels=in_channels, out_channels=hidden_dim, kernel_size=3, stride=stride, padding=0, bias=False
            )
            self.expand_bn = nn.BatchNorm2d(
                num_features=hidden_dim, eps=self.config.norm_eps, momentum=self.config.norm_momentum
            )
            self.project_conv = Conv2DSamePadding(
                in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.project_bn = nn.BatchNorm2d(
                num_features=out_channels, eps=self.config.norm_eps, momentum=self.config.norm_momentum
            )
        else:
            self.project_conv = Conv2DSamePadding(
                in_channels=in_channels, out_channels=hidden_dim, kernel_size=3, stride=stride, padding=0, bias=False
            )
            self.project_bn = nn.BatchNorm2d(
                num_features=hidden_dim, eps=self.config.norm_eps, momentum=self.config.norm_momentum
            )

        if self.use_se:
            self.squeeze_excitation = TrillssonSqueezeExcitation(
                config=self.config, in_channels=in_channels, out_channels=hidden_dim
            )
        self.gradient_checkpointing = False

    def forward(self, input_features):
        if self.expand_ratio != 1:
            hidden_states = self.expand_conv(input_features)
            hidden_states = self.expand_bn(hidden_states)
            hidden_states = self.activation(hidden_states)
        else:
            hidden_states = input_features

        if self.use_se:
            hidden_states = self.squeeze_excitation(hidden_states)
        hidden_states = self.project_conv(hidden_states)
        hidden_states = self.project_bn(hidden_states)
        if self.expand_ratio == 1:
            hidden_states = self.activation(hidden_states)
        if self.identity:
            if self.survival_probability:
                hidden_states = self.dropout(hidden_states)
            return input_features + hidden_states
        else:
            return hidden_states


class TrillssonEfficientNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TrillssonEfficientNetConfig
    base_model_prefix = "trillsson_efficientnet"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(
            module, (TrillssonStem, TrillssonInvertedResidual, TrillssonFusedInvertedResiduals, TrillssonTop)
        ):
            module.gradient_checkpointing = value


TRILLSSON_EFFICIENTNET_START_DOCSTRING = r"""

    TrillssonEfficientNet was proposed in [TRILLsson: Small, Universal Speech Representations for Paralinguistic Tasks]
    (https://arxiv.org/pdf/2203.00236) by Joel Shor, Subhashini Venugopalan. This model inherits from
    [`PreTrainedModel`]. Check the superclass documentation for the generic methods the library implements for all its
    model (such as downloading or saving etc.). This model is a PyTorch [torch.nn.Module](https:
        //pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to
        the PyTorch documentation for all matter related to general usage and behavior.

    Parameters:
        config ([`TrillssonEfficientNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TRILLSSON_EFFICIENTNET_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type *List[float]* or a *numpy.ndarray*, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into *input_values*, the [`TrillssonEfficientNetFeatureExtractor`] should
            be used for padding and conversion into a tensor of type *torch.FloatTensor*. See
            [`TrillssonEfficientNetFeatureExtractor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare TrillssonEfficientNet model outputting raw hidden-states without any specific head on top.",
    TRILLSSON_EFFICIENTNET_START_DOCSTRING,
)
class TrillssonEfficientNetModel(TrillssonEfficientNetPreTrainedModel):
    def __init__(self, config: TrillssonEfficientNetConfig, add_pooling_layer: bool = True):
        super().__init__(config)

        self.config = config
        # building first layer
        input_channel = make_divisible(
            24 * self.config.depth_multiplier, self.config.depth_divisible_by, self.config.min_depth
        )

        self.stem = TrillssonStem(
            config=self.config,
            in_channels=1,
            out_channels=input_channel,
            stride=2,
        )

        # building inverted residual blocks
        self.block = nn.ModuleList()
        current_num_blocks = 0
        total_blocks = float(sum(blocks_args[0] for blocks_args in self.config.block_configs))
        for expand_ratio, output_filters, num_repeat, stride, use_se, conv_type in self.config.block_configs:
            output_channel = make_divisible(
                output_filters * config.depth_multiplier, self.config.depth_divisible_by, self.config.min_depth
            )
            conv_block = TrillssonInvertedResidual if conv_type == 0 else TrillssonFusedInvertedResiduals
            for i in range(num_repeat):
                if i > 0:
                    stride = 1
                survival_probability = self.config.drop_connect_rate * current_num_blocks / total_blocks
                self.block.append(
                    conv_block(
                        config=self.config,
                        in_channels=input_channel,
                        out_channels=output_channel,
                        stride=stride,
                        expand_ratio=expand_ratio,
                        use_se=use_se,
                        survival_probability=survival_probability,
                    )
                )
                input_channel = output_channel
                current_num_blocks += 1

        # building last several layers
        if config.depth_multiplier > 1.0:
            output_channel = make_divisible(
                1280 * config.depth_multiplier, config.depth_divisible_by, config.min_depth
            )
        else:
            output_channel = 1280

        self.top = TrillssonTop(
            config=self.config,
            in_channels=input_channel,
            out_channels=output_channel,
        )
        self.pooler = nn.AdaptiveAvgPool2d((1, 1)) if add_pooling_layer else None
        self.dense = nn.Linear(output_channel, config.output_size)

        # Initialize weights
        self.post_init()

    @add_start_docstrings_to_model_forward(TRILLSSON_EFFICIENTNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_values is None:
            raise ValueError("You have to specify input_values")
        batch_size, _, frame_width, num_mel_bins = input_values.shape
        input_values = torch.reshape(input_values, [-1, 1, frame_width, num_mel_bins])

        all_hidden_states = () if output_hidden_states else None

        hidden_states = self.stem(input_values)
        for i, layer_module in enumerate(self.block):
            hidden_states = layer_module(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        hidden_states = self.top(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.pooler is not None:
            pooled_output = self.pooler(hidden_states)
            hidden_states = pooled_output
        else:
            pooled_output = None

        hidden_states = hidden_states.view(hidden_states.size(0), -1)
        main_net_output_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.reshape([batch_size, -1, main_net_output_dim])

        hidden_states = torch.mean(hidden_states, dim=1, keepdim=False)
        last_hidden_state = self.dense(hidden_states)
        if not return_dict:
            return tuple(v for v in [last_hidden_state, pooled_output, all_hidden_states] if v is not None)

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
        )


@add_start_docstrings(
    """
    Trillsson_efficient Model with a sequence classification head on top (a linear layer over the last hidden state
    output).
    """,
    TRILLSSON_EFFICIENTNET_START_DOCSTRING,
)
class TrillssonEfficientNetForSequenceClassification(TrillssonEfficientNetPreTrainedModel):
    def __init__(self, config: TrillssonEfficientNetConfig):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.trillsson = TrillssonEfficientNetModel(config)

        last_hidden_size = self.trillsson.dense.out_features

        # Classifier head
        self.dropout = nn.Dropout(config.classifier_dropout_prob, inplace=True)
        self.classifier = nn.Linear(last_hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.trillsson.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(TRILLSSON_EFFICIENTNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_SEQ_CLASS_CHECKPOINT,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.trillsson(
            input_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs.last_hidden_state if return_dict else outputs[1]

        logits = self.classifier(self.dropout(last_hidden_state))

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
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.last_hidden_state,
        )
