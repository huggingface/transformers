# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch LeViT model."""

import itertools
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
    ModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_levit import LevitConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "LevitConfig"
_FEAT_EXTRACTOR_FOR_DOC = "LevitFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "anugunj/levit-128S"
_EXPECTED_OUTPUT_SHAPE = [1, 16, 384]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "anugunj/levit-128S"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

LEVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "anugunj/levit-128S",
    # See all LeViT models at https://huggingface.co/models?filter=levit
]


class LevitConvEmbeddings(nn.Module):
    """
    LeViT Conv Embeddings with Batch Norm
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bn_weight_init=1
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, embeddings):
        embeddings = self.convolution(embeddings)
        embeddings = self.bn(embeddings)
        return embeddings


class LevitPatchEmbeddings(nn.Module):
    """
    LeViT patch embeddings
    """

    def __init__(self, config):
        super().__init__()
        self.embedding_layer_1 = LevitConvEmbeddings(
            config.num_channels, config.embed_dim[0] // 8, config.kernel_size, config.stride, config.padding
        )
        self.activation_layer_1 = nn.Hardswish()

        self.embedding_layer_2 = LevitConvEmbeddings(
            config.embed_dim[0] // 8, config.embed_dim[0] // 4, config.kernel_size, config.stride, config.padding
        )
        self.activation_layer_2 = nn.Hardswish()

        self.embedding_layer_3 = LevitConvEmbeddings(
            config.embed_dim[0] // 4, config.embed_dim[0] // 2, config.kernel_size, config.stride, config.padding
        )
        self.activation_layer_3 = nn.Hardswish()

        self.embedding_layer_4 = LevitConvEmbeddings(
            config.embed_dim[0] // 2, config.embed_dim[0], config.kernel_size, config.stride, config.padding
        )

    def forward(self, pixel_values):
        embeddings = self.embedding_layer_1(pixel_values)
        embeddings = self.activation_layer_1(embeddings)
        embeddings = self.embedding_layer_2(embeddings)
        embeddings = self.activation_layer_2(embeddings)
        embeddings = self.embedding_layer_3(embeddings)
        embeddings = self.activation_layer_3(embeddings)
        embeddings = self.embedding_layer_4(embeddings)
        return embeddings


class MLPLayerWithBN(nn.Module):
    """
    MLP Layer with Batch Norm
    """

    def __init__(self, input_dim, output_dim, bn_weight_init=1):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim, bias=False)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, hidden_state):
        hidden_state = self.linear(hidden_state)
        hidden_state = self.bn(hidden_state.flatten(0, 1)).reshape_as(hidden_state)
        return hidden_state


class LevitSubsample(nn.Module):
    """
    Subsampling Module
    """

    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, hidden_state):
        batch_size, N, channels = hidden_state.shape
        hidden_state = hidden_state.view(batch_size, self.resolution, self.resolution, channels)[
            :, :: self.stride, :: self.stride
        ].reshape(batch_size, -1, channels)
        return hidden_state


class LevitAttention(nn.Module):
    """
    LeViT Attention Module
    """

    def __init__(self, embed_dim, key_dim, num_heads, attention_ratio, resolution):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio
        self.out_dim_qkv = attention_ratio * key_dim * num_heads + key_dim * num_heads * 2
        self.out_dim_proj = attention_ratio * key_dim * num_heads

        self.qkv = MLPLayerWithBN(embed_dim, self.out_dim_qkv)
        self.activation = nn.Hardswish()
        self.projection = MLPLayerWithBN(self.out_dim_proj, embed_dim, bn_weight_init=0)

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets, idxs = {}, []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.ab = {}
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.ab:
            self.ab = {}  # clear ab cache

    def get_attention_biases(self, device):
        if self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.ab:
                self.ab[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.ab[device_key]

    def forward(self, hidden_state):
        batch_size, N, channels = hidden_state.shape
        qkv = self.qkv(hidden_state)
        query, key, value = qkv.view(batch_size, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.attention_ratio * self.key_dim], dim=3
        )
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention = query @ key.transpose(-2, -1) * self.scale + self.get_attention_biases(hidden_state.device)
        attention = attention.softmax(dim=-1)
        hidden_state = (attention @ value).transpose(1, 2).reshape(batch_size, N, self.out_dim_proj)
        hidden_state = self.projection(self.activation(hidden_state))
        return hidden_state


class LevitAttentionSubsample(nn.Module):
    """
    LeViT Attention Subsample Module
    """

    def __init__(self, input_dim, output_dim, key_dim, num_heads, attention_ratio, stride, resolution, resolution_):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio
        self.out_dim_kv = attention_ratio * key_dim * num_heads + key_dim * num_heads
        self.out_dim_proj = attention_ratio * key_dim * num_heads
        self.resolution_ = resolution_

        self.kv = MLPLayerWithBN(input_dim, self.out_dim_kv)
        self.q_subsample = LevitSubsample(stride, resolution)
        self.q = MLPLayerWithBN(input_dim, key_dim * num_heads)
        self.activation = nn.Hardswish()
        self.projection = MLPLayerWithBN(self.out_dim_proj, output_dim)

        self.ab = {}

        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(range(resolution_), range(resolution_)))
        N, N_ = len(points), len(points_)
        attention_offsets, idxs = {}, []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (abs(p1[0] * stride - p2[0] + (size - 1) / 2), abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(N_, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.ab:
            self.ab = {}  # clear ab cache

    def get_attention_biases(self, device):
        if self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.ab:
                self.ab[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.ab[device_key]

    def forward(self, hidden_state):
        batch_size, N, channels = hidden_state.shape
        key, value = (
            self.kv(hidden_state)
            .view(batch_size, N, self.num_heads, -1)
            .split([self.key_dim, self.attention_ratio * self.key_dim], dim=3)
        )
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        query = self.q(self.q_subsample(hidden_state))
        query = query.view(batch_size, self.resolution_**2, self.num_heads, self.key_dim).permute(0, 2, 1, 3)

        attention = query @ key.transpose(-2, -1) * self.scale + self.get_attention_biases(hidden_state.device)
        attention = attention.softmax(dim=-1)
        hidden_state = (attention @ value).transpose(1, 2).reshape(batch_size, -1, self.out_dim_proj)
        hidden_state = self.projection(self.activation(hidden_state))
        return hidden_state


class LevitMLPBlock(nn.Module):
    """
    MLP Residual Block a.k.a MLP 2X
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear_up = MLPLayerWithBN(input_dim, hidden_dim)
        self.activation = nn.Hardswish()
        self.linear_down = MLPLayerWithBN(hidden_dim, input_dim)

    def forward(self, hidden_state):
        hidden_state = self.linear_up(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.linear_down(hidden_state)
        return hidden_state


class LevitResidualBlock(nn.Module):
    """
    Residual Block for LeViT
    """

    def __init__(self, module, drop_rate):
        super().__init__()
        self.module = module
        self.drop_rate = drop_rate

    def forward(self, x):
        if self.training and self.drop_rate > 0:
            x = (
                x
                + self.module(x)
                * torch.rand(x.size(0), 1, 1, device=x.device).ge_(self.drop_rate).div(1 - self.drop_rate).detach()
            )
            return x
        else:
            x = x + self.module(x)
            return x


class LevitEncoder(nn.Module):
    """
    LeViT Encoder Module
    """

    def __init__(self, config):
        super().__init__()
        self.num_features = config.embed_dim[-1]
        self.embed_dim = config.embed_dim

        resolution = config.image_size // config.patch_size

        self.patch_embeddings = LevitPatchEmbeddings(config)

        self.blocks = []
        config.down_ops.append([""])

        for idx, (ed, kd, dpt, nh, attn_r, mlp_r, do) in enumerate(
            zip(
                config.embed_dim,
                config.key_dim,
                config.depth,
                config.num_heads,
                config.attention_ratio,
                config.mlp_ratio,
                config.down_ops,
            )
        ):
            for _ in range(dpt):
                self.blocks.append(
                    LevitResidualBlock(LevitAttention(ed, kd, nh, attn_r, resolution), config.drop_path)
                )
                if mlp_r > 0:
                    hidden_dim = ed * mlp_r
                    self.blocks.append(LevitResidualBlock(LevitMLPBlock(ed, hidden_dim), config.drop_path))

            if do[0] == "Subsample":
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    LevitAttentionSubsample(
                        *config.embed_dim[idx : idx + 2],
                        key_dim=do[1],
                        num_heads=do[2],
                        attention_ratio=do[3],
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_,
                    )
                )
                resolution = resolution_
                if do[4] > 0:
                    hidden_dim = config.embed_dim[idx + 1] * do[4]
                    self.blocks.append(
                        LevitResidualBlock(LevitMLPBlock(config.embed_dim[idx + 1], hidden_dim), config.drop_path)
                    )

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        embeddings = self.patch_embeddings(pixel_values)
        hidden_state = embeddings.flatten(2).transpose(1, 2)

        for layers in self.blocks:
            hidden_state = layers(hidden_state)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
        )


class LevitClassificationLayer(nn.Module):
    """
    LeViT Classification Layer
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.bn(x)
        x = self.linear(x)
        return x


class LevitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LevitConfig
    base_model_prefix = "levit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LevitModel):
            module.gradient_checkpointing = value


LEVIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LevitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LEVIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoFeatureExtractor`]. See
            [`AutoFeatureExtractor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Levit model outputting raw features without any specific head on top.",
    LEVIT_START_DOCSTRING,
)
class LevitModel(LevitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = LevitEncoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(LEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        encoder_outputs = self.encoder(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]

        # global average pooling, (N, C, H, W) -> (N, C)
        pooled_output = last_hidden_state.mean([-2, -1])

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    """
    Levit Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    LEVIT_START_DOCSTRING,
)
class LevitForImageClassification(LevitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.levit = LevitModel(config)

        # Classifier head
        self.classifier = (
            LevitClassificationLayer(config.embed_dim[-1], config.num_labels)
            if config.num_labels > 0
            else torch.nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(LEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Examples:

        ```python
        >>> from transformers import LevitFeatureExtractor, LevitForImageClassification
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> torch.manual_seed(3)  # doctest: +IGNORE_RESULT
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # note: we are loading a LevitForImageClassificationWithTeacher from the hub here
        >>> # there maybe cases where prediction might be wrong
        >>> feature_extractor = LevitFeatureExtractor.from_pretrained("anugunj/levit-128S")
        >>> model = LevitForImageClassification.from_pretrained("anugunj/levit-128S")

        >>> inputs = feature_extractor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
        >>> # model predicts one of the 1000 ImageNet classes
        >>> predicted_class_idx = logits.argmax(-1).item()
        >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
        Predicted class: tabby, tabby cat
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.levit(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        sequence_output = outputs[0]
        sequence_output = sequence_output.mean(1)
        logits = self.classifier(sequence_output)

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

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


@dataclass
class LevitForImageClassificationWithTeacherOutput(ModelOutput):
    """
    Output type of [`LevitForImageClassificationWithTeacher`].

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores as the average of the cls_logits and distillation logits.
        cls_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
            class token).
        distillation_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
            distillation token).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
    """

    logits: torch.FloatTensor = None
    cls_logits: torch.FloatTensor = None
    distillation_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@add_start_docstrings(
    """
    LeViT Model transformer with image classification heads on top (a linear layer on top of the final hidden state and
    a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet. .. warning::
           This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
           supported.
    """,
    LEVIT_START_DOCSTRING,
)
class LevitForImageClassificationWithTeacher(LevitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.levit = LevitModel(config)

        # Classifier head
        self.classifier = (
            LevitClassificationLayer(config.embed_dim[-1], config.num_labels)
            if config.num_labels > 0
            else torch.nn.Identity()
        )
        if config.distillation:
            self.classifier_distill = (
                LevitClassificationLayer(config.embed_dim[-1], config.num_labels)
                if config.num_labels > 0
                else torch.nn.Identity()
            )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(LEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=LevitForImageClassificationWithTeacherOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.levit(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        sequence_output = outputs[0]
        sequence_output = sequence_output.mean(1)
        cls_logits, distill_logits = self.classifier(sequence_output), self.classifier_distill(sequence_output)
        logits = (cls_logits + distill_logits) / 2

        if not return_dict:
            output = (logits, cls_logits, distill_logits) + outputs[2:]
            return output

        return LevitForImageClassificationWithTeacherOutput(
            logits=logits,
            cls_logits=cls_logits,
            distillation_logits=distill_logits,
            hidden_states=outputs.hidden_states,
        )
