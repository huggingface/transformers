# coding=utf-8
# Copyright 2022 Snapchat Research and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Efficientformer model."""


from typing import Optional, Union
import itertools

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_efficientformer import EfficientformerConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "EfficientformerConfig"
_FEAT_EXTRACTOR_FOR_DOC = "ViTFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "efficientformer-l1"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "snap-research/efficientformer-l1"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"


EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "efficientformer-l1",
    # See all Efficientformer models at https://huggingface.co/models?filter=efficientformer
]


class EfficientformerPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)`.
    """

    def __init__(self, config, num_channels, embed_dim, norm_layer = nn.BatchNorm2d):
        super().__init__()
        self.num_channels = num_channels

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=config.downsample_patch_size, stride=config.downsample_stride, padding=config.downsample_pad)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        embeddings = self.projection(pixel_values)
        embeddings = self.norm(embeddings)

        return (embeddings,)

class EfficientformerSelfAttention(nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8, attn_ratio=4, resolution=7):
        super().__init__()

        self.num_heads = num_heads
        self.key_dim = key_dim        
        self.attn_ratio = attn_ratio
        self.scale = key_dim ** -0.5
        self.total_key_dim = key_dim * num_heads
        self.expanded_key_dim = int(attn_ratio * key_dim)
        self.total_expanded_key_dim = self.expanded_key_dim * num_heads
        hidden_size = self.total_expanded_key_dim + self.total_key_dim * 2

        self.qkv = nn.Linear(dim, hidden_size)
        self.projection = nn.Linear(self.total_expanded_key_dim, dim)

        points = list(itertools.product(range(resolution), range(resolution)))
        num_points = len(points)
        attention_offsets = {}
        idxs = []
        for point_1 in points:
            for point_2 in points:
                offset = (abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(num_points, num_points))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, hidden_states, output_attentions = False): 
        batch_size, sequence_length, num_channels = hidden_states.shape
        qkv = self.qkv(hidden_states)
        query_layer, key_layer, value_layer = qkv.reshape(batch_size, sequence_length, self.num_heads, -1).split([self.key_dim, self.key_dim, self.expanded_key_dim], dim=3)
        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)

        attention_probs = ((query_layer @ key_layer.transpose(-2, -1)) * self.scale + (self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab))
        attention_probs = attention_probs.softmax(dim=-1)
        context_layer = (attention_probs @ value_layer).transpose(1, 2).reshape(batch_size, sequence_length, self.total_expanded_key_dim)
        context_layer = self.projection(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

class EfficientformerConvStem(nn.Module):
    def __init__(self, config, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(config.num_channels, out_channels // 2, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels // 2)

        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.ReLU()

    def forward(self, pixel_values):

        features = self.norm1(self.conv1(pixel_values))
        features = self.activation(features)
        features = self.norm2(self.conv2(features))
        features = self.activation(features)

        return features

# Copied from transformers.models.poolformer.modeling_poolformer.PoolFormerPooling
class EfficientformerPooling(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, hidden_states):
        output = self.pool(hidden_states) - hidden_states
        output = (output,)

        return output

class EfficientformerDenseMlp(nn.Module):
    def __init__(self, config, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = ACT2FN[config.hidden_act]
        self.drop1 = nn.Dropout(config.hidden_dropout_prob)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.drop1(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.drop2(hidden_states)

        return hidden_states

class EfficientformerConvMlp(nn.Module):

    def __init__(self, config, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.fc1(x)

        x = self.norm1(x)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)

        x = self.norm2(x)

        x = self.drop(x)
        return x

# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input, drop_prob=0.0, training=False, scale_by_keep=True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->Efficientformer
class EfficientformerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)

class EfficientformerMeta3D(nn.Module):
    
    def __init__(self, config, dim, norm_layer=nn.LayerNorm, drop_path=0.):

        super().__init__()

        self.norm1 = norm_layer(dim, eps=config.layer_norm_eps)
        self.token_mixer = EfficientformerSelfAttention(dim)
        self.norm2 = norm_layer(dim, eps=config.layer_norm_eps)
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        self.mlp = EfficientformerDenseMlp(config, in_features=dim, hidden_features=mlp_hidden_dim)

        self.drop_path = EfficientformerDropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = config.use_layer_scale
        if config.use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, hidden_states, output_attentions = False):
        self_attention_outputs = self.token_mixer(self.norm1(hidden_states), output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if self.use_layer_scale:
            hidden_states = hidden_states + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * attention_output)
            layer_output = self.drop_path(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(self.norm2(hidden_states)))
        else:
            hidden_states = hidden_states + self.drop_path(attention_output)
            layer_output = self.drop_path(self.mlp(self.norm2(hidden_states)))

        hidden_states = hidden_states + layer_output
        
        outputs = (hidden_states,) + outputs

        return hidden_states

class EfficientformerMeta4D(nn.Module):
    
    def __init__(self, config, dim, drop_path=0.):
        super().__init__()

        self.token_mixer = EfficientformerPooling(pool_size=config.pool_size)
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        self.mlp = EfficientformerConvMlp(config, in_features=dim, hidden_features=mlp_hidden_dim, drop=config.hidden_dropout_prob)

        self.drop_path = EfficientformerDropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = config.use_layer_scale
        if config.use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, hidden_states, output_attentions = False):
        self_attention_outputs = self.token_mixer(hidden_states)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if self.use_layer_scale:
            hidden_states = hidden_states + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * attention_output)
            layer_outputs = self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(hidden_states))
        else:
            hidden_states = hidden_states + self.drop_path(attention_output)
            layer_outputs = self.drop_path(self.mlp(hidden_states))

        hidden_states = hidden_states + layer_outputs

        outputs = (hidden_states,) + outputs

        return outputs

class EfficientformerFlat(nn.Module):
    
    def __init__(self, ):
        super().__init__()

    def forward(self, hidden_states, output_attentions = False):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = (hidden_states,)
        return hidden_states

class EfficientformerStage(nn.Module):
    def __init__(self, config, dim, index, layers):
        super().__init__()
        blocks = nn.ModuleList([])
        if index == 3 and config.vit_num == layers[index]:
            blocks.append(EfficientformerFlat())
        for block_idx in range(layers[index]):
            block_dpr = config.drop_path_rate * (
                    block_idx + sum(layers[:index])) / (sum(layers) - 1)
            if index == 3 and layers[index] - block_idx <= config.vit_num:
                blocks.append(EfficientformerMeta3D(config, dim, drop_path=block_dpr))
            else:
                blocks.append(EfficientformerMeta4D(config, dim, drop_path=block_dpr))
                if index == 3 and layers[index] - block_idx - 1 == config.vit_num:
                    blocks.append(EfficientformerFlat())

        self.blocks = blocks
    
    def forward(self, hidden_states, output_hidden_states = False, output_attentions = False, return_dict = False):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions and isinstance(layer_module, EfficientformerMeta3D):
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class EfficientformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        network = nn.ModuleList([])
        for i in range(len(config.layers)):
            stage = EfficientformerStage(config, config.embed_dims[i], i, config.layers)
            network.append(stage)
            if i >= len(config.layers) - 1:
                break
            if config.downsamples[i] or config.embed_dims[i] != config.embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    EfficientformerPatchEmbeddings(config, num_channels=config.embed_dims[i], embed_dim=config.embed_dims[i + 1])
                )

        self.network = nn.Sequential(*network)
    
    def forward(self, hidden_states, output_hidden_states = False, output_attentions = False, return_dict = True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.network):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, output_attentions)


            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

# Copied from transformers.models.vit.modeling_vit.ViTPreTrainedModel with ViT->Efficientformer,vit->efficientformer
class EfficientformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EfficientformerConfig
    base_model_prefix = "efficientformer"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


EFFICIENTFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`EfficientformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

EFFICIENTFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`ViTFeatureExtractor`]. See
            [`ViTFeatureExtractor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Efficientformer Model transformer outputting raw hidden-states without any specific head on top.",
    EFFICIENTFORMER_START_DOCSTRING,
)
# Copied from transformers.models.vit.modeling_vit.ViTModel with VIT->EFFICIENTFORMER,ViT->Efficientformer
class EfficientformerModel(EfficientformerPreTrainedModel):
    def __init__(self, config: EfficientformerConfig):
        super().__init__(config)
        self.config = config

        self.patch_embed = EfficientformerConvStem(config, config.embed_dims[0])

        self.encoder = EfficientformerEncoder(config)
        #self.norm = config.norm_layer(config.embed_dims[-1])
        self.norm = nn.LayerNorm(config.embed_dims[-1], eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> EfficientformerPatchEmbeddings:
        return self.embeddings.patch_embeddings

    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.patch_embed(pixel_values)
        encoder_outputs = self.encoder(embedding_output)

        sequence_output = encoder_outputs[0]
        sequence_output = self.norm(sequence_output)

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

@add_start_docstrings(
    """
    Efficientformer Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    EFFICIENTFORMER_START_DOCSTRING,
)
# Copied from transformers.models.vit.modeling_vit.ViTForImageClassification with VIT->EFFICIENTFORMER,ViT->Efficientformer,vit->efficientformer
class EfficientformerForImageClassification(EfficientformerPreTrainedModel):
    def __init__(self, config: EfficientformerConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.efficientformer = EfficientformerModel(config)

        # Classifier head
        self.classifier = nn.Linear(config.embed_dims[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.dist = config.distillation
        
        if self.dist:
            self.dist_head = nn.Linear(config.embed_dims[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.efficientformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]

        if self.dist:
            logits = self.classifier(sequence_output.mean(-2)), self.dist_head(sequence_output.mean(-2))
            if not self.training:
                logits = (logits[0] + logits[1]) / 2
        else:
            logits = self.classifier(sequence_output.mean(-2))

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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
