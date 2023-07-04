# coding=utf-8
# Copyright 2023 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch CCT model."""


from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import ImageClassifierOutputWithNoAttention, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_cct import CctConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "CctConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "rishabbala/cct_14_7x2_384"
_EXPECTED_OUTPUT_SHAPE = [1, 384]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "rishabbala/cct_14_7x2_384"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


CCT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "rishabbala/cct_14_7x2_384",
    "rishabbala/cct_14_7x2_224"
    # See all CCT models at https://huggingface.co/models?filter=cct
]


@dataclass
class BaseModelOutputWithSeqPool(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model prior to sequential pooling.
        hidden_state_post_pool (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model post sequential pooling.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_state_post_pool: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input, drop_prob: float = 0.0, training: bool = False):
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


# Copied from transformers.models.beit.modeling_beit.BeitDropPath
class CctDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# Copied from transformers.models.cvt.modeling_cvt.CvtConvEmbeddings with Cvt->Cct
class CctConvEmbeddings(nn.Module):
    """
    Image to Conv Embedding.
    """

    def __init__(self, patch_size, num_channels, embed_dim, stride, padding):
        super().__init__()
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.patch_size = patch_size
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.normalization = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values):
        pixel_values = self.projection(pixel_values)
        batch_size, num_channels, height, width = pixel_values.shape
        hidden_size = height * width
        # rearrange "b c h w -> b (h w) c"
        pixel_values = pixel_values.view(batch_size, num_channels, hidden_size).permute(0, 2, 1)
        if self.normalization:
            pixel_values = self.normalization(pixel_values)
        # rearrange "b (h w) c" -> b c h w"
        pixel_values = pixel_values.permute(0, 2, 1).view(batch_size, num_channels, height, width)
        return pixel_values


# Copied from transformers.models.cvt.modeling_cvt.CvtSelfAttentionConvProjection with Cvt->Cct
class CctSelfAttention(nn.Module):
    def __init__(self, embed_dim, kernel_size, padding, stride):
        super().__init__()
        self.convolution = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
            groups=embed_dim,
        )
        self.normalization = nn.BatchNorm2d(embed_dim)

    def forward(self, hidden_state):
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state)
        return hidden_state


class CctStage(nn.Module):
    """
    CCT stage composed of stacked transformer layers
    """

    def __init__(
        self, embed_dim=384, num_heads=6, mlp_ratio=3, drop_rate=0.0, attention_drop_rate=0.1, drop_path_rate=0.0
    ):
        super().__init__()
        dim_feedforward = mlp_ratio * embed_dim
        self.pre_norm = nn.LayerNorm(embed_dim)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.self_attn = CctSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, attention_drop_rate=attention_drop_rate, drop_rate=drop_rate
        )
        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        self.drop_path = CctDropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.activation = F.gelu

    def forward(self, hidden_state):
        hidden_state = hidden_state + self.drop_path(self.self_attn(self.pre_norm(hidden_state)))
        hidden_state = self.norm1(hidden_state)
        hidden_state = hidden_state + self.drop_path(
            self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(hidden_state)))))
        )

        return hidden_state


class CctEncoder(nn.Module):
    """
    Class that combines CctConvEmbeddings and CctStage. Output is of type BaseModelOutputWithSeqPool if return_dict is
    set to True, else the output is a Tuple
    """

    def __init__(self, config: CctConfig, sequence_length: int):
        super().__init__()
        assert sequence_length is not None, "Sequence Length required to initialize positional embedding"

        int(config.embed_dim * config.mlp_ratio)
        self.attention_pool = nn.Linear(config.embed_dim, 1)

        if config.pos_emb_type == "learnable":
            self.positional_emb = nn.Parameter(
                self.learnable_embedding(sequence_length, config.embed_dim), requires_grad=True
            )
        else:
            self.positional_emb = nn.Parameter(
                self.sinusoidal_embedding(sequence_length, config.embed_dim), requires_grad=False
            )

        self.dropout = nn.Dropout(config.drop_rate)
        stochastic_dropout_rate = [
            x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_transformer_layers)
        ]

        self.blocks = nn.ModuleList(
            [
                CctStage(
                    config.embed_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    config.drop_rate,
                    config.attention_drop_rate,
                    stochastic_dropout_rate[i],
                )
                for i in range(config.num_transformer_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True) -> BaseModelOutputWithSeqPool:
        all_hidden_states = ()

        hidden_state = pixel_values + self.positional_emb
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)
        hidden_state = self.dropout(hidden_state)

        for blk in self.blocks:
            hidden_state = blk(hidden_state)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

        hidden_state_pre_pool = self.norm(hidden_state)
        if output_hidden_states:
            all_hidden_states = all_hidden_states[:-1] + (hidden_state_pre_pool,)

        seq_pool_attn = F.softmax(self.attention_pool(hidden_state_pre_pool), dim=1)
        hidden_state_post_pool = torch.matmul(seq_pool_attn.transpose(-1, -2), hidden_state_pre_pool).squeeze(-2)
        seq_pool_attn = seq_pool_attn.squeeze()

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state_post_pool,)

        if not return_dict:
            if output_hidden_states:
                return (hidden_state_pre_pool, hidden_state_post_pool, all_hidden_states)
            else:
                return (hidden_state_pre_pool, hidden_state_post_pool)

        return BaseModelOutputWithSeqPool(
            last_hidden_state=hidden_state_pre_pool,
            hidden_state_post_pool=hidden_state_post_pool,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )

    @staticmethod
    def learnable_embedding(sequence_length, embed_dim):
        pe = torch.zeros(1, sequence_length, embed_dim)
        return nn.init.trunc_normal_(pe, std=0.2)

    @staticmethod
    def sinusoidal_embedding(sequence_length, embed_dim):
        pe = torch.FloatTensor(
            [[p / (10000 ** (2 * (i // 2) / embed_dim)) for i in range(embed_dim)] for p in range(sequence_length)]
        )
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


# Copied from transformers.models.cvt.modeling_cvt.CvtPreTrainedModel with Cvt->Cct,cvt->cct
class CctPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CctConfig
    base_model_prefix = "cct"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, CctStage):
            if self.config.cls_token[module.stage]:
                module.cls_token.data = nn.init.trunc_normal_(
                    torch.zeros(1, 1, self.config.embed_dim[-1]), mean=0.0, std=self.config.initializer_range
                )


CCT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`CctConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CCT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`CctImageProcessor.__call__`]
            for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Cct Model transformer outputting raw hidden-states without any specific head on top.",
    CCT_START_DOCSTRING,
)
# Copied from transformers.models.cvt.modeling_cvt.CvtModel with CVT->CCT,Cvt->Cct
class CctModel(CctPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.encoder = CctEncoder(config)
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(CCT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithCLSToken,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithCLSToken]:
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
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithCLSToken(
            last_hidden_state=sequence_output,
            cls_token_value=encoder_outputs.cls_token_value,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    """
    Cct Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    CCT_START_DOCSTRING,
)
# Copied from transformers.models.cvt.modeling_cvt.CvtForImageClassification with CVT->CCT,Cvt->Cct,cvt->cct
class CctForImageClassification(CctPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.cct = CctModel(config, add_pooling_layer=False)
        self.layernorm = nn.LayerNorm(config.embed_dim[-1])
        # Classifier head
        self.classifier = (
            nn.Linear(config.embed_dim[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CCT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.cct(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        cls_token = outputs[1]
        if self.config.cls_token[-1]:
            sequence_output = self.layernorm(cls_token)
        else:
            batch_size, num_channels, height, width = sequence_output.shape
            # rearrange "b c h w -> b (h w) c"
            sequence_output = sequence_output.view(batch_size, num_channels, height * width).permute(0, 2, 1)
            sequence_output = self.layernorm(sequence_output)

        sequence_output_mean = sequence_output.mean(dim=1)
        logits = self.classifier(sequence_output_mean)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
