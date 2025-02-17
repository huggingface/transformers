#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/d_fine/modular_d_fine_resnet.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_d_fine_resnet.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
# coding=utf-8
# Copyright 2025 Baidu Inc and The HuggingFace Inc. team.
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


import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...utils.backbone_utils import BackboneMixin
from .configuration_d_fine_resnet import DFineResNetConfig


# General docstring
_CONFIG_FOR_DOC = "DFineResNetConfig"


class DFineResNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DFineResNetConfig
    base_model_prefix = "resnet"
    main_input_name = "pixel_values"
    _no_split_modules = ["DFineResNetConvLayer", "DFineResNetShortCut"]

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        # copied from the `reset_parameters` method of `class Linear(Module)` in `torch`.
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


class DFineResNetLearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.scale * hidden_state + self.bias
        return hidden_state


class DFineResNetConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: str = "relu",
        use_learnable_affine_block: bool = False,
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = ACT2FN[activation] if activation is not None else nn.Identity()
        if activation and use_learnable_affine_block:
            self.lab = DFineResNetLearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.lab(hidden_state)
        return hidden_state


class DFineResNetConvLayerLight(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_learnable_affine_block=False):
        super().__init__()
        self.conv1 = DFineResNetConvLayer(
            in_channels,
            out_channels,
            kernel_size=1,
            activation=None,
            use_learnable_affine_block=use_learnable_affine_block,
        )
        self.conv2 = DFineResNetConvLayer(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            use_learnable_affine_block=use_learnable_affine_block,
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state


class DFineResNetEseModule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_state: Tensor) -> Tensor:
        identity = hidden_state
        hidden_state = hidden_state.mean((2, 3), keepdim=True)
        hidden_state = self.conv(hidden_state)
        hidden_state = self.sigmoid(hidden_state)
        hidden_state = torch.mul(identity, hidden_state)
        return hidden_state


class DFineResNetEmbeddings(nn.Module):
    def __init__(self, config: DFineResNetConfig):
        super().__init__()

        self.stem1 = DFineResNetConvLayer(
            config.stem_channels[0],
            config.stem_channels[1],
            kernel_size=3,
            stride=2,
            activation=config.hidden_act,
            use_learnable_affine_block=config.use_learnable_affine_block,
        )
        self.stem2a = DFineResNetConvLayer(
            config.stem_channels[1],
            config.stem_channels[1] // 2,
            kernel_size=2,
            stride=1,
            activation=config.hidden_act,
            use_learnable_affine_block=config.use_learnable_affine_block,
        )
        self.stem2b = DFineResNetConvLayer(
            config.stem_channels[1] // 2,
            config.stem_channels[1],
            kernel_size=2,
            stride=1,
            activation=config.hidden_act,
            use_learnable_affine_block=config.use_learnable_affine_block,
        )
        self.stem3 = DFineResNetConvLayer(
            config.stem_channels[1] * 2,
            config.stem_channels[1],
            kernel_size=3,
            stride=2,
            activation=config.hidden_act,
            use_learnable_affine_block=config.use_learnable_affine_block,
        )
        self.stem4 = DFineResNetConvLayer(
            config.stem_channels[1],
            config.stem_channels[2],
            kernel_size=1,
            stride=1,
            activation=config.hidden_act,
            use_learnable_affine_block=config.use_learnable_affine_block,
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)
        self.num_channels = config.num_channels

    def forward(self, pixel_values: Tensor) -> Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embedding = self.stem1(pixel_values)
        embedding = F.pad(embedding, (0, 1, 0, 1))
        emb_stem_2a = self.stem2a(embedding)
        emb_stem_2a = F.pad(emb_stem_2a, (0, 1, 0, 1))
        emb_stem_2a = self.stem2b(emb_stem_2a)
        pooled_emb = self.pool(embedding)
        embedding = torch.cat([pooled_emb, emb_stem_2a], dim=1)
        embedding = self.stem3(embedding)
        embedding = self.stem4(embedding)
        return embedding


class DFineResNetBasicLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
        layer_num: int,
        kernel_size: int = 3,
        residual: bool = False,
        light_block: bool = False,
        aggregation: str = "ese",
        drop_path: float = 0.0,
        use_learnable_affine_block: bool = False,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            temp_in_channels = in_channels if i == 0 else middle_channels
            if light_block:
                block = DFineResNetConvLayerLight(
                    in_channels=temp_in_channels,
                    out_channels=middle_channels,
                    kernel_size=kernel_size,
                    use_learnable_affine_block=use_learnable_affine_block,
                )
            else:
                block = DFineResNetConvLayer(
                    in_channels=temp_in_channels,
                    out_channels=middle_channels,
                    kernel_size=kernel_size,
                    use_learnable_affine_block=use_learnable_affine_block,
                    stride=1,
                )
            self.layers.append(block)

        # feature aggregation
        total_chs = in_channels + layer_num * middle_channels
        if aggregation == "se":
            aggregation_squeeze_conv = DFineResNetConvLayer(
                total_chs,
                out_channels // 2,
                kernel_size=1,
                stride=1,
                use_learnable_affine_block=use_learnable_affine_block,
            )
            aggregation_excitation_conv = DFineResNetConvLayer(
                out_channels // 2,
                out_channels,
                kernel_size=1,
                stride=1,
                use_learnable_affine_block=use_learnable_affine_block,
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        else:
            aggregation_conv = DFineResNetConvLayer(
                total_chs, out_channels, kernel_size=1, stride=1, use_learnable_affine_block=use_learnable_affine_block
            )
            att = DFineResNetEseModule(out_channels)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )
        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, hidden_state: Tensor) -> Tensor:
        identity = hidden_state
        output = [hidden_state]
        for layer in self.layers:
            hidden_state = layer(hidden_state)
            output.append(hidden_state)
        hidden_state = torch.cat(output, dim=1)
        hidden_state = self.aggregation(hidden_state)
        if self.residual:
            hidden_state = self.drop_path(hidden_state) + identity
        return hidden_state


class DFineResNetStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        block_num: int,
        layer_num: int,
        downsample: bool = True,
        light_block: bool = False,
        kernel_size: int = 3,
        aggregation: str = "se",
        drop_path: float = 0.0,
        use_learnable_affine_block: bool = False,
    ):
        super().__init__()
        if downsample:
            self.downsample = DFineResNetConvLayer(
                in_channels, in_channels, kernel_size=3, stride=2, groups=in_channels, activation=None
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                DFineResNetBasicLayer(
                    in_channels if i == 0 else out_channels,
                    mid_channels,
                    out_channels,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    aggregation=aggregation,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                    use_learnable_affine_block=use_learnable_affine_block,
                )
            )
        self.blocks = nn.ModuleList(blocks_list)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.downsample(hidden_state)
        for block in self.blocks:
            hidden_state = block(hidden_state)
        return hidden_state


class DFineResNetEncoder(nn.Module):
    def __init__(self, config: DFineResNetConfig):
        super().__init__()
        self.stages = nn.ModuleList([])
        for _, stage in enumerate(config.stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage
            self.stages.append(
                DFineResNetStage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_learnable_affine_block=config.use_learnable_affine_block,
                )
            )

    def forward(
        self, hidden_state: Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> BaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        for stage in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage(hidden_state)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


DFine_RESNET_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`DFineResNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DFine_RESNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`DFineImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """
    ResNet backbone, to be used with frameworks like DFine.
    """,
    DFine_RESNET_START_DOCSTRING,
)
class DFineResNetBackbone(DFineResNetPreTrainedModel, BackboneMixin):
    def __init__(self, config: DFineResNetConfig):
        super().__init__(config)
        super()._init_backbone(config)

        self.num_features = [config.embedding_size] + config.hidden_sizes
        self.embedder = DFineResNetEmbeddings(config=config)
        self.encoder = DFineResNetEncoder(config=config)

        # initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DFine_RESNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import DFineResNetConfig, DFineResNetBackbone
        >>> import torch

        >>> config = DFineResNetConfig()
        >>> model = DFineResNetBackbone(config)

        >>> pixel_values = torch.randn(1, 3, 224, 224)

        >>> with torch.no_grad():
        ...     outputs = model(pixel_values)

        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 2048, 7, 7]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        embedding_output = self.embedder(pixel_values)

        outputs = self.encoder(embedding_output, output_hidden_states=True, return_dict=True)

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


__all__ = ["DFineResNetBackbone", "DFineResNetPreTrainedModel"]
