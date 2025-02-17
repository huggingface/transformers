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


import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
)
from ..rt_detr.configuration_rt_detr_resnet import RTDetrResNetConfig
from ..rt_detr.modeling_rt_detr_resnet import RTDetrResNetBackbone, RTDetrResNetConvLayer, RTDetrResNetPreTrainedModel


class DFineResNetConfig(RTDetrResNetConfig):
    """
    Configuration class for D-FINE ResNet backbone.
    Extends RTDetrResNetConfig with D-FINE specific parameters.

    Args:
        stem_channels (`List[int]`, *optional*, defaults to [3, 32, 48]):
            Channel dimensions for the stem layers:
            - First number (3) is input image channels
            - Second number (32) is intermediate stem channels
            - Third number (48) is output stem channels
        stage_config (`List[List[Any]]` *optional*):
            Configuration for the four stages of the backbone.
            [in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num]
        use_learnable_affine_block (`bool`, *optional*, defaults to False):
            Whether to use Learnable Affine Blocks (LAB) in the network.
            LAB adds learnable scale and bias parameters after certain operations.
        **super_kwargs:
            Additional arguments from RTDetrResNetConfig, including standard
            ResNet parameters like hidden_act, layer_norm_eps, etc.
    """

    model_type = "d_fine_resnet"
    layer_types = ["basic", "bottleneck"]

    def __init__(
        self,
        stem_channels=[3, 32, 48],
        stage_config=[
            [48, 48, 128, 1, False, False, 3, 6],
            [128, 96, 512, 1, True, False, 3, 6],
            [512, 192, 1024, 3, True, True, 5, 6],
            [1024, 384, 2048, 1, True, True, 5, 6],
        ],
        use_learnable_affine_block=False,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        self.stem_channels = stem_channels
        self.stage_config = stage_config
        self.stage_config = stage_config
        self.use_learnable_affine_block = use_learnable_affine_block


class DFineResNetPreTrainedModel(RTDetrResNetPreTrainedModel):
    pass


class DFineResNetLearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.scale * hidden_state + self.bias
        return hidden_state


class DFineResNetConvLayer(RTDetrResNetConvLayer):
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
        super().__init__(in_channels, out_channels, kernel_size, stride, activation)
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
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


class DFineResNetBackbone(RTDetrResNetBackbone):
    def __init__(self, config: DFineResNetConfig):
        super().__init__(config=config)
        self.embedder = DFineResNetEmbeddings(config=config)
        self.encoder = DFineResNetEncoder(config=config)


__all__ = ["DFineResNetConfig", "DFineResNetBackbone", "DFineResNetPreTrainedModel"]
