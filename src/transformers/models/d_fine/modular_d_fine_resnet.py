# coding=utf-8
# Copyright 2024 Baidu Inc and The HuggingFace Inc. team.
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


from ..rt_detr.configuration_rt_detr_resnet import RTDetrResNetConfig
from ..rt_detr.modeling_rt_detr_resnet import RTDetrResNetBackbone, RTDetrResNetPreTrainedModel, RTDetrResNetEmbeddings, RTDetrResNetConvLayer
from torch import nn, Tensor
import torch
from typing import List, Any, Iterator, NamedTuple


class DFineResNetStageConfig(NamedTuple):
    # stage: [in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num]
    stage1: List[Any] = [48, 48, 128, 1, False, False, 3, 6]
    stage2: List[Any] = [128, 96, 512, 1, True, False, 3, 6]
    stage3: List[Any] = [512, 192, 1024, 3, True, True, 5, 6]
    stage4: List[Any] = [1024, 384, 2048, 1, True, True, 5, 6]

    def __iter__(self) -> Iterator[List[Any]]:
        # Create an iterator over the stages
        return iter([self.stage1, self.stage2, self.stage3, self.stage4])


class DFineResNetConfig(RTDetrResNetConfig):
    model_type = "d-fine-resnet"
    def __init__(
        self,
        stem_channels=[3, 32, 48],
        stage_config=DFineResNetStageConfig(),
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        self.stem_channels = stem_channels
        self.stage_config = stage_config


class DFineResNetPreTrainedModel(RTDetrResNetPreTrainedModel):
    pass


class DFineResNetConvLayer(RTDetrResNetConvLayer):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int = 3, 
            stride: int = 1,
            groups: int = 1, 
            activation: str = "relu"):
        super().__init__(in_channels, out_channels, kernel_size, stride, activation)
        self.lab = nn.Identity()
        self.convolution = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            groups=groups, 
            padding=kernel_size // 2, 
            bias=False
        )
    
    def forward(self, input: Tensor) -> Tensor:
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.lab(hidden_state)
        return hidden_state


class DFineResNetConvLayerLight(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
    ):
        super().__init__()
        self.conv1 = DFineResNetConvLayer(
            in_chs,
            out_chs,
            kernel_size=1,
            activation=None
        )
        self.conv2 = DFineResNetConvLayer(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(
            chs,
            chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


class DFineResNetEmbeddings(RTDetrResNetEmbeddings):
    def __init__(self, config: DFineResNetConfig):
        super().__init__(config=config)

        self.embedder = nn.Sequential(
            *[
                DFineResNetConvLayer(
                    config.stem_channels[0],
                    config.stem_channels[1],
                    kernel_size=3,
                    stride=2,
                    activation=config.hidden_act,
                ),
                DFineResNetConvLayer(
                    config.stem_channels[1],
                    config.stem_channels[1] // 2,
                    kernel_size=2,
                    stride=1,
                    activation=config.hidden_act,
                ),
                DFineResNetConvLayer(
                    config.stem_channels[1] // 2,
                    config.stem_channels[1],
                    kernel_size=2,
                    stride=1,
                    activation=config.hidden_act,
                ),
                DFineResNetConvLayer(
                    config.stem_channels[1] * 2,
                    config.stem_channels[1],
                    kernel_size=3,
                    stride=2,
                    activation=config.hidden_act,
                ),
                DFineResNetConvLayer(
                    config.stem_channels[1],
                    config.stem_channels[2],
                    kernel_size=1,
                    stride=1,
                    activation=config.hidden_act,
                )
            ]
        )


class DFineResNetBasicLayer(nn.Module):
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            layer_num,
            kernel_size=3,
            residual=False,
            light_block=False,
            agg='ese'):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    DFineResNetConvLayerLight(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size
                    )
                )
            else:
                self.layers.append(
                    DFineResNetConvLayer(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                    )
                )

        # feature aggregation
        total_chs = in_chs + layer_num * mid_chs
        if agg == 'se':
            aggregation_squeeze_conv = DFineResNetConvLayer(
                total_chs,
                out_chs // 2,
                kernel_size=1,
                stride=1
            )
            aggregation_excitation_conv = DFineResNetConvLayer(
                out_chs // 2,
                out_chs,
                kernel_size=1,
                stride=1
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        else:
            aggregation_conv = DFineResNetConvLayer(
                total_chs,
                out_chs,
                kernel_size=1,
                stride=1
            )
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )
    
    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class DFineResNetStage(nn.Module):
    def __init__(
            self, 
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=True,
            light_block=False,
            kernel_size=3,
            agg='se'):
        super().__init__()
        if downsample:
            self.downsample = DFineResNetConvLayer(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                DFineResNetBasicLayer(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    agg=agg
                )
            )
        self.blocks = nn.Sequential(*blocks_list)
    
    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class DFineResNetEncoder(nn.Module):
    def __init__(self, config: DFineResNetConfig):
        super().__init__()
        self.stages = nn.ModuleList([])
        for _, stage in enumerate(config.stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage
            self.stages.append(DFineResNetStage(in_channels, mid_channels, out_channels, block_num, layer_num, downsample, light_block, kernel_size ))


class DFineResNetBackbone(RTDetrResNetBackbone):
    def __init__(self, config: DFineResNetConfig):
        super().__init__(config=config)
        self.embedder = DFineResNetEmbeddings(config=config)
        self.encoder = DFineResNetEncoder(config=config)


__all__ = ["DFineResNetConfig", "DFineResNetStageConfig", "DFineResNetBackbone", "DFineResNetPreTrainedModel"]