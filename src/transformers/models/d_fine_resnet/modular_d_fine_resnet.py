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

from ...configuration_utils import PretrainedConfig
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
)
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices
from ..rt_detr.modeling_rt_detr_resnet import RTDetrResNetBackbone, RTDetrResNetConvLayer, RTDetrResNetPreTrainedModel


class DFineResNetConfig(BackboneConfigMixin, PretrainedConfig):
    """
    Configuration class for D-FINE ResNet backbone.
    Extends RTDetrResNetConfig with D-FINE specific parameters.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        embedding_size (`int`, *optional*, defaults to 64):
            Dimensionality (hidden size) for the embedding layer.
        hidden_sizes (`List[int]`, *optional*, defaults to `[256, 512, 1024, 2048]`):
            Dimensionality (hidden size) at each stage.
        depths (`List[int]`, *optional*, defaults to `[3, 4, 6, 3]`):
            Depth (number of layers) for each stage.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in each block. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"`
            are supported.
        out_features (`List[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        out_indices (`List[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        stem_channels (`List[int]`, *optional*, defaults to `[3, 32, 48]`):
            Channel dimensions for the stem layers:
            - First number (3) is input image channels
            - Second number (32) is intermediate stem channels
            - Third number (48) is output stem channels
        stage_in_channels (`List[int]`, *optional*, defaults to `[48, 128, 512, 1024]`):
            Input channel dimensions for each stage of the backbone.
            This defines how many channels the input to each stage will have.
        stage_mid_channels (`List[int]`, *optional*, defaults to `[48, 96, 192, 384]`):
            Mid-channel dimensions for each stage of the backbone.
            This defines the number of channels used in the intermediate layers of each stage.
        stage_out_channels (`List[int]`, *optional*, defaults to `[128, 512, 1024, 2048]`):
            Output channel dimensions for each stage of the backbone.
            This defines how many channels the output of each stage will have.
        stage_num_blocks (`List[int]`, *optional*, defaults to `[1, 1, 3, 1]`):
            Number of blocks to be used in each stage of the backbone.
            This controls the depth of each stage by specifying how many convolutional blocks to stack.
        stage_downsample (`List[bool]`, *optional*, defaults to `[False, True, True, True]`):
            Indicates whether to downsample the feature maps at each stage.
            If `True`, the spatial dimensions of the feature maps will be reduced.
        stage_light_block (`List[bool]`, *optional*, defaults to `[False, False, True, True]`):
            Indicates whether to use light blocks in each stage.
            Light blocks are a variant of convolutional blocks that may have fewer parameters.
        stage_kernel_size (`List[int]`, *optional*, defaults to `[3, 3, 5, 5]`):
            Kernel sizes for the convolutional layers in each stage.
        stage_numb_of_layers (`List[int]`, *optional*, defaults to `[6, 6, 6, 6]`):
            Number of layers to be used in each block of the stage.
        use_learnable_affine_block (`bool`, *optional*, defaults to `False`):
            Whether to use Learnable Affine Blocks (LAB) in the network.
            LAB adds learnable scale and bias parameters after certain operations.
    """

    model_type = "d_fine_resnet"

    def __init__(
        self,
        num_channels=3,
        embedding_size=64,
        depths=[3, 4, 6, 3],
        hidden_sizes=[256, 512, 1024, 2048],
        hidden_act="relu",
        out_features=None,
        out_indices=None,
        stem_channels=[3, 32, 48],
        stage_in_channels=[48, 128, 512, 1024],
        stage_mid_channels=[48, 96, 192, 384],
        stage_out_channels=[128, 512, 1024, 2048],
        stage_num_blocks=[1, 1, 3, 1],
        stage_downsample=[False, True, True, True],
        stage_light_block=[False, False, True, True],
        stage_kernel_size=[3, 3, 5, 5],
        stage_numb_of_layers=[6, 6, 6, 6],
        use_learnable_affine_block=False,
        **kwargs,
    ):
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.hidden_act = hidden_act
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
        self.stem_channels = stem_channels
        self.stage_in_channels = stage_in_channels
        self.stage_mid_channels = stage_mid_channels
        self.stage_out_channels = stage_out_channels
        self.stage_num_blocks = stage_num_blocks
        self.stage_downsample = stage_downsample
        self.stage_light_block = stage_light_block
        self.stage_kernel_size = stage_kernel_size
        self.stage_numb_of_layers = stage_numb_of_layers
        self.use_learnable_affine_block = use_learnable_affine_block

        if not (
            len(stage_in_channels)
            == len(stage_mid_channels)
            == len(stage_out_channels)
            == len(stage_num_blocks)
            == len(stage_downsample)
            == len(stage_light_block)
            == len(stage_kernel_size)
            == len(stage_numb_of_layers)
        ):
            raise ValueError("All stage configuration lists must have the same length.")
        super().__init__(**kwargs)


class DFineResNetPreTrainedModel(RTDetrResNetPreTrainedModel):
    pass


class DFineResNetLearnableAffineBlock(nn.Module):
    def __init__(self, scale_value: float = 1.0, bias_value: float = 0.0):
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
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, use_learnable_affine_block: bool = False
    ):
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
        total_channels = in_channels + layer_num * middle_channels
        aggregation_squeeze_conv = DFineResNetConvLayer(
            total_channels,
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
    def __init__(self, config: DFineResNetConfig, stage_index: int, drop_path: float = 0.0):
        super().__init__()
        in_channels = config.stage_in_channels[stage_index]
        mid_channels = config.stage_mid_channels[stage_index]
        out_channels = config.stage_out_channels[stage_index]
        num_blocks = config.stage_num_blocks[stage_index]
        num_layers = config.stage_numb_of_layers[stage_index]
        downsample = config.stage_downsample[stage_index]
        light_block = config.stage_light_block[stage_index]
        kernel_size = config.stage_kernel_size[stage_index]
        use_learnable_affine_block = config.use_learnable_affine_block

        if downsample:
            self.downsample = DFineResNetConvLayer(
                in_channels, in_channels, kernel_size=3, stride=2, groups=in_channels, activation=None
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(num_blocks):
            blocks_list.append(
                DFineResNetBasicLayer(
                    in_channels if i == 0 else out_channels,
                    mid_channels,
                    out_channels,
                    num_layers,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    drop_path=drop_path,
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
        for stage_index in range(len(config.stage_in_channels)):
            resnet_stage = DFineResNetStage(config, stage_index)
            self.stages.append(resnet_stage)

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
        self.embedder = DFineResNetEmbeddings(config)
        self.encoder = DFineResNetEncoder(config)


__all__ = ["DFineResNetConfig", "DFineResNetBackbone", "DFineResNetPreTrainedModel"]
