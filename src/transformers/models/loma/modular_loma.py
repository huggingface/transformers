# Copyright 2026 the HuggingFace Team. All rights reserved.
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
from torch import nn
from torch.nn import functional as F

from ..lightglue.configuration_lightglue import LightGlueConfig
from ..lightglue.image_processing_lightglue import LightGlueImageProcessor, LightGlueImageProcessorKwargs
from ..lightglue.image_processing_pil_lightglue import LightGlueImageProcessorPil
from ..lightglue.modeling_lightglue import (
    LightGlueAttention,
    LightGlueForKeypointMatching,
    LightGlueKeypointMatchingOutput,
    LightGlueMatchAssignmentLayer,
    LightGlueMLP,
    LightGluePositionalEncoder,
    LightGluePreTrainedModel,
    LightGlueTokenConfidenceLayer,
    LightGlueTransformerLayer,
)


class LoMaConfig(LightGlueConfig):
    r"""
    keypoint_detector_config (`Union[AutoConfig, dict]`, *optional*, defaults to `SuperPointConfig`):
        Configuration of the keypoint detector. The initial LoMa integration supports SuperPoint.
    input_descriptor_dim (`int`, *optional*, defaults to 256):
        Dimension of the local descriptors supplied by the descriptor network.
    descriptor_dim (`int`, *optional*, defaults to 256):
        Dimension of the descriptors used by the matching transformer.
    attention_head_dim (`int`, *optional*, defaults to 64):
        Dimension of each attention head. The number of attention heads is derived from `descriptor_dim` when it is
        not specified.
    num_hidden_layers (`int`, *optional*, defaults to 9):
        Number of self- and cross-attention layers in the matching transformer.
    filter_threshold (`float`, *optional*, defaults to 0.1):
        Confidence threshold used to retain mutual matches.
    depth_confidence (`float`, *optional*, defaults to -1.0):
        Compatibility setting for the inherited model skeleton. LoMa does not use adaptive early stopping.
    width_confidence (`float`, *optional*, defaults to -1.0):
        Compatibility setting for the inherited model skeleton. LoMa does not use adaptive keypoint pruning.
    positional_encoding_type (`str`, *optional*, defaults to `"learnable"`):
        Type of Fourier positional encoding applied in self-attention. Supported values are `"learnable"` and
        `"fixed"`.
    positional_encoding_gamma (`float`, *optional*, defaults to 1.0):
        Frequency scale used by the Fourier positional encoding.
    descriptor_hidden_blocks (`int`, *optional*, defaults to 5):
        Number of depthwise refinement blocks at each decoder scale in the local descriptor network.

    Examples:
        ```python
        >>> from transformers import LoMaConfig

        >>> config = LoMaConfig()
        >>> config.num_attention_heads
        4
        ```
    """

    model_type = "loma"

    input_descriptor_dim: int = 256
    descriptor_dim: int = 256
    attention_head_dim: int | None = None
    num_hidden_layers: int = 9
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    filter_threshold: float = 0.1
    positional_encoding_type: str = "learnable"
    positional_encoding_gamma: float = 1.0
    descriptor_hidden_blocks: int = 5

    # LoMa does not use LightGlue's adaptive early stopping or point pruning. These fields remain temporarily so the
    # inherited skeleton stays executable until the LoMa matching transformer replaces it.
    depth_confidence: float = -1.0
    width_confidence: float = -1.0

    def __post_init__(self, **kwargs):
        if self.num_attention_heads is None:
            if self.attention_head_dim is None:
                self.attention_head_dim = 64
            if self.descriptor_dim % self.attention_head_dim != 0:
                raise ValueError("descriptor_dim must be divisible by attention_head_dim")
            self.num_attention_heads = self.descriptor_dim // self.attention_head_dim
        elif self.attention_head_dim is None:
            self.attention_head_dim = self.descriptor_dim // self.num_attention_heads
        elif self.descriptor_dim // self.num_attention_heads != self.attention_head_dim:
            raise ValueError("descriptor_dim / num_attention_heads must equal attention_head_dim")

        if self.positional_encoding_type not in {"learnable", "fixed"}:
            raise ValueError("positional_encoding_type must be either 'learnable' or 'fixed'")

        super().__post_init__(**kwargs)


class LoMaConvRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        hidden_blocks: int,
    ) -> None:
        super().__init__()
        self.block1 = self._make_block(in_channels, hidden_channels, depthwise=False, kernel_size=1)
        self.hidden_blocks = nn.Sequential(
            *(
                self._make_block(hidden_channels, hidden_channels, depthwise=True, kernel_size=5)
                for _ in range(hidden_blocks)
            )
        )
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    @staticmethod
    def _make_block(in_channels: int, out_channels: int, depthwise: bool, kernel_size: int) -> nn.Sequential:
        if depthwise and out_channels % in_channels != 0:
            raise ValueError("The output channels of a depthwise block must be divisible by its input channels")

        groups = in_channels if depthwise else 1
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=groups,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        initial_features = self.block1(feature_map)
        refined_features = self.hidden_blocks(initial_features)
        return self.out_conv((refined_features + initial_features) / 1.4)


class LoMaVgg19Encoder(nn.Module):
    """VGG-19 with batch normalization that returns feature maps before each pooling operation."""

    def __init__(self) -> None:
        super().__init__()
        channels = [64, 64, "pool", 128, 128, "pool", 256, 256, 256, 256, "pool", 512, 512, 512, 512, "pool"]
        layers = []
        in_channels = 3
        for out_channels in channels:
            if out_channels == "pool":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend(
                    [
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = out_channels
        self.layers = nn.ModuleList(layers)

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        feature_maps = []
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                feature_maps.append(pixel_values)
            pixel_values = layer(pixel_values)
        return feature_maps


class LoMaDescriptorDecoder(nn.Module):
    def __init__(self, descriptor_dim: int, hidden_blocks: int) -> None:
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.scales = ("8", "4", "2", "1")
        self.layers = nn.ModuleDict(
            {
                "8": LoMaConvRefiner(512, 512, 256 + descriptor_dim, hidden_blocks),
                "4": LoMaConvRefiner(512, 256, 128 + descriptor_dim, hidden_blocks),
                "2": LoMaConvRefiner(256, 64, 32 + descriptor_dim, hidden_blocks),
                "1": LoMaConvRefiner(96, 32, 1 + descriptor_dim, hidden_blocks),
            }
        )

    def forward(
        self, feature_map: torch.Tensor, scale: str, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context is not None:
            feature_map = torch.cat((feature_map, context), dim=1)
        output = self.layers[scale](feature_map)
        return output[:, : self.descriptor_dim], output[:, self.descriptor_dim :]


class LoMaDescriptorNetwork(nn.Module):
    """LoMa's DeDoDe-B-style descriptor network.

    The network produces a dense descriptor grid from an RGB image. `describe_keypoints` bilinearly samples that grid
    at normalized keypoint coordinates in the `[-1, 1]` range used by `torch.nn.functional.grid_sample`.
    """

    def __init__(self, config: LoMaConfig) -> None:
        super().__init__()
        self.encoder = LoMaVgg19Encoder()
        self.decoder = LoMaDescriptorDecoder(config.input_descriptor_dim, config.descriptor_hidden_blocks)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feature_maps = self.encoder(pixel_values)
        descriptor_grid = pixel_values.new_zeros(
            pixel_values.shape[0], self.decoder.descriptor_dim, *feature_maps[-1].shape[-2:]
        )
        context = None

        for index, (feature_map, scale) in enumerate(zip(reversed(feature_maps), self.decoder.scales)):
            descriptor_update, context = self.decoder(feature_map, scale=scale, context=context)
            descriptor_grid = descriptor_grid + descriptor_update
            if index < len(self.decoder.scales) - 1:
                descriptor_grid = F.interpolate(
                    descriptor_grid, size=feature_maps[-(index + 2)].shape[-2:], mode="bilinear", align_corners=False
                )
                context = F.interpolate(
                    context, size=feature_maps[-(index + 2)].shape[-2:], mode="bilinear", align_corners=False
                )

        return descriptor_grid

    def describe_keypoints(self, pixel_values: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        descriptor_grid = self(pixel_values)
        return F.grid_sample(descriptor_grid.float(), keypoints[:, None], mode="bilinear", align_corners=False)[
            :, :, 0
        ].mT


class LoMaKeypointMatchingOutput(LightGlueKeypointMatchingOutput):
    pass


class LoMaPositionalEncoder(LightGluePositionalEncoder):
    pass


class LoMaAttention(LightGlueAttention):
    pass


class LoMaMLP(LightGlueMLP):
    pass


class LoMaTransformerLayer(LightGlueTransformerLayer):
    pass


class LoMaMatchAssignmentLayer(LightGlueMatchAssignmentLayer):
    pass


class LoMaTokenConfidenceLayer(LightGlueTokenConfidenceLayer):
    pass


class LoMaPreTrainedModel(LightGluePreTrainedModel):
    pass


class LoMaForKeypointMatching(LightGlueForKeypointMatching):
    pass


class LoMaImageProcessorKwargs(LightGlueImageProcessorKwargs):
    pass


class LoMaImageProcessor(LightGlueImageProcessor):
    pass


class LoMaImageProcessorPil(LightGlueImageProcessorPil):
    pass


__all__ = [
    "LoMaConfig",
    "LoMaPreTrainedModel",
    "LoMaForKeypointMatching",
    "LoMaImageProcessor",
    "LoMaImageProcessorPil",
]
