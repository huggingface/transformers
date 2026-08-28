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

from dataclasses import field

import torch
from torch import nn
from torch.nn import functional as F

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, can_return_tuple
from ..auto import CONFIG_MAPPING, AutoBackbone, AutoConfig
from ..auto.modeling_auto import AutoModelForKeypointDetection
from ..lightglue.configuration_lightglue import LightGlueConfig
from ..lightglue.image_processing_lightglue import LightGlueImageProcessor, LightGlueImageProcessorKwargs
from ..lightglue.modeling_lightglue import (
    LightGlueAttention,
    LightGlueKeypointMatchingOutput,
    LightGlueMatchAssignmentLayer,
    LightGlueMLP,
    LightGluePositionalEncoder,
    LightGluePreTrainedModel,
    LightGlueTransformerLayer,
)


class LoMaVgg19EncoderConfig(PreTrainedConfig):
    r"""
    in_channels (`int`, *optional*, defaults to 3):
        Number of input image channels.
    hidden_sizes (`list[int]`, *optional*, defaults to `[64, 128, 256, 512]`):
        Number of channels in the four VGG encoder stages.
    num_hidden_layers (`list[int]`, *optional*, defaults to `[2, 2, 4, 4]`):
        Number of convolution blocks in each VGG encoder stage.
    conv_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size used by the convolution blocks.
    pool_kernel_size (`int`, *optional*, defaults to 2):
        Kernel size used by the pooling layers between encoder stages.
    pool_stride (`int`, *optional*, defaults to 2):
        Stride used by the pooling layers between encoder stages.
    """

    in_channels: int = 3
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    num_hidden_layers: list[int] = field(default_factory=lambda: [2, 2, 4, 4])
    conv_kernel_size: int = 3
    pool_kernel_size: int = 2
    pool_stride: int = 2

    def __post_init__(self, **kwargs):
        if len(self.hidden_sizes) != len(self.num_hidden_layers):
            raise ValueError("hidden_sizes and num_hidden_layers must have the same length")
        if any(num_hidden_layers < 1 for num_hidden_layers in self.num_hidden_layers):
            raise ValueError("num_hidden_layers must contain only positive values")
        super().__post_init__(**kwargs)


class LoMaDescriptorDecoderConfig(PreTrainedConfig):
    r"""
    scales (`list[str]`, *optional*, defaults to `["14", "8", "4", "2", "1"]`):
        Names of the decoder stages, ordered from the auxiliary backbone resolution to the image resolution.
    hidden_sizes (`list[int]`, *optional*, defaults to `[768, 512, 256, 64, 32]`):
        Number of channels in the intermediate convolution of each decoder stage.
    context_channels (`list[int]`, *optional*, defaults to `[512, 256, 128, 32, 1]`):
        Number of context channels produced by each decoder stage.
    """

    scales: list[str] = field(default_factory=lambda: ["14", "8", "4", "2", "1"])
    hidden_sizes: list[int] = field(default_factory=lambda: [768, 512, 256, 64, 32])
    context_channels: list[int] = field(default_factory=lambda: [512, 256, 128, 32, 1])

    def __post_init__(self, **kwargs):
        if len(self.scales) != len(self.hidden_sizes) or len(self.scales) != len(self.context_channels):
            raise ValueError("scales, hidden_sizes, and context_channels must have the same length")
        super().__post_init__(**kwargs)


class LoMaConfig(LightGlueConfig):
    r"""
    keypoint_detector_config (`Union[AutoConfig, dict]`, *optional*, defaults to `SuperPointConfig`):
        Configuration of the keypoint detector. The initial LoMa integration supports SuperPoint.
    input_descriptor_dim (`int`, *optional*, defaults to 256):
        Dimension of the local descriptors supplied by the descriptor network.
    descriptor_dim (`int`, *optional*, defaults to 256):
        Dimension of the descriptors used by the matching transformer.
    head_dim (`int`, *optional*, defaults to 64):
        Dimension of each attention head. The number of attention heads is derived from `descriptor_dim` when it is
        not specified.
    num_hidden_layers (`int`, *optional*, defaults to 9):
        Number of self- and cross-attention layers in the matching transformer.
    filter_threshold (`float`, *optional*, defaults to 0.1):
        Confidence threshold used to retain mutual matches.
    positional_encoding_type (`str`, *optional*, defaults to `"learnable"`):
        Type of Fourier positional encoding applied in self-attention. Supported values are `"learnable"` and
        `"fixed"`.
    positional_encoding_gamma (`float`, *optional*, defaults to 1.0):
        Frequency scale used by the Fourier positional encoding.
    descriptor_hidden_blocks (`int`, *optional*, defaults to 5):
        Number of depthwise refinement blocks at each decoder scale in the local descriptor network.
    encoder_config (`LoMaVgg19EncoderConfig`, *optional*):
        Configuration for LoMa's VGG-19-style local encoder.
    decoder_config (`LoMaDescriptorDecoderConfig`, *optional*):
        Configuration for LoMa's multi-scale local descriptor decoder.
    backbone_config (`Union[AutoConfig, dict]`, *optional*):
        Configuration for the auxiliary backbone encoder (DINOv2 by default) used in the descriptor network.

    Examples:
        ```python
        >>> from transformers import LoMaConfig

        >>> config = LoMaConfig()
        >>> config.num_attention_heads
        4
        ```
    """

    model_type = "loma"
    sub_configs = {
        "keypoint_detector_config": AutoConfig,
        "encoder_config": LoMaVgg19EncoderConfig,
        "decoder_config": LoMaDescriptorDecoderConfig,
        "backbone_config": AutoConfig,
    }

    input_descriptor_dim: int = 256
    descriptor_dim: int = 256
    head_dim: int | None = None
    num_hidden_layers: int = 9
    num_attention_heads: int | None = None
    filter_threshold: float = 0.1
    positional_encoding_type: str = "learnable"
    positional_encoding_gamma: float = 1.0
    descriptor_hidden_blocks: int = 5
    hidden_act: str = "gelu"
    encoder_config: dict | LoMaVgg19EncoderConfig | None = None
    decoder_config: dict | LoMaDescriptorDecoderConfig | None = None
    backbone_config: dict | PreTrainedConfig | None = None

    # Fields inherited from LightGlueConfig but not used by LoMa's matching architecture.
    depth_confidence = AttributeError()
    width_confidence = AttributeError()

    def __post_init__(self, **kwargs):
        if self.num_attention_heads is None:
            if self.head_dim is None:
                self.head_dim = 64
            if self.descriptor_dim % self.head_dim != 0:
                raise ValueError("descriptor_dim must be divisible by head_dim")
            self.num_attention_heads = self.descriptor_dim // self.head_dim
        elif self.head_dim is None:
            self.head_dim = self.descriptor_dim // self.num_attention_heads
        elif self.descriptor_dim // self.num_attention_heads != self.head_dim:
            raise ValueError("descriptor_dim / num_attention_heads must equal head_dim")

        if self.positional_encoding_type not in {"learnable", "fixed"}:
            raise ValueError("positional_encoding_type must be either 'learnable' or 'fixed'")

        if isinstance(self.keypoint_detector_config, dict):
            self.keypoint_detector_config["model_type"] = self.keypoint_detector_config.get("model_type", "superpoint")
            self.keypoint_detector_config = CONFIG_MAPPING[self.keypoint_detector_config["model_type"]](
                **self.keypoint_detector_config, attn_implementation="eager"
            )
        elif self.keypoint_detector_config is None:
            self.keypoint_detector_config = CONFIG_MAPPING["superpoint"](attn_implementation="eager")

        if isinstance(self.encoder_config, dict):
            self.encoder_config = LoMaVgg19EncoderConfig(**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = LoMaVgg19EncoderConfig()

        if isinstance(self.decoder_config, dict):
            self.decoder_config = LoMaDescriptorDecoderConfig(**self.decoder_config)
        elif self.decoder_config is None:
            self.decoder_config = LoMaDescriptorDecoderConfig()

        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="dinov2",
            default_config_kwargs={
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "patch_size": 14,
                "image_size": 518,
                "out_features": ["stage24"],
                "reshape_hidden_states": False,
            },
            **kwargs,
        )

        self.num_key_value_heads = self.num_attention_heads
        self.intermediate_size = self.descriptor_dim * 2
        self.hidden_size = self.descriptor_dim
        PreTrainedConfig.__post_init__(self, **kwargs)


class LoMaConvBlock(nn.Module):
    """A single convolution block: Conv2d → BatchNorm2d → ReLU → Conv2d (1x1)."""

    def __init__(self, in_channels: int, out_channels: int, depthwise: bool, kernel_size: int) -> None:
        super().__init__()
        if depthwise and out_channels % in_channels != 0:
            raise ValueError("The output channels of a depthwise block must be divisible by its input channels")
        groups = in_channels if depthwise else 1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=groups
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise(hidden_states)
        return hidden_states


class LoMaConvRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        hidden_blocks: int,
    ) -> None:
        super().__init__()
        self.block1 = LoMaConvBlock(in_channels, hidden_channels, depthwise=False, kernel_size=1)
        self.hidden_blocks = nn.ModuleList(
            [
                LoMaConvBlock(hidden_channels, hidden_channels, depthwise=True, kernel_size=5)
                for _ in range(hidden_blocks)
            ]
        )
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        initial_features = self.block1(feature_map)
        refined_features = initial_features
        for block in self.hidden_blocks:
            refined_features = block(refined_features)
        return self.out_conv((refined_features + initial_features) / 1.4)


class LoMaVgg19Encoder(nn.Module):
    """VGG-19 with batch normalization that returns feature maps before each pooling operation."""

    def __init__(self, config: LoMaVgg19EncoderConfig) -> None:
        super().__init__()
        layers = []
        in_channels = config.in_channels
        for out_channels, num_hidden_layers in zip(config.hidden_sizes, config.num_hidden_layers):
            for _ in range(num_hidden_layers):
                layers.extend(
                    [
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=config.conv_kernel_size,
                            padding=config.conv_kernel_size // 2,
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=config.pool_kernel_size, stride=config.pool_stride))
        self.layers = nn.ModuleList(layers)

    def forward(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        feature_maps = []
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                feature_maps.append(pixel_values)
            pixel_values = layer(pixel_values)
        return feature_maps


class LoMaDescriptorDecoder(nn.Module):
    def __init__(self, config: "LoMaConfig") -> None:
        super().__init__()
        descriptor_dim = config.input_descriptor_dim
        hidden_blocks = config.descriptor_hidden_blocks
        backbone_hidden_size = config.backbone_config.hidden_size
        self.descriptor_dim = descriptor_dim
        self.scales = config.decoder_config.scales
        feature_channels = [backbone_hidden_size, *reversed(config.encoder_config.hidden_sizes)]
        if len(feature_channels) != len(self.scales):
            raise ValueError("decoder_config.scales must have one more entry than encoder_config.hidden_sizes")

        layers = {}
        previous_context_channels = 0
        for scale, num_feature_channels, hidden_size, context_channels in zip(
            self.scales,
            feature_channels,
            config.decoder_config.hidden_sizes,
            config.decoder_config.context_channels,
        ):
            layers[scale] = LoMaConvRefiner(
                num_feature_channels + previous_context_channels,
                hidden_size,
                context_channels + descriptor_dim,
                hidden_blocks,
            )
            previous_context_channels = context_channels
        self.layers = nn.ModuleDict(layers)

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

    def __init__(self, config: "LoMaConfig") -> None:
        super().__init__()
        self.encoder = LoMaVgg19Encoder(config.encoder_config)
        self.auxiliary_backbone = AutoBackbone.from_config(config.backbone_config)
        self.auxiliary_backbone.eval()
        self.decoder = LoMaDescriptorDecoder(config)

    @torch.inference_mode()
    def _extract_auxiliary_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.auxiliary_backbone.eval()
        backbone_output = self.auxiliary_backbone(pixel_values)
        backbone_features = backbone_output.feature_maps[-1]
        # Reshape from (batch, seq, hidden) to (batch, hidden, h, w) spatial format.
        if backbone_features.ndim == 3:
            batch_size, seq_len, hidden_size = backbone_features.shape
            h = pixel_values.shape[2] // self.auxiliary_backbone.config.patch_size
            w = pixel_values.shape[3] // self.auxiliary_backbone.config.patch_size
            # Strip CLS token if present (seq_len = h*w + 1 with CLS, h*w without).
            if seq_len == h * w + 1:
                backbone_features = backbone_features[:, 1:]
            backbone_features = backbone_features.transpose(1, 2).reshape(batch_size, hidden_size, h, w)
        return backbone_features

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.shape[1] == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)
        elif pixel_values.shape[1] != 3:
            raise ValueError("LoMaDescriptorNetwork expects one grayscale or three RGB channels")

        feature_maps = self.encoder(pixel_values)

        # Frozen auxiliary backbone (DINOv2) provides high-level features at stride 14.
        feature_maps.append(self._extract_auxiliary_features(pixel_values).clone())

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
    def __init__(self, config: "LoMaConfig") -> None:
        super().__init__(config)
        self.positional_encoding_type = config.positional_encoding_type
        self.gamma = config.positional_encoding_gamma


class LoMaAttention(LightGlueAttention):
    def __init__(self, config: "LoMaConfig", layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.is_causal = False


class LoMaMLP(LightGlueMLP):
    pass


class LoMaTransformerLayer(LightGlueTransformerLayer):
    pass


class LoMaMatchAssignmentLayer(LightGlueMatchAssignmentLayer):
    def forward(
        self,
        descriptors: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_keypoints, descriptor_dim = descriptors.shape
        m_descriptors = self.final_projection(descriptors) / self.descriptor_dim**0.25
        m_descriptors = m_descriptors.reshape(batch_size // 2, 2, num_keypoints, descriptor_dim)
        descriptors_0, descriptors_1 = m_descriptors[:, 0], m_descriptors[:, 1]

        similarity = descriptors_0 @ descriptors_1.transpose(-1, -2)
        if mask is not None:
            mask = mask.reshape(batch_size // 2, 2, num_keypoints)
            mask_0, mask_1 = mask[:, 0], mask[:, 1]
            similarity = similarity.masked_fill(
                ~(mask_0.unsqueeze(-1).bool() & mask_1.unsqueeze(-2).bool()), torch.finfo(similarity.dtype).min
            )

        if self.training:
            matchability = self.matchability(descriptors)
            matchability = matchability.reshape(batch_size // 2, 2, num_keypoints, 1)
            matchability_0, matchability_1 = matchability[:, 0], matchability[:, 1]
            scores_0 = F.log_softmax(similarity, dim=2)
            scores_1 = F.log_softmax(similarity.transpose(-1, -2), dim=2).transpose(-1, -2)
            half_batch = batch_size // 2
            scores = similarity.new_zeros((half_batch, num_keypoints + 1, num_keypoints + 1))
            scores[:, :-1, :-1] = scores_0 + scores_1
            scores[:, :-1, -1] = matchability_0.squeeze(-1)
            scores[:, -1, :-1] = matchability_1.squeeze(-1)
            return scores
        return F.softmax(similarity, dim=2) * F.softmax(similarity, dim=1)


class LoMaPreTrainedModel(LightGluePreTrainedModel):
    config: LoMaConfig
    base_model_prefix = "model"


@auto_docstring(
    custom_intro="""
    LoMa model taking image pairs as inputs and returning keypoint correspondences.
    """
)
class LoMaForKeypointMatching(LoMaPreTrainedModel):
    def __init__(self, config: LoMaConfig):
        super().__init__(config)
        self.keypoint_detector = AutoModelForKeypointDetection.from_config(config.keypoint_detector_config)
        self.descriptor_network = LoMaDescriptorNetwork(config)
        # CODEPATH: input_descriptor_dim != descriptor_dim → all released LoMa checkpoints use 256 for both
        self.input_projection = (
            nn.Identity()
            if config.input_descriptor_dim == config.descriptor_dim
            else nn.Linear(config.input_descriptor_dim, config.descriptor_dim, bias=config.attention_bias)
        )
        self.positional_encoder = LoMaPositionalEncoder(config)
        self.layers = nn.ModuleList(
            [LoMaTransformerLayer(config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.match_assignment = LoMaMatchAssignmentLayer(config)
        self.filter_threshold = config.filter_threshold
        self.num_hidden_layers = config.num_hidden_layers
        self.post_init()

    def _match_image_pair(
        self,
        keypoints: torch.Tensor,
        descriptors: torch.Tensor,
        mask: torch.Tensor,
        output_hidden_states: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...] | None]:
        batch_size, _, num_keypoints, descriptor_dim = descriptors.shape

        # Interleave image pairs into (batch*2, num_keypoints, dim) for LightGlue-compatible layers
        descriptors = self.input_projection(descriptors.reshape(batch_size * 2, num_keypoints, descriptor_dim))
        keypoints_flat = keypoints.reshape(batch_size * 2, num_keypoints, 2)
        position_embeddings = self.positional_encoder(keypoints_flat)
        position_embeddings = position_embeddings[0]  # (cos, sin) tuple

        mask_flat = mask.reshape(batch_size * 2, num_keypoints)
        # Build 4D attention mask: (batch*2, 1, 1, num_keypoints)
        attention_mask = mask_flat[:, None, None, :].to(dtype=descriptors.dtype)
        attention_mask = (1.0 - attention_mask) * torch.finfo(descriptors.dtype).min

        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            descriptors, hidden_states, _ = layer(
                descriptors, position_embeddings, attention_mask, output_hidden_states=output_hidden_states
            )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + hidden_states

        scores = self.match_assignment(descriptors, mask_flat)

        # Extract mutual matches from the score matrix
        maximum_scores_0, maximum_scores_1 = scores.max(dim=2), scores.max(dim=1)
        matches_0, matches_1 = maximum_scores_0.indices, maximum_scores_1.indices
        indices = torch.arange(num_keypoints, device=scores.device)[None]
        mutual_0 = indices == matches_1.gather(1, matches_0)
        mutual_1 = indices == matches_0.gather(1, matches_1)

        mask_paired = mask.reshape(batch_size, 2, num_keypoints)
        valid_0 = mutual_0 & (maximum_scores_0.values > self.filter_threshold) & mask_paired[:, 0].bool()
        valid_1 = mutual_1 & valid_0.gather(1, matches_1) & mask_paired[:, 1].bool()
        matching_scores_0 = torch.where(valid_0, maximum_scores_0.values, torch.zeros_like(maximum_scores_0.values))
        matching_scores_1 = torch.where(valid_1, maximum_scores_1.values, torch.zeros_like(maximum_scores_1.values))
        matches_0 = torch.where(valid_0, matches_0, torch.full_like(matches_0, -1))
        matches_1 = torch.where(valid_1, matches_1, torch.full_like(matches_1, -1))
        matches = torch.stack((matches_0, matches_1), dim=1)
        matching_scores = torch.stack((matching_scores_0, matching_scores_1), dim=1)
        prune = torch.full_like(matches, self.num_hidden_layers)
        return matches, matching_scores, prune, all_hidden_states

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ) -> tuple | LoMaKeypointMatchingOutput:
        if pixel_values.ndim != 5 or pixel_values.size(1) != 2:
            raise ValueError("pixel_values must have shape (batch_size, 2, num_channels, height, width)")

        output_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states)
        batch_size, _, num_channels, height, width = pixel_values.shape
        pixel_values_flat = pixel_values.reshape(batch_size * 2, num_channels, height, width)

        keypoint_detections = self.keypoint_detector(pixel_values_flat)
        keypoints, _, _, mask = keypoint_detections[:4]
        keypoints = keypoints.reshape(batch_size, 2, -1, 2).to(pixel_values)
        mask = mask.reshape(batch_size, 2, -1)

        descriptor_keypoints = keypoints.reshape(batch_size * 2, -1, 2) * 2 - 1
        descriptors = self.descriptor_network.describe_keypoints(pixel_values_flat, descriptor_keypoints)
        descriptors = descriptors.reshape(batch_size, 2, -1, self.config.input_descriptor_dim).to(pixel_values)

        image_size = keypoints.new_tensor((width, height))
        absolute_keypoints = keypoints * image_size
        normalized_keypoints = (absolute_keypoints - image_size / 2) / (image_size.max() / 2)

        matches, matching_scores, prune, hidden_states = self._match_image_pair(
            normalized_keypoints, descriptors, mask, output_hidden_states
        )

        return LoMaKeypointMatchingOutput(
            matches=matches,
            matching_scores=matching_scores,
            keypoints=keypoints,
            prune=prune,
            mask=mask.to(torch.int),
            hidden_states=hidden_states,
            attentions=None,
        )


class LoMaImageProcessorKwargs(LightGlueImageProcessorKwargs):
    pass


class LoMaImageProcessor(LightGlueImageProcessor):
    pass


__all__ = [
    "LoMaConfig",
    "LoMaPreTrainedModel",
    "LoMaForKeypointMatching",
    "LoMaImageProcessor",
]
