# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""PyTorch RoMa model."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...backbone_utils import load_backbone
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from .configuration_roma import RomaConfig


# The five coarse-to-fine scales of the RoMa decoder, from coarse (1/16) to fine (1/1).
ROMA_SCALES = ["16", "8", "4", "2", "1"]
# The high-resolution upsample refinement pass skips the DINOv2 (1/16) scale and only runs the fine scales.
ROMA_UPSAMPLE_SCALES = ["8", "4", "2", "1"]
# Only the coarsest scale runs the Gaussian-Process + transformer global matcher.
ROMA_COARSE_SCALES = [16]


@auto_docstring(
    custom_intro="""
    Base class for outputs of RoMa keypoint matching models. RoMa is a dense matcher: it predicts a dense `warp`
    field and a `certainty` map, from which a variable number of sparse correspondences are sampled. To allow
    batching, a fixed `num_samples` correspondences are returned in `matches` / `matching_scores`.
    """
)
@dataclass
class RomaKeypointMatchingOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
        Loss computed during training.
    matches (`torch.FloatTensor` of shape `(batch_size, num_samples, 4)`):
        Sampled sparse correspondences, as normalized `(x_0, y_0, x_1, y_1)` coordinates in `[-1, 1]`, where
        `(x_0, y_0)` is in the first image and `(x_1, y_1)` in the second.
    matching_scores (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
        Certainty (in `[0, 1]`) of each sampled correspondence.
    warp (`torch.FloatTensor` of shape `(batch_size, height, width, 4)`):
        The dense warp field. In symmetric mode the width is doubled: the left half holds the `image0 -> image1`
        warp and the right half the `image1 -> image0` warp.
    certainty (`torch.FloatTensor` of shape `(batch_size, height, width)`):
        The dense certainty (in `[0, 1]`) associated with `warp`.
    """

    loss: torch.FloatTensor | None = None
    matches: torch.FloatTensor | None = None
    matching_scores: torch.FloatTensor | None = None
    warp: torch.FloatTensor | None = None
    certainty: torch.FloatTensor | None = None


def normalized_coordinate_grid(batch_size: int, height: int, width: int, device, dtype=torch.float32) -> torch.Tensor:
    """Return a `(batch_size, 2, height, width)` grid of normalized `(x, y)` coordinates spanning `[-1, 1]`."""
    coords = torch.meshgrid(
        torch.linspace(-1 + 1 / height, 1 - 1 / height, height, device=device, dtype=dtype),
        torch.linspace(-1 + 1 / width, 1 - 1 / width, width, device=device, dtype=dtype),
        indexing="ij",
    )
    coords = torch.stack((coords[1], coords[0]))
    return coords[None].expand(batch_size, 2, height, width)


def classification_to_flow(class_logits: torch.Tensor) -> torch.Tensor:
    """
    Convert the coarse classification logits into a soft (refined) flow by taking the local probability-weighted
    average of the anchor coordinates around the arg-max anchor. Mirrors RoMa's `cls_to_flow_refine`.
    """
    batch_size, num_classes, height, width = class_logits.shape
    device = class_logits.device
    resolution = round(math.sqrt(num_classes))
    grid = torch.meshgrid(
        torch.linspace(-1 + 1 / resolution, 1 - 1 / resolution, resolution, device=device),
        torch.linspace(-1 + 1 / resolution, 1 - 1 / resolution, resolution, device=device),
        indexing="ij",
    )
    grid = torch.stack([grid[1], grid[0]], dim=-1).reshape(num_classes, 2)
    probabilities = class_logits.softmax(dim=1)
    mode = probabilities.max(dim=1).indices
    index = (
        torch.stack((mode - 1, mode, mode + 1, mode - resolution, mode + resolution), dim=1)
        .clamp(0, num_classes - 1)
        .long()
    )
    neighbours = torch.gather(probabilities, dim=1, index=index)[..., None]
    flow = sum(neighbours[:, i] * grid[index[:, i]] for i in range(5))
    flow = flow / neighbours.sum(dim=1)
    return flow


def local_correlation(
    feature0: torch.Tensor,
    feature1: torch.Tensor,
    local_radius: int,
    warp: torch.Tensor,
    sample_mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """
    Native (non-CUDA) local correlation: for every location, correlate `feature0` with a `(2r+1)x(2r+1)` window of
    `feature1` sampled around the predicted `warp` coordinate. Returns `(batch, (2r+1)**2, height, width)`.
    """
    num_offsets = (2 * local_radius + 1) ** 2
    batch_size, channels, height, width = feature0.size()
    warp = warp.permute(0, 2, 3, 1)
    device = feature0.device
    dtype = feature0.dtype
    local_window = torch.meshgrid(
        torch.linspace(-2 * local_radius / height, 2 * local_radius / height, 2 * local_radius + 1, device=device),
        torch.linspace(-2 * local_radius / width, 2 * local_radius / width, 2 * local_radius + 1, device=device),
        indexing="ij",
    )
    local_window = (
        torch.stack((local_window[1], local_window[0]), dim=-1)[None]
        .expand(1, 2 * local_radius + 1, 2 * local_radius + 1, 2)
        .reshape(1, num_offsets, 2)
    )
    correlation = torch.empty((batch_size, num_offsets, height, width), device=device, dtype=dtype)
    for b in range(batch_size):
        local_window_coords = (warp[b, :, :, None] + local_window[:, None, None]).reshape(
            1, height, width * num_offsets, 2
        )
        window_feature = F.grid_sample(
            feature1[b : b + 1].float(),
            local_window_coords.float(),
            padding_mode=padding_mode,
            align_corners=False,
            mode=sample_mode,
        )
        window_feature = window_feature.reshape(channels, height, width, num_offsets)
        correlation[b] = (feature0[b, ..., None] / (channels**0.5) * window_feature).sum(dim=0).permute(2, 0, 1)
    return correlation


def kernel_density_estimate(matches: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """Gaussian kernel density estimate of the sampled matches, used to balance the sparse-match distribution."""
    scores = (-(torch.cdist(matches, matches) ** 2) / (2 * std**2)).exp()
    return scores.sum(dim=-1)


class RomaVGG19(nn.Module):
    """
    The fine-feature pyramid of RoMa: a VGG-19 (with batch-norm) feature extractor whose feature maps are read off
    *before* each max-pooling layer, producing features at strides 1, 2, 4 and 8.
    """

    def __init__(self, config: RomaConfig):
        super().__init__()
        # VGG-19 configuration "E": number of convs per stage, "M" marks a max-pool. The per-stage channel counts
        # are `config.cnn_feature_dims` (defaults to [64, 128, 256, 512], the ImageNet VGG-19 widths).
        c0, c1, c2, c3 = config.cnn_feature_dims
        vgg_config = [c0, c0, "M", c1, c1, "M", c2, c2, c2, c2, "M", c3, c3, c3, c3, "M"]
        layers = []
        in_channels = 3
        for value in vgg_config:
            if value == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, value, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(value))
                layers.append(nn.ReLU(inplace=True))
                in_channels = value
        self.layers = nn.ModuleList(layers)

    def forward(self, pixel_values: torch.Tensor) -> dict[int, torch.Tensor]:
        feature_pyramid = {}
        scale = 1
        hidden_state = pixel_values
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                feature_pyramid[scale] = hidden_state
                scale = scale * 2
            hidden_state = layer(hidden_state)
        return feature_pyramid


class RomaCosineKernel(nn.Module):
    """Cosine-similarity kernel with temperature `T`, as used by the RoMa Gaussian Process."""

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        cosine = torch.einsum("bnd,bmd->bnm", x, y) / (x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps)
        return ((cosine - 1.0) / self.temperature).exp()


class RomaGaussianProcess(nn.Module):
    """
    Gaussian-Process coordinate regressor. Embeds the support-image positions with a Fourier basis and predicts, for
    every query location, a smoothed positional embedding via a GP posterior with a cosine kernel.
    """

    def __init__(self, config: RomaConfig):
        super().__init__()
        self.kernel = RomaCosineKernel(config.kernel_temperature)
        self.sigma_noise = 0.1
        self.pos_conv = nn.Conv2d(2, config.gp_dim, kernel_size=1, stride=1)

    def project_to_basis(self, coords: torch.Tensor) -> torch.Tensor:
        return torch.cos(8 * math.pi * self.pos_conv(coords))

    def get_positional_encoding(self, support_features: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = support_features.shape
        coords = normalized_coordinate_grid(batch_size, height, width, support_features.device)
        return self.project_to_basis(coords)

    def forward(self, query_features: torch.Tensor, support_features: torch.Tensor) -> torch.Tensor:
        batch_size, _, query_height, query_width = query_features.shape
        _, _, support_height, support_width = support_features.shape
        positional_encoding = self.get_positional_encoding(support_features)
        _, embedding_dim, _, _ = positional_encoding.shape

        query = query_features.float().flatten(2).transpose(1, 2)
        support = support_features.float().flatten(2).transpose(1, 2)
        positional_encoding = positional_encoding.flatten(2).transpose(1, 2)

        kernel_support_support = self.kernel(support, support)
        kernel_query_support = self.kernel(query, support)
        num_support = support_height * support_width
        sigma_noise = self.sigma_noise * torch.eye(num_support, device=query.device)[None]

        cholesky = torch.linalg.cholesky(kernel_support_support + sigma_noise)
        embedded = torch.cholesky_solve(positional_encoding, cholesky, upper=False)
        posterior = kernel_query_support @ embedded
        posterior = posterior.transpose(1, 2).reshape(batch_size, embedding_dim, query_height, query_width)
        return posterior


class RomaTransformerAttention(nn.Module):
    """Standard multi-head self-attention (no bias on the qkv projection), matching RoMa's DINOv2-style block."""

    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(
            batch_size, seq_len, 3, self.num_attention_heads, hidden_size // self.num_attention_heads
        )
        query, key, value = torch.unbind(qkv, 2)
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        attention_output = F.scaled_dot_product_attention(query, key, value)
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        return self.proj(attention_output)


class RomaTransformerMlp(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * mlp_ratio)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size * mlp_ratio, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


class RomaTransformerBlock(nn.Module):
    """Pre-norm transformer block (no layer-scale, no drop-path) used by the coarse coordinate decoder."""

    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = RomaTransformerAttention(hidden_size, num_attention_heads)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = RomaTransformerMlp(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class RomaCoordinateDecoder(nn.Module):
    """
    The transformer match decoder. It consumes the concatenation of the GP posterior and the projected coarse query
    features and predicts, per location, anchor classification logits plus a certainty logit.
    """

    def __init__(self, config: RomaConfig):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                RomaTransformerBlock(config.decoder_hidden_size, config.decoder_num_attention_heads)
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.to_out = nn.Linear(config.decoder_hidden_size, config.num_anchors)

    def forward(self, gp_posterior: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = gp_posterior.shape
        tokens = torch.cat((gp_posterior, features), dim=1)
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = self.blocks(tokens)
        out = self.to_out(tokens)
        out = out.transpose(1, 2).reshape(batch_size, -1, height, width)
        class_logits, certainty = out[:, :-1], out[:, -1:]
        return class_logits, certainty


class RomaConvRefiner(nn.Module):
    """
    Coarse-to-fine convolutional refiner. Given the query/support features and the current flow, it warps the
    support features to the query, builds a local correlation volume and a displacement embedding, and predicts a
    flow delta plus a certainty delta.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        kernel_size: int,
        hidden_blocks: int,
        displacement_emb_dim: int,
        local_corr_radius: int,
        batch_norm_eps: float,
    ):
        super().__init__()
        self.local_corr_radius = local_corr_radius
        self.block1 = self.create_block(in_dim, hidden_dim, kernel_size, batch_norm_eps, bias=True)
        self.hidden_blocks = nn.Sequential(
            *[self.create_block(hidden_dim, hidden_dim, kernel_size, batch_norm_eps) for _ in range(hidden_blocks)]
        )
        self.out_conv = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.disp_emb = nn.Conv2d(2, displacement_emb_dim, kernel_size=1, stride=1, padding=0)

    def create_block(
        self, in_dim: int, out_dim: int, kernel_size: int, batch_norm_eps: float, bias: bool = True
    ) -> nn.Sequential:
        # Depthwise separable: a grouped conv (groups=in_dim) followed by a 1x1 conv.
        conv1 = nn.Conv2d(
            in_dim, out_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=in_dim, bias=bias
        )
        norm = nn.BatchNorm2d(out_dim, eps=batch_norm_eps, momentum=0.01)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0)
        return nn.Sequential(conv1, norm, relu, conv2)

    def forward(
        self, query_features: torch.Tensor, support_features: torch.Tensor, flow: torch.Tensor, scale_factor: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = query_features.shape
        warped_support = F.grid_sample(
            support_features, flow.permute(0, 2, 3, 1), align_corners=False, mode="bilinear"
        )

        query_coords = normalized_coordinate_grid(batch_size, height, width, query_features.device)
        displacement = flow - query_coords
        displacement_embedding = self.disp_emb(40 / 32 * scale_factor * displacement)

        if self.local_corr_radius:
            correlation = local_correlation(query_features, support_features, self.local_corr_radius, flow)
            features = torch.cat((query_features, warped_support, displacement_embedding, correlation), dim=1)
        else:
            features = torch.cat((query_features, warped_support, displacement_embedding), dim=1)

        # The padded RoMa decoder over-allocates the input channels; zero-pad the feature volume to match.
        expected_channels = self.block1[0].in_channels
        if features.shape[1] != expected_channels:
            features = F.pad(features, (0, 0, 0, 0, 0, expected_channels - features.shape[1]))

        hidden_states = self.block1(features)
        hidden_states = self.hidden_blocks(hidden_states)
        out = self.out_conv(hidden_states)
        delta_flow, delta_certainty = out[:, :-1], out[:, -1:]
        return delta_flow, delta_certainty


class RomaDecoder(nn.Module):
    """The full coarse-to-fine RoMa decoder: global GP + transformer matcher at 1/16, then convolutional refinement."""

    def __init__(self, config: RomaConfig):
        super().__init__()
        self.config = config
        self.embedding_decoder = RomaCoordinateDecoder(config)
        self.gps = nn.ModuleDict({"16": RomaGaussianProcess(config)})

        proj_in_dims = [config.backbone_config.hidden_size] + list(reversed(config.cnn_feature_dims))
        self.proj = nn.ModuleDict(
            {
                scale: nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1),
                    nn.BatchNorm2d(out_dim, eps=config.batch_norm_eps),
                )
                for scale, in_dim, out_dim in zip(ROMA_SCALES, proj_in_dims, config.proj_out_dims)
            }
        )
        self.conv_refiner = nn.ModuleDict(
            {
                scale: RomaConvRefiner(
                    in_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    out_dim=2 + 1,
                    kernel_size=config.refiner_kernel_size,
                    hidden_blocks=config.refiner_hidden_blocks,
                    displacement_emb_dim=emb_dim,
                    local_corr_radius=radius,
                    batch_norm_eps=config.batch_norm_eps,
                )
                for scale, hidden_dim, emb_dim, radius in zip(
                    ROMA_SCALES,
                    config.refiner_hidden_dims,
                    config.refiner_displacement_emb_dims,
                    config.refiner_local_corr_radius,
                )
            }
        )
        self.refine_init = 4

    def forward(
        self,
        query_pyramid: dict[int, torch.Tensor],
        support_pyramid: dict[int, torch.Tensor],
        scale_factor: float = 1.0,
        upsample: bool = False,
        init_flow: torch.Tensor | None = None,
        init_certainty: torch.Tensor | None = None,
    ) -> dict[int, dict[str, torch.Tensor]]:
        # The coarse pass runs every scale (16 -> 1) starting from a placeholder flow; the high-resolution upsample
        # pass runs only the fine scales (8 -> 1), initialized from the coarse flow/certainty.
        scales = ROMA_UPSAMPLE_SCALES if upsample else ROMA_SCALES
        sizes = {scale: query_pyramid[scale].shape[-2:] for scale in query_pyramid}
        finest_height, finest_width = sizes[1]
        coarsest_scale = int(scales[0])
        if upsample:
            flow = F.interpolate(init_flow, size=sizes[coarsest_scale], mode="bilinear", align_corners=False)
            certainty = F.interpolate(init_certainty, size=sizes[coarsest_scale], mode="bilinear", align_corners=False)
        else:
            flow = normalized_coordinate_grid(
                query_pyramid[coarsest_scale].shape[0], *sizes[coarsest_scale], query_pyramid[coarsest_scale].device
            )
            certainty = torch.zeros_like(flow[:, :1])

        correspondences = {}
        for scale in scales:
            scale_int = int(scale)
            query_features = self.proj[scale](query_pyramid[scale_int])
            support_features = self.proj[scale](support_pyramid[scale_int])

            if scale_int in ROMA_COARSE_SCALES:
                gp_posterior = self.gps[scale](query_features, support_features)
                class_logits, certainty = self.embedding_decoder(gp_posterior, query_features)
                flow = classification_to_flow(class_logits).permute(0, 3, 1, 2)

            delta_flow, delta_certainty = self.conv_refiner[scale](
                query_features, support_features, flow, scale_factor=scale_factor
            )
            displacement = scale_int * torch.stack(
                (
                    delta_flow[:, 0] / (self.refine_init * finest_width),
                    delta_flow[:, 1] / (self.refine_init * finest_height),
                ),
                dim=1,
            )
            flow = flow + displacement
            certainty = certainty + delta_certainty

            correspondences[scale_int] = {"certainty": certainty, "flow": flow}

            if scale != "1":
                flow = F.interpolate(flow, size=sizes[scale_int // 2], mode="bilinear").detach()
                certainty = F.interpolate(certainty, size=sizes[scale_int // 2], mode="bilinear").detach()
        return correspondences


@auto_docstring
class RomaPreTrainedModel(PreTrainedModel):
    config_class = RomaConfig
    base_model_prefix = "roma"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _no_split_modules = ["RomaTransformerBlock"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            init.ones_(module.weight)
            init.zeros_(module.bias)
            init.zeros_(module.running_mean)
            init.ones_(module.running_var)
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)


@auto_docstring(
    custom_intro="""
    The RoMa encoder: a frozen DINOv2 backbone providing robust coarse (1/16) features fused with a VGG-19 fine
    feature pyramid (strides 1, 2, 4, 8). Returns the full feature pyramid as `feature_maps`.
    """
)
class RomaModel(RomaPreTrainedModel):
    def __init__(self, config: RomaConfig):
        super().__init__(config)
        self.config = config
        self.backbone = load_backbone(config)
        self.cnn = RomaVGG19(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    @can_return_tuple
    @auto_docstring
    def forward(self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]) -> BackboneOutput:
        feature_pyramid = self.extract_feature_pyramid(pixel_values, **kwargs)
        feature_maps = tuple(feature_pyramid[scale] for scale in sorted(feature_pyramid))
        return BackboneOutput(feature_maps=feature_maps)

    def extract_feature_pyramid(
        self, pixel_values: torch.FloatTensor, upsample: bool = False, **kwargs
    ) -> dict[int, torch.Tensor]:
        # Accept either a batch of single images `(batch, 3, H, W)` or a batch of image pairs
        # `(batch, 2, 3, H, W)`; in the latter case the two images are stacked in blocks: `[all image0, all image1]`.
        if pixel_values.ndim == 5:
            pixel_values = torch.cat((pixel_values[:, 0], pixel_values[:, 1]), dim=0)
        feature_pyramid = self.cnn(pixel_values)
        # The high-resolution upsample pass uses only the fine VGG features and skips the (expensive) DINOv2 backbone.
        if not upsample:
            backbone_output = self.backbone(pixel_values, **kwargs)
            feature_pyramid[16] = backbone_output.feature_maps[-1]
        return feature_pyramid


@auto_docstring(
    custom_intro="""
    RoMa model for dense keypoint matching: it predicts a dense warp and certainty between a pair of images and
    samples sparse correspondences from them.
    """
)
class RomaForKeypointMatching(RomaPreTrainedModel):
    def __init__(self, config: RomaConfig):
        super().__init__(config)
        self.config = config
        self.roma = RomaModel(config)
        self.decoder = RomaDecoder(config)
        self.post_init()

    def sample_matches(
        self, warp: torch.Tensor, certainty: torch.Tensor, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample `num_samples` sparse correspondences from a single dense `warp` / `certainty`, density-balanced."""
        certainty = certainty.clone()
        certainty[certainty > self.config.sample_threshold] = 1.0
        matches = warp.reshape(-1, 4)
        certainty = certainty.reshape(-1)

        expansion_factor = 4
        num_candidates = min(expansion_factor * num_samples, len(certainty))
        candidate_idx = torch.multinomial(certainty, num_samples=num_candidates, replacement=False)
        candidate_matches, candidate_certainty = matches[candidate_idx], certainty[candidate_idx]

        density = kernel_density_estimate(candidate_matches, std=0.1)
        probability = 1 / (density + 1)
        probability[density < 10] = 1e-7
        balanced_idx = torch.multinomial(
            probability, num_samples=min(num_samples, len(candidate_certainty)), replacement=False
        )
        return candidate_matches[balanced_idx], candidate_certainty[balanced_idx]

    def _split_query_support(self, pyramid: dict[int, torch.Tensor]) -> tuple[dict, dict]:
        """Split a jointly-encoded `[all image0, all image1]` pyramid into query/support (symmetric or not)."""
        if self.config.symmetric:
            query = pyramid
            support = {
                scale: torch.cat((features.chunk(2)[1], features.chunk(2)[0]), dim=0)
                for scale, features in pyramid.items()
            }
        else:
            query = {scale: features.chunk(2)[0] for scale, features in pyramid.items()}
            support = {scale: features.chunk(2)[1] for scale, features in pyramid.items()}
        return query, support

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_values_upsampled: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> RomaKeypointMatchingOutput:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, 2, num_channels, height, width)`):
            Pairs of images to be matched, at the coarse resolution. The pair is the second dimension.
        pixel_values_upsampled (`torch.FloatTensor` of shape `(batch_size, 2, num_channels, up_height, up_width)`, *optional*):
            The same image pairs at the (higher) upsample resolution. Required to run the high-resolution refinement
            pass when `config.upsample_predictions=True`; the dense warp/certainty are then returned at this resolution.
        labels (`torch.LongTensor`, *optional*):
            Not supported yet; training is not implemented.
        """
        if labels is not None:
            raise NotImplementedError("Training of RoMa is not supported yet.")
        if pixel_values.ndim != 5 or pixel_values.shape[1] != 2:
            raise ValueError(
                f"`pixel_values` must have shape (batch, 2, channels, height, width), got {tuple(pixel_values.shape)}."
            )

        batch_size = pixel_values.shape[0]
        height, width = pixel_values.shape[-2:]

        # Coarse pass: encode both images jointly (block order), then run the full coarse-to-fine decoder.
        coarse_pyramid = self.roma.extract_feature_pyramid(pixel_values, **kwargs)
        query_pyramid, support_pyramid = self._split_query_support(coarse_pyramid)
        coarse = self.decoder(query_pyramid, support_pyramid, scale_factor=math.sqrt(height * width / (560**2)))

        if self.config.upsample_predictions and pixel_values_upsampled is not None:
            # High-resolution refinement pass: fine (VGG-only) features, decoder scales 8->1 initialized from the
            # coarse flow/certainty. The dense outputs are produced at the upsample resolution.
            up_height, up_width = pixel_values_upsampled.shape[-2:]
            hr_pyramid = self.roma.extract_feature_pyramid(pixel_values_upsampled, upsample=True)
            hr_query, hr_support = self._split_query_support(hr_pyramid)
            refined = self.decoder(
                hr_query,
                hr_support,
                scale_factor=math.sqrt(up_height * up_width / (560**2)),
                upsample=True,
                init_flow=coarse[1]["flow"],
                init_certainty=coarse[1]["certainty"],
            )
            warp, certainty = self._build_warp(refined[1], coarse[16], batch_size, up_height, up_width)
        else:
            warp, certainty = self._build_warp(coarse[1], coarse[16], batch_size, height, width)

        matches, matching_scores = [], []
        for b in range(batch_size):
            sampled_matches, sampled_scores = self.sample_matches(warp[b], certainty[b], self.config.num_samples)
            matches.append(sampled_matches)
            matching_scores.append(sampled_scores)
        matches = torch.stack(matches)
        matching_scores = torch.stack(matching_scores)

        return RomaKeypointMatchingOutput(
            matches=matches,
            matching_scores=matching_scores,
            warp=warp,
            certainty=certainty,
        )

    def _build_warp(
        self, finest: dict, coarse_scale16: dict, batch_size: int, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble the dense warp + certainty from the finest correspondences, mirroring RoMa's `match()`. The
        certainty is attenuated using the coarse (1/16) certainty, which always comes from the coarse pass.
        """
        flow = finest["flow"]
        certainty = finest["certainty"]

        if self.config.attenuate_certainty:
            low_res_certainty = F.interpolate(
                coarse_scale16["certainty"], size=(height, width), align_corners=False, mode="bilinear"
            )
            low_res_certainty = 0.5 * low_res_certainty * (low_res_certainty < 0)
            certainty = certainty - low_res_certainty

        flow = flow.permute(0, 2, 3, 1)
        certainty = certainty.sigmoid()

        # Zero out the certainty wherever the predicted warp leaves the image, then clamp the warp.
        out_of_bounds = (flow.abs() > 1).sum(dim=-1) > 0
        certainty[out_of_bounds[:, None]] = 0
        flow = torch.clamp(flow, -1, 1)

        query_coords = normalized_coordinate_grid(batch_size, height, width, flow.device, dtype=flow.dtype).permute(
            0, 2, 3, 1
        )
        if self.config.symmetric:
            forward_warp, backward_warp = flow.chunk(2)
            query_warp = torch.cat((query_coords, forward_warp), dim=-1)
            support_warp = torch.cat((backward_warp, query_coords), dim=-1)
            warp = torch.cat((query_warp, support_warp), dim=2)
            certainty_a, certainty_b = certainty.chunk(2)
            certainty_map = torch.cat((certainty_a, certainty_b), dim=3)[:, 0]
        else:
            warp = torch.cat((query_coords, flow), dim=-1)
            certainty_map = certainty[:, 0]
        return warp, certainty_map


__all__ = ["RomaPreTrainedModel", "RomaModel", "RomaForKeypointMatching"]
