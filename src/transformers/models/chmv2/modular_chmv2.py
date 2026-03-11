# Copyright 2026 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""CHMv2 model — Canopy Height Model v2, adapted from DPT."""

import torch
from torch import nn

from ... import initialization as init
from ...backbone_utils import consolidate_backbone_kwargs_to_config, load_backbone
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring, can_return_tuple, requires_backends
from ..auto import AutoConfig
from ..depth_anything.modeling_depth_anything import (
    DepthAnythingPreActResidualLayer,
)
from ..dpt.image_processing_dpt_fast import DPTImageProcessorFast
from ..dpt.modeling_dpt import DPTReassembleLayer, _get_backbone_hidden_size


@auto_docstring(checkpoint="facebook/dinov3-vitl16-chmv2-dpt-head")
class CHMv2Config(PreTrainedConfig):
    r"""
    backbone_config (`Union[dict, "PreTrainedConfig"]`, *optional*):
        The configuration of the backbone model. Only DINOv3ViTConfig is currently supported.
    patch_size (`int`, *optional*, defaults to 16):
        The patch size used by the backbone vision transformer.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    reassemble_factors (`list[float]`, *optional*, defaults to `[4, 2, 1, 0.5]`):
        The up/downsampling factors of the reassemble layers.
    post_process_channels (`list[int]`, *optional*, defaults to `[128, 256, 512, 1024]`):
        The output channel sizes of the reassemble stage for each backbone feature level.
    fusion_hidden_size (`int`, *optional*, defaults to 256):
        The number of channels before fusion.
    head_hidden_size (`int`, *optional*, defaults to 128):
        The number of channels in the hidden layer of the depth estimation head.
    number_output_channels (`int`, *optional*, defaults to 256):
        Number of output channels for the CHMv2 head (number of depth bins).
    readout_type (`str`, *optional*, defaults to `"project"`):
        Type of readout operation for the CLS token. One of `["ignore", "add", "project"]`.
    min_depth (`float`, *optional*, defaults to 0.001):
        The minimum depth value for depth bin calculation.
    max_depth (`float`, *optional*, defaults to 96.0):
        The maximum depth value for depth bin calculation.
    bins_strategy (`str`, *optional*, defaults to `"chmv2_mixlog"`):
        The strategy for depth bins distribution. One of `["linear", "log", "chmv2_mixlog"]`.
    norm_strategy (`str`, *optional*, defaults to `"chmv2_mixlog"`):
        The normalization strategy for depth prediction. One of `["linear", "softmax", "sigmoid", "chmv2_mixlog"]`.

    ```python
    >>> from transformers import CHMv2Config, CHMv2ForDepthEstimation

    >>> configuration = CHMv2Config()
    >>> model = CHMv2ForDepthEstimation(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "chmv2"
    sub_configs = {"backbone_config": AutoConfig}

    def __init__(
        self,
        backbone_config: dict | None = None,
        patch_size: int | None = 16,
        initializer_range: float | None = 0.02,
        reassemble_factors: list[float] | None = None,
        post_process_channels: list[int] | None = None,
        fusion_hidden_size: int | None = 256,
        head_hidden_size: int | None = 128,
        number_output_channels: int | None = 256,
        readout_type: str | None = "project",
        min_depth: float | None = 0.001,
        max_depth: float | None = 96.0,
        bins_strategy: str | None = "chmv2_mixlog",
        norm_strategy: str | None = "chmv2_mixlog",
        **kwargs,
    ):
        if reassemble_factors is None:
            reassemble_factors = [4, 2, 1, 0.5]
        if post_process_channels is None:
            post_process_channels = [128, 256, 512, 1024]

        default_config_kwargs = {
            "image_size": 416,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "num_register_tokens": 4,
            "key_bias": True,
            "out_indices": [6, 12, 18, 24],
            "reshape_hidden_states": True,
            "apply_layernorm": True,
            "layer_norm_eps": 1e-6,
            "return_class_token": True,
        }

        backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=backbone_config,
            default_config_type="dinov3_vit",
            default_config_kwargs=default_config_kwargs,
            **kwargs,
        )

        self.backbone_config = backbone_config
        self.patch_size = patch_size
        self.initializer_range = initializer_range
        self.reassemble_factors = reassemble_factors
        self.post_process_channels = post_process_channels
        self.fusion_hidden_size = fusion_hidden_size
        self.head_hidden_size = head_hidden_size
        self.number_output_channels = number_output_channels
        self.readout_type = readout_type

        if bins_strategy not in ["linear", "log", "chmv2_mixlog"]:
            raise ValueError("bins_strategy must be one of ['linear', 'log', 'chmv2_mixlog']")
        if norm_strategy not in ["linear", "softmax", "sigmoid", "chmv2_mixlog"]:
            raise ValueError("norm_strategy must be one of ['linear', 'softmax', 'sigmoid', 'chmv2_mixlog']")

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.bins_strategy = bins_strategy
        self.norm_strategy = norm_strategy

        super().__init__(**kwargs)


class CHMv2ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    ensure_multiple_of (`int`, *optional*, defaults to 1):
        If `do_resize` is `True`, the image is resized to a size that is a multiple of this value. Can be overridden
        by `ensure_multiple_of` in `preprocess`.
    keep_aspect_ratio (`bool`, *optional*, defaults to `False`):
        If `True`, the image is resized to the largest possible size such that the aspect ratio is preserved. Can
        be overridden by `keep_aspect_ratio` in `preprocess`.
    do_reduce_labels (`bool`, *optional*, defaults to `self.do_reduce_labels`):
        Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
        is used for background, and background itself is not included in all classes of a dataset (e.g.
        ADE20k). The background label will be replaced by 255.
    """

    ensure_multiple_of: int
    size_divisor: int
    keep_aspect_ratio: bool
    do_reduce_labels: bool


class CHMv2ImageProcessorFast(DPTImageProcessorFast):
    do_resize = False
    do_pad = True
    size_divisor = 16
    ensure_multiple_of = 16
    keep_aspect_ratio = True
    image_mean = [0.420, 0.411, 0.296]
    image_std = [0.213, 0.156, 0.143]
    valid_kwargs = CHMv2ImageProcessorKwargs

    def post_process_depth_estimation(
        self,
        outputs: "DepthEstimatorOutput",
        target_sizes: TensorType | list[tuple[int, int]] | None | None = None,
    ) -> list[dict[str, TensorType]]:
        """
        Converts the raw output of [`DepthEstimatorOutput`] into final depth predictions and depth PIL images.
        Only supports PyTorch.

        Args:
            outputs ([`DepthEstimatorOutput`]):
                Raw outputs of the model.
            target_sizes (`TensorType` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                (height, width) of each image in the batch. If left to None, predictions will not be resized.

        Returns:
            `List[Dict[str, TensorType]]`: A list of dictionaries of tensors representing the processed depth
            predictions.
        """
        requires_backends(self, "torch")

        predicted_depth = outputs.predicted_depth

        if (target_sizes is not None) and (len(predicted_depth) != len(target_sizes)):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the predicted depth"
            )

        results = []
        target_sizes = [None] * len(predicted_depth) if target_sizes is None else target_sizes
        for depth, target_size in zip(predicted_depth, target_sizes):
            if target_size is not None:
                depth = torch.nn.functional.interpolate(
                    depth[None, None, ...], size=target_size, mode="bilinear", align_corners=True
                ).squeeze()

            results.append({"predicted_depth": depth})

        return results


class CHMv2ReassembleLayer(DPTReassembleLayer):
    pass


class CHMv2ReassembleStage(nn.Module):
    """
    Reassemble stage that processes hidden states from the backbone into image-like feature
    representations at various resolutions.
    """

    def __init__(self, config: CHMv2Config):
        super().__init__()
        self.config = config
        self.readout_type = config.readout_type

        self.layers = nn.ModuleList()
        for out_channels, factor in zip(config.post_process_channels, config.reassemble_factors):
            self.layers.append(
                CHMv2ReassembleLayer(
                    config=config,
                    channels=out_channels,
                    factor=factor,
                )
            )

        hidden_size = _get_backbone_hidden_size(config)
        if self.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.layers)):
                self.readout_projects.append(nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.GELU()))

    def forward(self, hidden_states: list[torch.Tensor], patch_height=None, patch_width=None) -> list[torch.Tensor]:
        out = []

        for layer_idx, hidden_state in enumerate(hidden_states):
            if isinstance(hidden_state, (tuple, list)) and len(hidden_state) == 2:
                hidden_state, cls_token = hidden_state[0], hidden_state[1]
                feature_shape = hidden_state.shape

                if self.readout_type == "project":
                    hidden_state = hidden_state.flatten(2).transpose(1, 2)
                    readout = cls_token.unsqueeze(1).expand_as(hidden_state)
                    hidden_state = self.readout_projects[layer_idx](torch.cat((hidden_state, readout), -1))
                    hidden_state = hidden_state.permute(0, 2, 1).reshape(feature_shape)
                elif self.readout_type == "add":
                    hidden_state = hidden_state.flatten(2) + cls_token.unsqueeze(-1)
                    hidden_state = hidden_state.reshape(feature_shape)
            else:
                if hidden_state.dim() == 3:
                    hidden_state = hidden_state[:, 1:]
                    batch_size, _, num_channels = hidden_state.shape
                    hidden_state = hidden_state.reshape(batch_size, patch_height, patch_width, num_channels)
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()

            hidden_state = self.layers[layer_idx](hidden_state)
            out.append(hidden_state)

        return out


class CHMv2PreActResidualLayer(DepthAnythingPreActResidualLayer):
    pass


class CHMv2FeatureFusionLayer(nn.Module):
    def __init__(self, config: CHMv2Config, is_first_layer: bool = False):
        super().__init__()
        self.is_first_layer = is_first_layer

        self.projection = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=1, bias=True)

        if not is_first_layer:
            self.residual_layer1 = CHMv2PreActResidualLayer(config)

        self.residual_layer2 = CHMv2PreActResidualLayer(config)

    def forward(self, hidden_state, residual=None, size=None):
        if residual is not None and not self.is_first_layer:
            if hidden_state.shape != residual.shape:
                _, _, height, width = hidden_state.shape
                residual = nn.functional.interpolate(
                    residual, size=(height, width), mode="bilinear", align_corners=False
                )
            hidden_state = hidden_state + self.residual_layer1(residual)

        hidden_state = self.residual_layer2(hidden_state)

        modifier = {"scale_factor": 2} if size is None else {"size": size}

        hidden_state = nn.functional.interpolate(
            hidden_state,
            **modifier,
            mode="bilinear",
            align_corners=True,
        )

        hidden_state = self.projection(hidden_state)

        return hidden_state


class CHMv2UpsampleConvHead(nn.Module):
    """
    Convolutional head with intermediate upsampling.

    Architecture: Conv3x3 -> 2x bilinear upsample -> Conv3x3 -> ReLU -> Conv1x1.
    """

    def __init__(self, features, number_output_channels, n_hidden_channels=128):
        super().__init__()
        self.head = nn.ModuleList(
            [
                nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(features // 2, n_hidden_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(n_hidden_channels, number_output_channels, kernel_size=1, stride=1, padding=0),
            ]
        )

    def forward(self, hidden_states):
        for layer in self.head:
            hidden_states = layer(hidden_states)
        return hidden_states


class CHMv2Head(nn.Module):
    """
    CHMv2 dense-prediction head adapted from DPT.

    Integrates reassemble, projection convs, feature fusion, and UpConv depth head.
    """

    def __init__(self, config: CHMv2Config):
        super().__init__()
        self.config = config

        self.reassemble_stage = CHMv2ReassembleStage(config)

        self.convs = nn.ModuleList()
        for channel in config.post_process_channels:
            self.convs.append(nn.Conv2d(channel, config.fusion_hidden_size, kernel_size=3, padding=1, bias=False))

        self.fusion_layers = nn.ModuleList()
        for idx in range(len(config.post_process_channels)):
            self.fusion_layers.append(CHMv2FeatureFusionLayer(config, is_first_layer=(idx == 0)))

        self.conv_depth = CHMv2UpsampleConvHead(
            features=config.fusion_hidden_size,
            number_output_channels=config.number_output_channels,
            n_hidden_channels=config.head_hidden_size,
        )

    def forward_features(self, hidden_states: list[torch.Tensor], patch_height: int, patch_width: int) -> torch.Tensor:
        hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)

        features = [self.convs[i](feature) for i, feature in enumerate(hidden_states)]
        features.reverse()

        fused_hidden_state = self.fusion_layers[0](features[0])
        for i in range(1, len(self.fusion_layers)):
            fused_hidden_state = self.fusion_layers[i](fused_hidden_state, features[i])

        return fused_hidden_state

    def forward(self, hidden_states: list[torch.Tensor], patch_height: int, patch_width: int) -> torch.Tensor:
        out = self.forward_features(hidden_states, patch_height, patch_width)
        out = self.conv_depth(out)
        return out


class CHMv2FeaturesToDepth(nn.Module):
    """Converts raw logits from the CHMv2 head into a depth map using depth bins."""

    def __init__(self, config: CHMv2Config):
        super().__init__()
        self.min_depth = config.min_depth
        self.max_depth = config.max_depth
        self.bins_strategy = config.bins_strategy
        self.norm_strategy = config.norm_strategy
        self._mixlog_max_clamp_value = 1e-4
        self._mixlog_eps_shift = 1e-8
        self._mixlog_eps = 1e-12

    def _create_mixlog_bins(self, n_bins: int, device: torch.device) -> torch.Tensor:
        """
        Creates mixed log bins interpolated between linear and log distributions.

        The max_depth is divided by 8.0 internally; this scaling is reversed in
        `_create_outputs_with_mixlog_norm` by multiplying by 8.0.
        """
        scaled_max_depth = self.max_depth / 8.0
        linear = torch.linspace(self.min_depth, scaled_max_depth, n_bins, device=device)
        log = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(self.min_depth, device=device)),
                torch.log(torch.tensor(scaled_max_depth, device=device)),
                n_bins,
                device=device,
            )
        )
        interp_weight = torch.linspace(1.0, 0.0, n_bins, device=device)
        bins = interp_weight * log + (1.0 - interp_weight) * linear
        return bins

    def _create_outputs_with_mixlog_norm(self, input: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        """Converts depth bin logits to depth values using mixlog normalization."""
        logits = torch.relu(input)

        min_per_sample = logits.amin(dim=1, keepdim=True)
        shift = (-min_per_sample).clamp_min(0.0).clamp_max(self._mixlog_max_clamp_value) + self._mixlog_eps_shift
        logits_pos = logits + shift

        denom = logits_pos.sum(dim=1, keepdim=True)
        denom = torch.nan_to_num(denom, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(self._mixlog_eps)
        weights = logits_pos / denom

        bins_broadcast = bins.view(1, -1, 1, 1).clamp_min(self._mixlog_eps)
        output = (weights * bins_broadcast).sum(dim=1, keepdim=True).clamp_min(self._mixlog_eps)

        output = output * 8.0

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_bins = x.shape[1]

        if n_bins > 1:
            if self.bins_strategy == "linear":
                bins = torch.linspace(self.min_depth, self.max_depth, n_bins, device=x.device)
            elif self.bins_strategy == "log":
                bins = torch.linspace(
                    torch.log(torch.tensor(self.min_depth)),
                    torch.log(torch.tensor(self.max_depth)),
                    n_bins,
                    device=x.device,
                )
                bins = torch.exp(bins)
            else:
                bins = self._create_mixlog_bins(n_bins, x.device)

            if self.norm_strategy in ["linear", "softmax", "sigmoid"]:
                if self.norm_strategy == "linear":
                    logit = torch.relu(x)
                    eps = 0.1
                    logit = logit + eps
                    logit = logit / logit.sum(dim=1, keepdim=True)
                elif self.norm_strategy == "softmax":
                    logit = torch.softmax(x, dim=1)
                else:
                    logit = torch.sigmoid(x)
                    logit = logit / logit.sum(dim=1, keepdim=True)
                output = torch.einsum("ikmn,k->imn", [logit, bins]).unsqueeze(dim=1)
            else:
                output = self._create_outputs_with_mixlog_norm(x, bins)
        else:
            output = torch.relu(x) + self.min_depth

        return output


@auto_docstring
class CHMv2PreTrainedModel(PreTrainedModel):
    config: CHMv2Config
    base_model_prefix = "chmv2"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module) -> None:
        super()._init_weights(module)
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            init.trunc_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)


@auto_docstring(
    custom_intro="""
    CHMv2 Model with a depth estimation head on top (consisting of convolutional layers) e.g. for canopy height
    estimation.
    """
)
class CHMv2ForDepthEstimation(CHMv2PreTrainedModel):
    def __init__(self, config: CHMv2Config):
        super().__init__(config)

        self.backbone = load_backbone(config)
        self.head = CHMv2Head(config)
        self.features_to_depth = CHMv2FeaturesToDepth(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DepthEstimatorOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.
        """
        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size

        backbone_output = self.backbone(pixel_values, **kwargs)
        intermediate_features = list(zip(backbone_output.feature_maps, backbone_output.cls_tokens))

        head_output = self.head(intermediate_features, patch_height, patch_width)

        predicted_depth = self.features_to_depth(head_output)
        predicted_depth = predicted_depth.squeeze(dim=1)

        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=backbone_output.hidden_states,
            attentions=backbone_output.attentions,
        )


__all__ = [
    "CHMv2Config",
    "CHMv2ImageProcessorFast",
    "CHMv2ForDepthEstimation",
    "CHMv2PreTrainedModel",
]
