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
"""PyTorch CHMv2 model.

This is adapted from DPT (https://arxiv.org/abs/2103.13413) with the following changes:
- Uses a 4-layer Convolutional head with intermediate upsampling (instead of 3)
- No projection layer after fusion blocks
- A Conv2D layer is applied after UpConvHead
"""

import torch
from torch import nn

from ...backbone_utils import load_backbone
from ...modeling_outputs import DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from .configuration_chmv2 import CHMv2Config


logger = logging.get_logger(__name__)


class CHMv2Interpolate(nn.Module):
    """Interpolation module for upsampling feature maps."""

    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )


class CHMv2ConvModule(nn.Module):
    """
    A conv block that bundles conv/norm/activation layers.

    Args:
        in_channels (`int`): Number of channels in the input feature map.
        out_channels (`int`): Number of channels produced by the convolution.
        kernel_size (`int`): Size of the convolving kernel.
        stride (`int`): Stride of the convolution.
        padding (`int`): Zero-padding added to both sides of the input.
        bias (`bool`): Whether to use bias in convolution.
        use_norm (`bool`): Whether to use normalization layer.
        use_activation (`bool`): Whether to use activation layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        use_norm=False,
        use_activation=True,
    ):
        super().__init__()
        self.use_norm = use_norm
        self.use_activation = use_activation

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        if self.use_norm:
            self.norm = nn.SyncBatchNorm(out_channels)

        if self.use_activation:
            self.activation = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        if self.use_activation:
            x = self.activation(x)
        return x


class CHMv2ReassembleLayer(nn.Module):
    """Reassemble layer that projects and resizes feature maps."""

    def __init__(self, in_channels, out_channels, factor, use_batchnorm=False):
        super().__init__()
        self.projection = CHMv2ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
        )

        if factor > 1:
            self.resize = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=int(factor), stride=int(factor), padding=0
            )
        elif factor == 1:
            self.resize = nn.Identity()
        else:
            self.resize = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=int(1 / factor), padding=1)

        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.batchnorm = nn.SyncBatchNorm(in_channels)
        else:
            self.batchnorm = nn.Identity()

    def forward(self, hidden_state):
        hidden_state = self.batchnorm(hidden_state)
        hidden_state = self.projection(hidden_state)
        hidden_state = self.resize(hidden_state)
        return hidden_state


class CHMv2ReassembleStage(nn.Module):
    """
    Reassemble stage that processes hidden states from the backbone.

    This class reassembles the hidden states of the backbone into image-like feature representations at various
    resolutions.

    Args:
        config (`CHMv2Config`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config: CHMv2Config):
        super().__init__()
        self.config = config
        self.readout_type = config.readout_type

        in_channels = [config.reassemble_hidden_size] * len(config.neck_hidden_sizes)
        factors = config.reassemble_factors

        self.layers = nn.ModuleList()
        for idx, (out_channels, factor) in enumerate(zip(config.neck_hidden_sizes, factors)):
            self.layers.append(
                CHMv2ReassembleLayer(
                    in_channels=in_channels[idx],
                    out_channels=out_channels,
                    factor=factor,
                    use_batchnorm=config.use_batchnorm,
                )
            )

        if self.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            for idx in range(len(self.layers)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels[idx], in_channels[idx]), nn.GELU())
                )

    def forward(self, hidden_states: list[torch.Tensor], patch_height=None, patch_width=None) -> list[torch.Tensor]:
        """
        Args:
            hidden_states (`list[torch.FloatTensor]`):
                List of hidden states from the backbone. Each element is a tuple of (feature_map, cls_token).
        """
        out = []

        for i, hidden_state in enumerate(hidden_states):
            if isinstance(hidden_state, (tuple, list)) and len(hidden_state) == 2:
                x, cls_token = hidden_state[0], hidden_state[1]
                feature_shape = x.shape

                if self.readout_type == "project":
                    x = x.flatten(2).permute((0, 2, 1))
                    readout = cls_token.unsqueeze(1).expand_as(x)
                    x = self.readout_projects[i](torch.cat((x, readout), -1))
                    x = x.permute(0, 2, 1).reshape(feature_shape)
                elif self.readout_type == "add":
                    x = x.flatten(2) + cls_token.unsqueeze(-1)
                    x = x.reshape(feature_shape)
            else:
                x = hidden_state
                if x.dim() == 3:
                    x = x[:, 1:]
                    batch_size, _, num_channels = x.shape
                    x = x.reshape(batch_size, patch_height, patch_width, num_channels)
                    x = x.permute(0, 3, 1, 2).contiguous()

            x = self.layers[i](x)
            out.append(x)

        return out


class CHMv2PreActResidualLayer(nn.Module):
    """
    Pre-activate residual unit.

    Args:
        config (`CHMv2Config`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config: CHMv2Config):
        super().__init__()

        self.activation1 = nn.ReLU()
        self.convolution1 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=config.use_bias,
        )

        self.activation2 = nn.ReLU()
        self.convolution2 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=config.use_bias,
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.activation1(hidden_state)
        hidden_state = self.convolution1(hidden_state)
        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)

        return hidden_state + residual


class CHMv2FeatureFusionLayer(nn.Module):
    """
    Feature fusion layer, merges feature maps from different stages.

    This layer follows the CHMv2 design where there is NO projection layer after fusion.

    Args:
        config (`CHMv2Config`):
            Model configuration class defining the model architecture.
        is_first_layer (`bool`):
            Whether this is the first fusion layer (which has no res_conv_unit1).
    """

    def __init__(self, config: CHMv2Config, is_first_layer: bool = False):
        super().__init__()
        self.is_first_layer = is_first_layer

        if not is_first_layer:
            self.residual_layer1 = CHMv2PreActResidualLayer(config)

        self.residual_layer2 = CHMv2PreActResidualLayer(config)

    def forward(self, hidden_state, residual=None, size=None):
        if residual is not None and not self.is_first_layer:
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(
                    residual, size=(hidden_state.shape[2], hidden_state.shape[3]), mode="bilinear", align_corners=False
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

        return hidden_state


class CHMv2UpConvHead(nn.Module):
    """
    A 4-layer Convolutional head with intermediate upsampling.
    kaiming_init is used on the 2nd Conv2d module.

    This follows the CHMv2 design from dinov3:
    - Conv3x3(features, features // 2)
    - 2x-Upsampling (bilinear)
    - Conv3x3(features // 2, n_hidden_channels)
    - ReLU
    - Conv1x1(n_hidden_channels, n_output_channels)

    Args:
        features (`int`): Number of input channels.
        n_output_channels (`int`): Number of output channels.
        n_hidden_channels (`int`): Number of channels in hidden layer. Default: 128.
    """

    def __init__(self, features, n_output_channels, n_hidden_channels=128):
        super().__init__()
        self.n_output_channels = n_output_channels

        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            CHMv2Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, n_hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_hidden_channels, n_output_channels, kernel_size=1, stride=1, padding=0),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.head[2].weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.head(x)


class CHMv2Head(nn.Module):
    """
    CHMv2 Head for dense prediction.

    This head is adapted from DPT (https://arxiv.org/abs/2103.13413) with the following changes:
    - no projection layer after fusion blocks
    - bias is applied in FeatureFusionBlocks.
    - a kaiming init is applied in the second Conv layer of UpConvHead (UpConvHeadCHMv2)

    This head integrates all processing stages:
    - Reassemble stage: processes backbone features
    - Convs: projects features to fusion_hidden_size
    - Fusion blocks: merges features from different scales
    - conv_depth: convolutional head with upsampling

    Args:
        config (`CHMv2Config`): Model configuration.
    """

    def __init__(self, config: CHMv2Config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size

        self.reassemble_stage = CHMv2ReassembleStage(config)

        self.convs = nn.ModuleList()
        for channel in config.neck_hidden_sizes:
            self.convs.append(nn.Conv2d(channel, config.fusion_hidden_size, kernel_size=3, padding=1, bias=False))

        self.fusion_layers = nn.ModuleList()
        for idx in range(len(config.neck_hidden_sizes)):
            self.fusion_layers.append(CHMv2FeatureFusionLayer(config, is_first_layer=(idx == 0)))

        self.conv_depth = CHMv2UpConvHead(
            features=config.fusion_hidden_size,
            n_output_channels=config.n_output_channels,
            n_hidden_channels=config.head_hidden_size,
        )

    def forward_features(
        self, hidden_states: list[torch.Tensor], patch_height: int, patch_width: int
    ) -> torch.Tensor:
        """
        Process features through reassemble, convs, and fusion stages.

        Args:
            hidden_states: List of hidden states from backbone
            patch_height: Height in patches
            patch_width: Width in patches

        Returns:
            Fused feature tensor
        """
        hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)

        features = [self.convs[i](feature) for i, feature in enumerate(hidden_states)]

        features = features[::-1]

        fused_hidden_state = self.fusion_layers[0](features[0])

        for i in range(1, len(self.fusion_layers)):
            fused_hidden_state = self.fusion_layers[i](fused_hidden_state, features[i])

        return fused_hidden_state

    def forward(self, hidden_states: list[torch.Tensor], patch_height: int, patch_width: int) -> torch.Tensor:
        out = self.forward_features(hidden_states, patch_height, patch_width)
        out = self.conv_depth(out)
        return out


def _create_chmv2_mixlog_bins(min_depth: float, max_depth: float, n_bins: int, device: torch.device) -> torch.Tensor:
    """
    Creates mixed log bins for the CHMv2 model.
    Bins are interpolated between linear and log distributions.

    Note: max_depth is divided by 8.0 because the CHMv2 model was trained
    with internally scaled depth values. The scaling is reversed in
    `_create_outputs_with_chmv2_mixlog_norm` by multiplying by 8.0.
    """
    scaled_max_depth = max_depth / 8.0
    linear = torch.linspace(min_depth, scaled_max_depth, n_bins, device=device)
    log = torch.exp(
        torch.linspace(
            torch.log(torch.tensor(min_depth, device=device)),
            torch.log(torch.tensor(scaled_max_depth, device=device)),
            n_bins,
            device=device,
        )
    )
    t = torch.linspace(1.0, 0.0, n_bins, device=device)
    bins = t * log + (1.0 - t) * linear
    return bins


def _create_outputs_with_chmv2_mixlog_norm(
    input: torch.Tensor,
    bins: torch.Tensor,
    max_clamp_value: float = 1e-4,
    eps_shift: float = 1e-8,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Converts depth bin logits to depth values using mixlog normalization.

    This function implements a "soft-argmax" style depth prediction, where the output
    is a weighted sum of depth bins, with weights derived from the input logits.
    The CHMv2 model outputs values that are 8x smaller than actual depth in meters,
    so we multiply by 8.0 at the end.

    Args:
        input: Raw logits from the decoder head.
        bins: Depth bin centers created by `_create_chmv2_mixlog_bins`.
        max_clamp_value: Maximum value for the positive shift.
        eps_shift: Epsilon added to shift to prevent division by zero.
        eps: Epsilon for numerical stability in division and final clamping.

    Returns:
        Depth map in meters (after x8.0 to the outputs).
    """
    y = torch.relu(input)

    m = y.amin(dim=1, keepdim=True)
    shift = (-m).clamp_min(0.0).clamp_max(max_clamp_value) + eps_shift
    y_pos = y + shift

    denom = y_pos.sum(dim=1, keepdim=True)
    denom = torch.nan_to_num(denom, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(eps)
    weights = y_pos / denom

    bins_broadcast = bins.view(1, -1, 1, 1).clamp_min(eps)
    output = (weights * bins_broadcast).sum(dim=1, keepdim=True).clamp_min(eps)

    output = output * 8.0

    return output


class CHMv2FeaturesToDepth(nn.Module):
    """
    Module that converts feature maps (logits) into a depth map.

    This module converts the raw output logits from the CHMv2 head into depth values
    using depth bins and various normalization strategies.

    Args:
        config (`CHMv2Config`): Model configuration.
    """

    def __init__(self, config: CHMv2Config):
        super().__init__()
        self.min_depth = config.min_depth
        self.max_depth = config.max_depth
        self.bins_strategy = config.bins_strategy
        self.norm_strategy = config.norm_strategy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (`torch.Tensor`): Raw logits of shape (N, n_bins, H, W).

        Returns:
            `torch.Tensor`: Depth map of shape (N, 1, H, W).
        """
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
                bins = _create_chmv2_mixlog_bins(self.min_depth, self.max_depth, n_bins, x.device)

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
                output = _create_outputs_with_chmv2_mixlog_norm(x, bins)
        else:
            output = torch.relu(x) + self.min_depth

        return output


@auto_docstring
class CHMv2PreTrainedModel(PreTrainedModel):
    config_class = CHMv2Config
    base_model_prefix = "chmv2"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    supports_gradient_checkpointing = True


@auto_docstring(
    custom_intro="""
    CHMv2 Model with a depth estimation head on top (consisting of convolutional layers) e.g. for canopy height estimation.
    """
)
class CHMv2ForCanopyHeightEstimation(CHMv2PreTrainedModel):
    _no_split_modules = ["DINOv3ViTEmbeddings"]

    def __init__(self, config: CHMv2Config):
        super().__init__(config)

        self.backbone = load_backbone(config)
        self.head = CHMv2Head(config)
        self.features_to_depth = CHMv2FeaturesToDepth(config)

        self.post_init()

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | DepthEstimatorOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/chmv2-large-hf")
        >>> model = AutoModelForDepthEstimation.from_pretrained("facebook/chmv2-large-hf")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # interpolate to original size
        >>> post_processed_output = image_processor.post_process_depth_estimation(
        ...     outputs,
        ...     target_sizes=[(image.height, image.width)],
        ... )

        >>> # visualize the prediction
        >>> predicted_depth = post_processed_output[0]["predicted_depth"]
        >>> depth = predicted_depth * 255 / predicted_depth.max()
        >>> depth = depth.detach().cpu().numpy()
        >>> depth = Image.fromarray(depth.astype("uint8"))
        ```"""
        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        hidden_states = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size

        head_output = self.head(hidden_states, patch_height, patch_width)

        depth_logits = nn.functional.interpolate(
            head_output,
            (int(patch_height * patch_size), int(patch_width * patch_size)),
            mode="bilinear",
            align_corners=True,
        )

        predicted_depth = self.features_to_depth(depth_logits)
        predicted_depth = predicted_depth.squeeze(dim=1)

        if not return_dict:
            if output_hidden_states:
                output = (predicted_depth,) + outputs[1:]
            else:
                output = (predicted_depth,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


__all__ = ["CHMv2ForCanopyHeightEstimation", "CHMv2PreTrainedModel"]
