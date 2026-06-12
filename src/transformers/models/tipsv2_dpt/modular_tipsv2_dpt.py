# Copyright 2026 Google LLC and the HuggingFace Inc. team. All rights reserved.
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
"""TIPSv2-DPT model."""

from collections.abc import Sized
from dataclasses import dataclass

import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from typing_extensions import Unpack

from ...activations import ACT2FN
from ...backbone_utils import consolidate_backbone_kwargs_to_config, load_backbone
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import DepthEstimatorOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..auto import AutoConfig
from ..dpt.modeling_dpt import DPTReassembleLayer
from ..tipsv2.image_processing_tipsv2 import Tipsv2ImageProcessor


logger = logging.get_logger(__name__)


@auto_docstring
class Tipsv2DptImageProcessor(Tipsv2ImageProcessor):
    def post_process_depth_estimation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]] | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Converts the output of a depth estimation model into final depth predictions.

        Args:
            outputs:
                Raw outputs of the model. Must have a `predicted_depth` attribute of shape
                `(batch_size, height, width)`.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of `(height, width)` tuples giving the desired output size for each image.
                When provided, each depth map is resized with bilinear interpolation. If `None`,
                predictions are returned at the decoder resolution.

        Returns:
            `list[dict[str, torch.Tensor]]`: One dict per image with key `"predicted_depth"`,
            mapping to a tensor of shape `(height, width)`.
        """
        predicted_depth = outputs.predicted_depth  # (B, H, W)

        if target_sizes is not None and len(predicted_depth) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the predicted depth"
            )

        target_sizes = [None] * len(predicted_depth) if target_sizes is None else target_sizes
        results = []
        for depth, target_size in zip(predicted_depth, target_sizes):
            if target_size is not None:
                depth = nn.functional.interpolate(
                    depth[None, None], size=target_size, mode="bilinear", align_corners=False
                ).squeeze()
            results.append({"predicted_depth": depth})
        return results

    def post_process_normal_estimation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]] | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Converts the output of a normal estimation model into L2-normalized surface normal maps.

        Args:
            outputs:
                Raw outputs of the model. Must have a `normals` attribute of shape
                `(batch_size, 3, height, width)`.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of `(height, width)` tuples giving the desired output size for each image.
                When provided, each normal map is resized with bicubic interpolation and then
                re-normalized. If `None`, predictions are returned at the decoder resolution.

        Returns:
            `list[dict[str, torch.Tensor]]`: One dict per image with key `"normals"`,
            mapping to an L2-normalized tensor of shape `(3, height, width)`.
        """
        normals = outputs.normals  # (B, 3, H, W)
        normals = nn.functional.normalize(normals, p=2, dim=1)

        if target_sizes is not None and len(normals) != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the normals output"
            )

        target_sizes = [None] * len(normals) if target_sizes is None else target_sizes
        results = []
        for normal, target_size in zip(normals, target_sizes):
            if target_size is not None:
                normal = nn.functional.interpolate(
                    normal.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
                ).squeeze(0)
            results.append({"normals": normal})
        return results

    def post_process_semantic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]] | None = None,
    ) -> list[torch.Tensor]:
        """
        Converts the output of a semantic segmentation model into semantic segmentation maps.

        Args:
            outputs:
                Raw outputs of the model. Must have a `logits` attribute of shape
                `(batch_size, num_labels, height, width)`.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of `(height, width)` tuples giving the desired output size for each image.
                When provided, logits are resized with bilinear interpolation before argmax.
                If `None`, argmax is computed at the decoder resolution.

        Returns:
            `list[torch.Tensor]`: One tensor per image of shape `(height, width)` containing
            the predicted class index for each pixel.
        """
        logits = outputs.logits  # (B, num_labels, H, W)

        if target_sizes is not None and len(logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")

        semantic_segmentation = []
        for idx in range(len(logits)):
            logit = logits[idx].unsqueeze(0)
            if target_sizes is not None:
                logit = nn.functional.interpolate(logit, size=target_sizes[idx], mode="bilinear", align_corners=False)
            semantic_segmentation.append(logit[0].argmax(dim=0))
        return semantic_segmentation


@auto_docstring
@strict
class Tipsv2DptConfig(PreTrainedConfig):
    r"""
    neck_hidden_sizes (`list[int]`, *optional*, defaults to `[96, 192, 384, 768]`):
        Output channel counts for the four reassemble stages. Default corresponds to
        `hidden_size // [8, 4, 2, 1]` for the b14 backbone (`hidden_size=768`). For
        other variants scale proportionally: l14 → `[128, 256, 512, 1024]`,
        so400m14 → `[144, 288, 576, 1152]`, g14 → `[192, 384, 768, 1536]`.
    fusion_hidden_size (`int`, *optional*, defaults to 256):
        Number of channels throughout the DPT feature-fusion neck.
    reassemble_factors (`list[float]`, *optional*, defaults to `[4, 2, 1, 0.5]`):
        Up/down-sample factors applied at each reassemble stage (shallowest to
        deepest). A factor > 1 uses a transposed convolution; a factor < 1 uses a
        strided convolution; a factor of 1 is the identity.
    readout_act (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
        Activation applied after the CLS-token readout projection. The original
        implementation uses the tanh-approximation variant of GELU; the default
        `"gelu_pytorch_tanh"` matches that numerically.
    num_depth_bins (`int`, *optional*, defaults to 256):
        Number of depth bins used by the depth-estimation head. The head predicts a
        probability distribution over `num_depth_bins` evenly-spaced values between
        `min_depth` and `max_depth` and computes a soft-expectation depth.
    min_depth (`float`, *optional*, defaults to 0.001):
        Lower bound (metres) of the depth-bin range.
    max_depth (`float`, *optional*, defaults to 10.0):
        Upper bound (metres) of the depth-bin range.
    semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
        Label index to ignore in the cross-entropy loss for semantic segmentation.

    Example:

    ```python
    >>> from transformers import Tipsv2DptConfig, Tipsv2DptForDepthEstimation

    >>> configuration = Tipsv2DptConfig()
    >>> model = Tipsv2DptForDepthEstimation(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "tipsv2_dpt"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    neck_hidden_sizes: list[int] | tuple[int, ...] = (96, 192, 384, 768)
    fusion_hidden_size: int = 256
    reassemble_factors: list[int | float] | tuple[int | float, ...] = (4, 2, 1, 0.5)
    readout_act: str = "gelu_pytorch_tanh"
    num_depth_bins: int = 256
    min_depth: float = 0.001
    max_depth: float = 10.0
    semantic_loss_ignore_index: int = 255

    def __post_init__(self, **kwargs):
        # Use num_labels and label2id when set. Otherwise fall back to num_labels=num_seg_classes which
        # is set on original configs.
        num_labels = kwargs.get("num_labels")
        label2id = kwargs.get("label2id")
        num_seg_classes = kwargs.pop("num_seg_classes", None)
        if num_labels is not None and num_seg_classes is not None and num_labels != num_seg_classes:
            logger.info(
                f"`num_labels` ({num_labels}) and `num_seg_classes` ({num_seg_classes}) are both set in"
                "the config and are not equal. The value from `num_labels` will be used."
            )
        elif (
            label2id is not None
            and num_seg_classes is not None
            and isinstance(label2id, Sized)
            and len(label2id) != num_seg_classes
        ):
            logger.info(
                f"`label2id` (len={len(label2id)}) and `num_seg_classes` ({num_seg_classes}) are both set in"
                "the config and are not equal. The value from `label2id` will be used."
            )
        if num_labels is None and label2id is None and num_seg_classes is not None:
            kwargs["num_labels"] = num_seg_classes

        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="tipsv2_vision_model",
            default_config_kwargs={
                "out_indices": [3, 6, 9, 12],
                "apply_layernorm": True,
                "reshape_hidden_states": False,
            },
            **kwargs,
        )
        super().__post_init__(**kwargs)


class Tipsv2DptReassembleLayer(DPTReassembleLayer):
    pass


class Tipsv2DptReassembleStage(nn.Module):
    """
    Reassembles TIPSv2 backbone token sequences into spatial feature maps at multiple resolutions.

    Each input is a token sequence `(B, 1 + num_register_tokens + N, D)`. The stage:
    1. Slices the CLS token and drops register tokens, keeping the N patch tokens.
    2. Applies an always-on "project" CLS-readout: concat(patches, CLS) → Linear(2D→D) → activation.
    3. Reshapes to `(B, D, patch_height, patch_width)`.
    4. Applies a per-stage 1×1 projection conv and spatial resize.
    """

    def __init__(self, config: Tipsv2DptConfig):
        super().__init__()
        self.num_register_tokens = config.backbone_config.num_register_tokens
        hidden_size = config.backbone_config.hidden_size

        self.readout_projects = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), ACT2FN[config.readout_act])
                for _ in config.neck_hidden_sizes
            ]
        )
        self.layers = nn.ModuleList(
            [
                Tipsv2DptReassembleLayer(config, channels=channels, factor=factor)
                for channels, factor in zip(config.neck_hidden_sizes, config.reassemble_factors)
            ]
        )

    def forward(
        self,
        hidden_states: list[torch.Tensor],
        patch_height: int,
        patch_width: int,
    ) -> list[torch.Tensor]:
        out = []
        for stage_idx, hidden_state in enumerate(hidden_states):
            cls_token = hidden_state[:, 0]  # (B, D)
            patch_tokens = hidden_state[:, 1 + self.num_register_tokens :]  # (B, N, D)
            batch_size, num_patches, hidden_size = patch_tokens.shape

            readout = cls_token.unsqueeze(1).expand(-1, num_patches, -1)
            patch_tokens = self.readout_projects[stage_idx](torch.cat([patch_tokens, readout], dim=-1))

            patch_tokens = patch_tokens.reshape(batch_size, patch_height, patch_width, hidden_size)
            patch_tokens = patch_tokens.permute(0, 3, 1, 2).contiguous()

            patch_tokens = self.layers[stage_idx](patch_tokens)
            out.append(patch_tokens)

        return out


class Tipsv2DptPreActResidualLayer(nn.Module):
    def __init__(self, config: Tipsv2DptConfig):
        super().__init__()
        self.activation1 = nn.ReLU()
        self.convolution1 = nn.Conv2d(
            config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.activation2 = nn.ReLU()
        self.convolution2 = nn.Conv2d(
            config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.activation1(hidden_state)
        hidden_state = self.convolution1(hidden_state)
        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)
        return hidden_state + residual


class Tipsv2DptFeatureFusionLayer(nn.Module):
    def __init__(self, config: Tipsv2DptConfig, align_corners: bool = True, has_residual: bool = True):
        super().__init__()
        self.align_corners = align_corners
        self.has_residual = has_residual
        self.projection = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=1, bias=True)
        if has_residual:
            self.residual_layer1 = Tipsv2DptPreActResidualLayer(config)
        self.residual_layer2 = Tipsv2DptPreActResidualLayer(config)

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is not None and self.has_residual:
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(
                    residual, size=(hidden_state.shape[2], hidden_state.shape[3]), mode="bilinear", align_corners=False
                )
            hidden_state = hidden_state + self.residual_layer1(residual)

        hidden_state = self.residual_layer2(hidden_state)
        hidden_state = nn.functional.interpolate(
            hidden_state, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        hidden_state = self.projection(hidden_state)
        return hidden_state


class Tipsv2DptFeatureFusionStage(nn.Module):
    def __init__(self, config: Tipsv2DptConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Tipsv2DptFeatureFusionLayer(config, has_residual=(idx > 0))
                for idx in range(len(config.neck_hidden_sizes))
            ]
        )

    def forward(self, hidden_states: list[torch.Tensor]) -> list[torch.Tensor]:
        fused_hidden_states = []
        fused_hidden_state = None
        for hidden_state, layer in zip(reversed(hidden_states), self.layers):
            if fused_hidden_state is None:
                fused_hidden_state = layer(hidden_state)
            else:
                fused_hidden_state = layer(fused_hidden_state, hidden_state)
            fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states


class Tipsv2DptNeck(nn.Module):
    """
    Neck module: reassemble → 3×3 channel-projection convs → feature fusion → trailing 3×3 project conv.

    Returns a single spatial feature map `(B, fusion_hidden_size, H', W')`.
    """

    def __init__(self, config: Tipsv2DptConfig):
        super().__init__()
        self.reassemble_stage = Tipsv2DptReassembleStage(config)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(channel, config.fusion_hidden_size, kernel_size=3, padding=1, bias=False)
                for channel in config.neck_hidden_sizes
            ]
        )

        self.fusion_stage = Tipsv2DptFeatureFusionStage(config)

        # trailing project conv not present in DPTNeck — part of the original DPTHead
        self.project = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=3, padding=1)

    def forward(
        self,
        hidden_states: list[torch.Tensor],
        patch_height: int,
        patch_width: int,
    ) -> torch.Tensor:
        hidden_states = self.reassemble_stage(hidden_states, patch_height=patch_height, patch_width=patch_width)
        features = [self.convs[idx](feature) for idx, feature in enumerate(hidden_states)]
        fused = self.fusion_stage(features)
        return self.project(fused[-1])


@dataclass
class Tipsv2DptNormalEstimatorOutput(ModelOutput):
    r"""
    normals (`torch.FloatTensor` of shape `(batch_size, 3, height, width)`):
        Raw normal map predictions (unnormalized).
    hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        Hidden states of the backbone.
    attentions (`tuple(torch.FloatTensor)`, *optional*):
        Attention weights of the backbone.
    """

    normals: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class Tipsv2DptOutput(ModelOutput):
    r"""
    predicted_depth (`torch.FloatTensor` of shape `(batch_size, height, width)`):
        Soft-bin-expectation depth map at decoder resolution (metres).
    normals (`torch.FloatTensor` of shape `(batch_size, 3, height, width)`):
        Raw normal map predictions (unnormalized).
    logits (`torch.FloatTensor` of shape `(batch_size, num_labels, height, width)`):
        Segmentation logits at decoder resolution.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        Hidden states of the backbone.
    attentions (`tuple(torch.FloatTensor)`, *optional*):
        Attention weights of the backbone.
    """

    predicted_depth: torch.FloatTensor | None = None
    normals: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class Tipsv2DptDecoder(nn.Module):
    """Channel-last linear head: `(B, fusion_hidden_size, H, W)` → `(B, out_channels, H, W)`."""

    def __init__(self, config: Tipsv2DptConfig, out_channels: int):
        super().__init__()
        self.head = nn.Linear(config.fusion_hidden_size, out_channels)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = hidden_state.permute(0, 2, 3, 1)  # (B, H, W, C)
        hidden_state = self.head(hidden_state)  # (B, H, W, out_channels)
        hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()  # (B, out_channels, H, W)
        return hidden_state


@auto_docstring
class Tipsv2DptPreTrainedModel(PreTrainedModel):
    config: Tipsv2DptConfig
    base_model_prefix = "tipsv2_dpt"
    main_input_name = "pixel_values"
    input_modalities = ["image"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module) -> None:
        super()._init_weights(module)
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


@auto_docstring(
    custom_intro="""
    TIPSv2-DPT Model with three independent heads for depth estimation, surface normal estimation,
    and semantic segmentation — running a single shared backbone forward pass.
    """
)
class Tipsv2DptModel(Tipsv2DptPreTrainedModel):
    def __init__(self, config: Tipsv2DptConfig):
        super().__init__(config)
        self.backbone = load_backbone(config)
        self.depth_neck = Tipsv2DptNeck(config)
        self.depth_decoder = Tipsv2DptDecoder(config, out_channels=config.num_depth_bins)
        self.normals_neck = Tipsv2DptNeck(config)
        self.normals_decoder = Tipsv2DptDecoder(config, out_channels=3)
        self.segmentation_neck = Tipsv2DptNeck(config)
        self.segmentation_decoder = Tipsv2DptDecoder(config, out_channels=config.num_labels)
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tipsv2DptOutput:
        outputs = self.backbone.forward_with_filtered_kwargs(pixel_values, **kwargs)
        feature_maps = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_height = height // self.config.backbone_config.patch_size
        patch_width = width // self.config.backbone_config.patch_size

        depth_fused = self.depth_neck(feature_maps, patch_height=patch_height, patch_width=patch_width)
        depth_logits = self.depth_decoder(torch.relu(depth_fused))
        probs = torch.relu(depth_logits) + self.config.min_depth
        probs = probs / probs.sum(dim=1, keepdim=True)
        depth_bins = torch.linspace(
            self.config.min_depth,
            self.config.max_depth,
            self.config.num_depth_bins,
            device=depth_logits.device,
            dtype=depth_logits.dtype,
        )
        predicted_depth = (probs * depth_bins.view(1, -1, 1, 1)).sum(dim=1)

        normals_fused = self.normals_neck(feature_maps, patch_height=patch_height, patch_width=patch_width)
        normals = self.normals_decoder(normals_fused)

        seg_fused = self.segmentation_neck(feature_maps, patch_height=patch_height, patch_width=patch_width)
        seg_logits = self.segmentation_decoder(seg_fused)

        return Tipsv2DptOutput(
            predicted_depth=predicted_depth,
            normals=normals,
            logits=seg_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    TIPSv2-DPT Model with a monocular depth estimation head.
    """
)
class Tipsv2DptForDepthEstimation(Tipsv2DptPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = {"normals_head", "normals_neck", "segmentation_head", "segmentation_neck"}

    def __init__(self, config: Tipsv2DptConfig):
        super().__init__(config)
        self.backbone = load_backbone(config)
        self.neck = Tipsv2DptNeck(config)
        self.decoder = Tipsv2DptDecoder(config, out_channels=config.num_depth_bins)
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DepthEstimatorOutput:
        outputs = self.backbone.forward_with_filtered_kwargs(pixel_values, **kwargs)
        feature_maps = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_height = height // self.config.backbone_config.patch_size
        patch_width = width // self.config.backbone_config.patch_size

        fused = self.neck(feature_maps, patch_height=patch_height, patch_width=patch_width)
        logits = self.decoder(torch.relu(fused))  # (B, num_depth_bins, H', W')

        probs = torch.relu(logits) + self.config.min_depth
        probs = probs / probs.sum(dim=1, keepdim=True)
        depth_bins = torch.linspace(
            self.config.min_depth,
            self.config.max_depth,
            self.config.num_depth_bins,
            device=logits.device,
            dtype=logits.dtype,
        )
        predicted_depth = (probs * depth_bins.view(1, -1, 1, 1)).sum(dim=1)  # (B, H', W')

        return DepthEstimatorOutput(
            predicted_depth=predicted_depth,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    TIPSv2-DPT Model with a surface normal estimation head.
    """
)
class Tipsv2DptForNormalEstimation(Tipsv2DptPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = {"depth_head", "depth_neck", "segmentation_head", "segmentation_neck"}

    def __init__(self, config: Tipsv2DptConfig):
        super().__init__(config)
        self.backbone = load_backbone(config)
        self.neck = Tipsv2DptNeck(config)
        self.decoder = Tipsv2DptDecoder(config, out_channels=3)
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tipsv2DptNormalEstimatorOutput:
        outputs = self.backbone.forward_with_filtered_kwargs(pixel_values, **kwargs)
        feature_maps = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_height = height // self.config.backbone_config.patch_size
        patch_width = width // self.config.backbone_config.patch_size

        fused = self.neck(feature_maps, patch_height=patch_height, patch_width=patch_width)
        normals = self.decoder(fused)  # (B, 3, H', W') — unnormalized

        return Tipsv2DptNormalEstimatorOutput(
            normals=normals,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    TIPSv2-DPT Model with a semantic segmentation head.
    """
)
class Tipsv2DptForSemanticSegmentation(Tipsv2DptPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = {"depth_head", "depth_neck", "normals_head", "normals_neck"}

    def __init__(self, config: Tipsv2DptConfig):
        super().__init__(config)
        self.backbone = load_backbone(config)
        self.neck = Tipsv2DptNeck(config)
        self.decoder = Tipsv2DptDecoder(config, out_channels=config.num_labels)
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
    ) -> SemanticSegmenterOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth segmentation maps for computing the loss. Indices should be in
            `[0, ..., config.num_labels - 1]`. Pixels with index `config.semantic_loss_ignore_index`
            are ignored when computing the loss.
        """
        outputs = self.backbone.forward_with_filtered_kwargs(pixel_values, **kwargs)
        feature_maps = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_height = height // self.config.backbone_config.patch_size
        patch_width = width // self.config.backbone_config.patch_size

        fused = self.neck(feature_maps, patch_height=patch_height, patch_width=patch_width)
        seg_logits = self.decoder(fused)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
            upsampled_logits = nn.functional.interpolate(
                seg_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = loss_fct(upsampled_logits, labels)

        return SemanticSegmenterOutput(
            loss=loss,
            logits=seg_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Tipsv2DptConfig",
    "Tipsv2DptImageProcessor",
    "Tipsv2DptPreTrainedModel",
    "Tipsv2DptModel",
    "Tipsv2DptForDepthEstimation",
    "Tipsv2DptForNormalEstimation",
    "Tipsv2DptForSemanticSegmentation",
]
