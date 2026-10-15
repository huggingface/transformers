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
"""PyTorch Cellpose model with a DINOv3 backbone."""

import math

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple
from ..cellpose.modeling_cellpose import CellposeInstanceSegmentationOutput
from ..dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTEmbeddings,
    DINOv3ViTEncoder,
    DINOv3ViTLayerScale,
    DINOv3ViTPreTrainedModel,
    DINOv3ViTRopePositionEmbedding,
    augment_patches_center_coordinates,
    get_patches_center_coordinates,
)


@auto_docstring(checkpoint="ArthurZ/cellpose-dino")
@strict
class CellposeDinoConfig(PreTrainedConfig):
    r"""
    patch_stride (`int`, *optional*, defaults to 8):
        Stride of the patch embedding convolution. Cellpose uses a stride smaller than `patch_size` (with
        `patch_stride // 2` padding) to predict at a finer resolution.
    rope_theta (`float`, *optional*, defaults to 100.0):
        The base period of the RoPE embeddings.
    query_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a bias to the query projection.
    key_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a bias to the key projection.
    value_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a bias to the value projection.
    proj_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a bias to the output projection.
    layerscale_value (`float`, *optional*, defaults to 1.0):
        Initial value to use for layer scale.
    use_gated_mlp (`bool`, *optional*, defaults to `False`):
        Whether to use the SwiGLU feedforward neural network.
    num_register_tokens (`int`, *optional*, defaults to 4):
        The number of register tokens.
    pos_embed_shift (`float`, *optional*):
        Amount to randomly shift position embedding coordinates in [-shift, shift],
        applied only in training mode if not `None`.
    pos_embed_jitter (`float`, *optional*):
        Amount to randomly jitter position embedding coordinates in log-uniform value in [1/jitter, jitter],
        applied only in training mode if not `None`.
    pos_embed_rescale (`float`, *optional*, defaults to 2.0):
        Amount to randomly rescale position embedding coordinates in log-uniform value in [1/rescale, rescale],
        applied only in training mode if not `None`.
    num_outputs (`int`, *optional*, defaults to 3):
        Number of predicted maps per pixel: vertical flow, horizontal flow and cell probability.

    Example:

    ```python
    >>> from transformers import CellposeDinoConfig, CellposeDinoForInstanceSegmentation

    >>> configuration = CellposeDinoConfig()
    >>> model = CellposeDinoForInstanceSegmentation(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "cellpose_dino"

    image_size: int = 384
    patch_size: int = 16
    patch_stride: int = 8
    num_channels: int = 3
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    hidden_act: str = "gelu"
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    rope_theta: float = 100.0
    query_bias: bool = True
    key_bias: bool = True
    value_bias: bool = True
    proj_bias: bool = True
    mlp_bias: bool = True
    layerscale_value: float = 1.0
    drop_path_rate: float | int = 0.0
    use_gated_mlp: bool = False
    num_register_tokens: int = 4
    pos_embed_shift: float | None = None
    pos_embed_jitter: float | None = None
    pos_embed_rescale: float | None = 2.0
    num_outputs: int = 3


class CellposeDinoEmbeddings(DINOv3ViTEmbeddings):
    """
    Construct the CLS token, register tokens and patch embeddings, with an overlapping patch convolution.
    """

    def __init__(self, config: CellposeDinoConfig):
        nn.Module.__init__(self)
        self.config = config
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.register_tokens = nn.Parameter(torch.empty(1, config.num_register_tokens, config.hidden_size))
        padding = config.patch_stride // 2 if config.patch_stride < config.patch_size else 0
        self.patch_embeddings = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_stride,
            padding=padding,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeddings = self.patch_embeddings(pixel_values.to(dtype=self.patch_embeddings.weight.dtype))
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        return torch.cat([cls_token, register_tokens, patch_embeddings], dim=1)


class CellposeDinoRopePositionEmbedding(DINOv3ViTRopePositionEmbedding):
    periods: torch.Tensor

    def __init__(self, config: CellposeDinoConfig):
        nn.Module.__init__(self)
        self.config = config
        self.base = config.rope_theta
        self.head_dim = config.hidden_size // config.num_attention_heads
        # Persistent: the checkpoints store bf16-rounded periods, recomputing them in fp32 drifts the rope by ~1e-2
        periods = self.base ** (2 * torch.arange(self.head_dim // 4, dtype=torch.float32) / (self.head_dim // 2))
        self.periods = nn.Buffer(periods)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = pixel_values.shape
        num_patches_h = height // self.config.patch_stride
        num_patches_w = width // self.config.patch_stride

        patch_coords = get_patches_center_coordinates(
            num_patches_h, num_patches_w, dtype=torch.float32, device=pixel_values.device
        )
        if self.training:
            patch_coords = augment_patches_center_coordinates(
                patch_coords,
                shift=self.config.pos_embed_shift,
                jitter=self.config.pos_embed_jitter,
                rescale=self.config.pos_embed_rescale,
            )

        # (height * width, 2, head_dim / 4) -> (height * width, head_dim / 2) -> (height * width, head_dim)
        angles = 2 * math.pi * patch_coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2)
        angles = angles.tile(2)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        dtype = pixel_values.dtype
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


@auto_docstring
class CellposeDinoPreTrainedModel(DINOv3ViTPreTrainedModel):
    config: CellposeDinoConfig
    base_model_prefix = "encoder"
    _keys_to_ignore_on_load_unexpected = [r"W2", r"mask_token", r"bias_mask"]

    @torch.no_grad()
    def _init_weights(self, module) -> None:
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.trunc_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, CellposeDinoEmbeddings):
            init.trunc_normal_(module.cls_token, mean=0.0, std=self.config.initializer_range)
            if module.config.num_register_tokens > 0:
                init.trunc_normal_(module.register_tokens, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, CellposeDinoLayerScale):
            init.constant_(module.lambda1, self.config.layerscale_value)
        elif isinstance(module, CellposeDinoRopePositionEmbedding):
            periods = module.base ** (
                2 * torch.arange(module.head_dim // 4, dtype=torch.float32) / (module.head_dim // 2)
            )
            init.copy_(module.periods, periods)


class CellposeDinoLayerScale(DINOv3ViTLayerScale):
    pass


class CellposeDinoEncoder(DINOv3ViTEncoder):
    pass


@auto_docstring
class CellposeDinoModel(CellposeDinoPreTrainedModel):
    def __init__(self, config: CellposeDinoConfig):
        super().__init__(config)
        self.embeddings = CellposeDinoEmbeddings(config)
        self.rope_embeddings = CellposeDinoRopePositionEmbedding(config)
        self.model = CellposeDinoEncoder(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    @can_return_tuple
    @auto_docstring
    def forward(self, pixel_values: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> BaseModelOutputWithPooling:
        pixel_values = pixel_values.to(self.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.embeddings(pixel_values)
        position_embeddings = self.rope_embeddings(pixel_values)

        encoder_outputs: BaseModelOutput = self.model(hidden_states, position_embeddings=position_embeddings, **kwargs)
        sequence_output = self.norm(encoder_outputs.last_hidden_state)
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=sequence_output[:, 0, :],
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CellposeDinoInstanceSegmentationOutput(CellposeInstanceSegmentationOutput):
    pass


@auto_docstring(
    custom_intro="""
    Cellpose with a DINOv3 backbone: predicts per-pixel flows and cell probabilities, turned into instance masks by
    [`CellposeImageProcessor.post_process_instance_segmentation`].
    """
)
class CellposeDinoForInstanceSegmentation(CellposeDinoPreTrainedModel):
    def __init__(self, config: CellposeDinoConfig):
        super().__init__(config)
        self.encoder = CellposeDinoModel(config)
        self.readout = nn.Linear(config.hidden_size, config.num_outputs * config.patch_stride**2)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self, pixel_values: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> CellposeDinoInstanceSegmentationOutput:
        outputs = self.encoder(pixel_values, **kwargs)
        batch_size, _, height, width = pixel_values.shape
        stride = self.config.patch_stride
        patch_tokens = outputs.last_hidden_state[:, 1 + self.config.num_register_tokens :]
        logits = self.readout(patch_tokens).reshape(batch_size, height // stride, width // stride, -1)
        # The original model unpatchifies with a `conv_transpose2d` on an identity kernel, i.e. a pixel shuffle
        logits = F.pixel_shuffle(logits.permute(0, 3, 1, 2), stride)
        return CellposeDinoInstanceSegmentationOutput(
            logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


__all__ = [
    "CellposeDinoConfig",
    "CellposeDinoPreTrainedModel",
    "CellposeDinoModel",
    "CellposeDinoForInstanceSegmentation",
]
