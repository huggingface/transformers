# Copyright 2026 Meta Platforms, Inc. and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Sapiens2 License. You may obtain a copy of the License at
#
#     https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...image_processing_backends import TorchvisionBackend
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from ...modeling_outputs import SemanticSegmenterOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple, maybe_autocast
from ..dinov3_vit.configuration_dinov3_vit import DINOv3ViTConfig
from ..dinov3_vit.modeling_dinov3_vit import (
    DINOv3ViTBackbone,
    DINOv3ViTEmbeddings,
    DINOv3ViTEncoder,
    DINOv3ViTLayer,
    DINOv3ViTLayerScale,
    DINOv3ViTModel,
    DINOv3ViTPreTrainedModel,
    augment_patches_center_coordinates,
    eager_attention_forward,
    get_patches_center_coordinates,
    rotate_half,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/sapiens2-pretrain-0.4b")
@strict
class Sapiens2Config(DINOv3ViTConfig):
    r"""
    rope_theta (`float`, *optional*, defaults to 100.0):
        The base period of the RoPE embeddings.
    query_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a bias to the query projection.
    key_bias (`bool`, *optional*, defaults to `False`):
        Whether to add a bias to the key projection.
    value_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a bias to the value projection.
    proj_bias (`bool`, *optional*, defaults to `True`):
        Whether to add a bias to the output projection.
    layerscale_value (`float`, *optional*, defaults to 1.0):
        Initial value to use for layer scale.
    use_gated_mlp (`bool`, *optional*, defaults to `False`):
        Whether to use the SwiGLU feedforward neural network.
    num_register_tokens (`int`, *optional*, defaults to 0):
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
    apply_layernorm (`bool`, *optional*, defaults to `True`):
        Whether to apply layer normalization to the feature maps when used as backbone.
    reshape_hidden_states (`bool`, *optional*, defaults to `True`):
        Whether to reshape the hidden states to spatial dimensions when used as backbone.
    use_mask_token (`bool`, *optional*, defaults to `False`):
        Whether to use a mask token in the embeddings (needed for masked image modeling pretraining).
    use_qk_norm (`bool`, *optional*, defaults to `True`):
        Whether to apply RMSNorm to queries and keys before RoPE in attention layers.
    num_key_value_heads (`int`, *optional*):
        Number of key/value heads for GQA layers. Defaults to `num_attention_heads // 2`.
        Set to `None` to disable GQA and use full multi-head attention everywhere.
    layer_types (`list[str]`, *optional*):
        Per-layer attention type, one of `"full_attention"` or `"grouped_query_attention"`. Computed automatically
        from `num_first_full_attention_layers` and `num_last_full_attention_layers` if not provided.
    num_first_full_attention_layers (`int`, *optional*, defaults to 8):
        Number of initial transformer layers that use full multi-head attention.
        Layers at or after this index switch to GQA with `num_key_value_heads`.
    num_last_full_attention_layers (`int`, *optional*, defaults to 8):
        Number of final transformer layers that use full multi-head attention.
        Layers before `num_hidden_layers - num_last_full_attention_layers` use GQA with `num_key_value_heads`.
    pos_embed_dtype (`str`, *optional*, defaults to `"bfloat16"`):
        Dtype used for positional embedding computations (RoPE angles, cos/sin).
    semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
        Label index ignored when computing the segmentation loss.
    """

    model_type = "sapiens2"

    # TODO(guarin): This is needed to load the original checkpoints but makes unit tests fail.
    # transformers_weights = "sapiens2_0.4b_pretrain.safetensors"

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    use_mask_token: bool = False
    use_gated_mlp: bool = True
    hidden_act: str = "silu"
    layer_norm_eps: float = 1e-6
    num_register_tokens: int = 8
    key_bias: bool = True
    use_qk_norm: bool = True
    num_key_value_heads: int | None = None
    layer_types: list[str] | None = None
    num_first_full_attention_layers: int = 8
    num_last_full_attention_layers: int = 8
    pos_embed_dtype: str = "bfloat16"
    semantic_loss_ignore_index: int = 255

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads // 2
        if self.layer_types is None:
            self.layer_types = [
                "full_attention"
                if (
                    i < self.num_first_full_attention_layers
                    or i >= self.num_hidden_layers - self.num_last_full_attention_layers
                )
                else "grouped_query_attention"
                for i in range(self.num_hidden_layers)
            ]
        super().__post_init__(**kwargs)


class Sapiens2Embeddings(DINOv3ViTEmbeddings):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        if not config.use_mask_token:
            del self.mask_token

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> torch.Tensor:
        if bool_masked_pos is not None and not self.config.use_mask_token:
            raise ValueError("bool_masked_pos requires use_mask_token=True in the config")
        return super().forward(pixel_values, bool_masked_pos)


class Sapiens2RopePositionEmbedding(nn.Module):
    periods: torch.Tensor

    def __init__(self, config: Sapiens2Config):
        super().__init__()

        self.patch_size = config.patch_size
        self.pos_embed_shift = config.pos_embed_shift
        self.pos_embed_jitter = config.pos_embed_jitter
        self.pos_embed_rescale = config.pos_embed_rescale
        self.base = config.rope_theta
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.pos_embed_dtype = getattr(torch, config.pos_embed_dtype)

        periods = self.base ** (
            2 * torch.arange(self.head_dim // 4, dtype=self.pos_embed_dtype) / (self.head_dim // 2)
        )
        self.register_buffer("periods", periods, persistent=True)  # persistent=True to match original checkpoints

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = pixel_values.shape
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        device = pixel_values.device
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

        with maybe_autocast(device_type=device_type, enabled=False):
            patch_coords = get_patches_center_coordinates(
                num_patches_h, num_patches_w, dtype=self.pos_embed_dtype, device=device
            )
            if self.training:
                patch_coords = augment_patches_center_coordinates(
                    patch_coords,
                    shift=self.pos_embed_shift,
                    jitter=self.pos_embed_jitter,
                    rescale=self.pos_embed_rescale,
                )

            # (height * width, 2, head_dim / 4) -> (height * width, head_dim / 2) -> (height * width, head_dim)
            angles = 2 * math.pi * patch_coords[:, :, None] / self.periods[None, None, :].to(self.pos_embed_dtype)
            angles = angles.flatten(1, 2)
            angles = angles.tile(2)

            cos = torch.cos(angles)
            sin = torch.sin(angles)

        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors, but only to the patch tokens,
    ignoring the prefix tokens (cls token and register tokens).

    Casts all q/k tokens to the rope dtype before applying the rotation and casts back afterwards.
    This matches the original model behavior where all tokens are cast to rope_dtype before the
    prefix/patch split, even though RoPE is only applied to patch tokens.
    """
    q_dtype, k_dtype = q.dtype, k.dtype
    rope_dtype = cos.dtype
    q = q.to(rope_dtype)
    k = k.to(rope_dtype)

    num_tokens = q.shape[-2]
    num_patches = sin.shape[-2]
    num_prefix_tokens = num_tokens - num_patches

    q_prefix_tokens, q_patches = q.split((num_prefix_tokens, num_patches), dim=-2)
    k_prefix_tokens, k_patches = k.split((num_prefix_tokens, num_patches), dim=-2)

    q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)

    q = torch.cat((q_prefix_tokens, q_patches), dim=-2)
    k = torch.cat((k_prefix_tokens, k_patches), dim=-2)

    q = q.to(q_dtype)
    k = k.to(k_dtype)

    return q, k


class Sapiens2Attention(nn.Module):
    def __init__(self, config: Sapiens2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = False
        self.scaling = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.use_qk_norm = config.use_qk_norm

        self.num_kv_heads = (
            self.num_heads if config.layer_types[layer_idx] == "full_attention" else config.num_key_value_heads
        )
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.query_bias)
        self.k_proj = nn.Linear(self.embed_dim, kv_dim, bias=config.key_bias)
        self.v_proj = nn.Linear(self.embed_dim, kv_dim, bias=config.value_bias)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.proj_bias)

        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=config.layer_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, patches, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, patches, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.num_kv_heads != self.num_heads:
            factor = self.num_heads // self.num_kv_heads
            key_states = key_states.repeat_interleave(factor, dim=1)
            value_states = value_states.repeat_interleave(factor, dim=1)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, patches, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Sapiens2LayerScale(DINOv3ViTLayerScale):
    pass


class Sapiens2Layer(DINOv3ViTLayer):
    def __init__(self, config: Sapiens2Config, layer_idx: int):
        super().__init__(config)
        self.attention = Sapiens2Attention(config, layer_idx=layer_idx)
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_scale2 = nn.Identity()


class Sapiens2ConvTransposeLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
        activation: str = "silu",
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = ACT2FN[activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(hidden_states)))


class Sapiens2ConvLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 1, bias: bool = True, activation: str = "silu"
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = ACT2FN[activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(hidden_states)))


class Sapiens2SegmentationHead(nn.Module):
    def __init__(self, config: Sapiens2Config):
        super().__init__()
        deconv_channels = (config.hidden_size, 512, 256, 128, 64)
        self.deconv_layers = nn.ModuleList(
            Sapiens2ConvTransposeLayer(in_ch, out_ch)
            for in_ch, out_ch in zip(deconv_channels[:-1], deconv_channels[1:])
        )
        self.conv_layers = nn.ModuleList(Sapiens2ConvLayer(64, 64) for _ in range(2))
        self.classifier = nn.Conv2d(64, config.num_labels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.deconv_layers:
            hidden_states = layer(hidden_states)
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)
        return self.classifier(hidden_states)


class Sapiens2PreTrainedModel(DINOv3ViTPreTrainedModel):
    base_model_prefix = "sapiens2"

    @torch.no_grad()
    def _init_weights(self, module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.trunc_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.ConvTranspose2d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        elif isinstance(module, nn.RMSNorm):
            init.ones_(module.weight)
        elif isinstance(module, Sapiens2Embeddings):
            init.trunc_normal_(module.cls_token, mean=0.0, std=self.config.initializer_range)
            if module.config.num_register_tokens > 0:
                init.trunc_normal_(module.register_tokens, mean=0.0, std=self.config.initializer_range)
            if module.config.use_mask_token:
                init.zeros_(module.mask_token)
        elif isinstance(module, Sapiens2LayerScale):
            init.constant_(module.lambda1, self.config.layerscale_value)
        elif isinstance(module, Sapiens2RopePositionEmbedding):
            periods = module.base ** (
                2 * torch.arange(module.head_dim // 4, dtype=module.pos_embed_dtype) / (module.head_dim // 2)
            )
            init.copy_(module.periods, periods)


class Sapiens2Encoder(DINOv3ViTEncoder):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.layer = nn.ModuleList([Sapiens2Layer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.post_init()


class Sapiens2Model(DINOv3ViTModel):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()


class Sapiens2Backbone(DINOv3ViTBackbone):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()


@auto_docstring(checkpoint="facebook/sapiens2-seg-0.4b")
class Sapiens2ForSemanticSegmentation(Sapiens2PreTrainedModel):
    def __init__(self, config: Sapiens2Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.sapiens2 = Sapiens2Model(config)
        self.decode_head = Sapiens2SegmentationHead(config)
        self.post_init()

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
            Ground truth semantic segmentation maps for computing the loss.
            Indices should be in `[0, ..., config.num_labels - 1]`.
            If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).
        """
        if labels is not None and self.config.num_labels == 1:
            raise ValueError("The number of labels should be greater than one")

        outputs = self.sapiens2(pixel_values, **kwargs)

        batch_size, _, height, width = pixel_values.shape
        patch_height = height // self.config.patch_size
        patch_width = width // self.config.patch_size

        patch_tokens = outputs.last_hidden_state[:, 1 + self.config.num_register_tokens :]
        feature_map = patch_tokens.transpose(1, 2).reshape(batch_size, -1, patch_height, patch_width)

        logits = self.decode_head(feature_map)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, ignore_index=self.config.semantic_loss_ignore_index)

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Sapiens2ImageProcessor(TorchvisionBackend):
    # Note: original Sapiens2 uses cv2.INTER_AREA for downsampling and cv2.INTER_CUBIC for upsampling
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 1024, "width": 768}
    do_resize = True
    do_rescale = True
    do_normalize = True

    def post_process_semantic_segmentation(
        self, outputs: SemanticSegmenterOutput, target_sizes: list[tuple] | None = None
    ) -> list[torch.Tensor]:
        """
        Converts the output of [`Sapiens2ForSemanticSegmentation`] into semantic segmentation maps.

        Args:
            outputs (`SemanticSegmenterOutput`):
                Raw outputs of the model.
            target_sizes (`list[tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size `(height, width)` of each prediction.
                If unset, predictions will not be resized.

        Returns:
            `list[torch.Tensor]` of length `batch_size`, where each item is a semantic segmentation map of
            shape `(height, width)` corresponding to the target size (if `target_sizes` is specified).
            Each entry corresponds to a semantic class id.
        """
        logits = outputs.logits

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(len(logits)):
                resized_logits = F.interpolate(
                    logits[idx].unsqueeze(0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                )
                semantic_segmentation.append(resized_logits[0].argmax(dim=0))
        else:
            semantic_segmentation = list(logits.argmax(dim=1))

        return semantic_segmentation


__all__ = [
    "Sapiens2Config",
    "Sapiens2Model",
    "Sapiens2PreTrainedModel",
    "Sapiens2Backbone",
    "Sapiens2ForSemanticSegmentation",
    "Sapiens2ImageProcessor",
]
