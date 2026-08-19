# Copyright 2026 H Company and the HuggingFace Inc. team. All rights reserved.
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

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from ... import initialization as init
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...masking_utils import create_bidirectional_mask, sliding_window_bidirectional_overlay
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, MaskedLMOutput
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    logging,
    torch_compilable_check,
)
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..gemma4.modeling_gemma4 import Gemma4RMSNorm
from ..gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from ..llama.modeling_llama import eager_attention_forward, repeat_kv
from ..nemotron.modeling_nemotron import NemotronMLP
from ..siglip2.image_processing_siglip2 import convert_image_to_patches
from .configuration_neomme import NeoMMEConfig


logger = logging.get_logger(__name__)


def _validate_image_dimensions(height: int, width: int) -> None:
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError(f"height must be a positive integer, got {height!r}.")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError(f"width must be a positive integer, got {width!r}.")


def get_resize_output_size(
    height: int, width: int, max_side: int | None, max_pixels: int | None, min_pixels: int | None
) -> tuple[int, int]:
    """Compute integer height and width from the configured image size targets."""
    _validate_image_dimensions(height, width)
    scale = (min_pixels / (height * width)) ** 0.5 if min_pixels is not None and height * width < min_pixels else 1.0
    if max_side is not None:
        scale = min(scale, max_side / max(height, width))
    if max_pixels is not None:
        scale = min(scale, (max_pixels / (height * width)) ** 0.5)
    if scale == 1.0:
        return height, width
    return max(1, round(height * scale)), max(1, round(width * scale))


class NeoMMEImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to `self.patch_size`):
        Side, in pixels, of one patch token. The image is padded to a whole multiple of it.
    max_side (`int`, *optional*):
        Longest-side cap in pixels. Unset means no longest-side resize.
    max_pixels (`int`, *optional*):
        Maximum pixel-area target before integer dimension rounding. Unset means no area target.
    min_pixels (`int`, *optional*):
        Minimum pixel-area target before integer dimension rounding; may upscale the image. Maximum targets take
        precedence when both bounds apply.
    """

    patch_size: int
    max_side: int | None
    max_pixels: int | None
    min_pixels: int | None


@auto_docstring
class NeoMMEImageProcessor(TorchvisionBackend):
    r"""
    Constructs an image processor for NeoMME.

    The processor converts each image to RGB and splits it into row-major patches. It preserves the original size by
    default, but it can resize images when `max_side`, `max_pixels`, or `min_pixels` is set. It pads the bottom and
    right edges to complete the final patch. A batch returns one concatenated patch tensor and one grid height and
    width pair per image.
    """

    valid_kwargs = NeoMMEImageProcessorKwargs

    resample = PILImageResampling.BILINEAR
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]

    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True

    patch_size = 32
    # Checkpoints may set resize limits; unset limits preserve native resolution.
    max_side = None
    max_pixels = None
    min_pixels = None

    model_input_names = ["pixel_values", "image_grid_hw"]

    def __init__(self, **kwargs: Unpack[NeoMMEImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[NeoMMEImageProcessorKwargs]) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`] with `pixel_values` of shape `(total_patches, 3 * patch_size ** 2)` and
            `image_grid_hw` of shape `(batch_size, 2)`. `pixel_values` concatenates patches from every image in the
            batch.
        """
        return super().preprocess(images, **kwargs)

    def _validate_preprocess_kwargs(self, **kwargs) -> tuple:
        # Generic resize validation requires a fixed `size`; NeoMME computes one per image.
        kwargs.pop("do_resize", None)
        self._validate_size_settings(
            kwargs.get("patch_size", self.patch_size),
            kwargs.get("max_side", self.max_side),
            kwargs.get("max_pixels", self.max_pixels),
            kwargs.get("min_pixels", self.min_pixels),
        )
        return super()._validate_preprocess_kwargs(**kwargs)

    @staticmethod
    def _validate_size_settings(
        patch_size: int, max_side: int | None, max_pixels: int | None, min_pixels: int | None
    ) -> None:
        if not isinstance(patch_size, int) or isinstance(patch_size, bool) or patch_size <= 0:
            raise ValueError(f"patch_size must be a positive integer, got {patch_size!r}.")
        for name, value in (("max_side", max_side), ("max_pixels", max_pixels), ("min_pixels", min_pixels)):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None, got {value!r}.")

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        patch_size: int,
        max_side: int | None,
        max_pixels: int | None,
        min_pixels: int | None,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        unsupported = sorted(
            name
            for name in ("size", "crop_size", "do_center_crop", "do_pad", "pad_size", "image_seq_length")
            if kwargs.get(name) not in (None, False)
        )
        if unsupported:
            raise ValueError(f"NeoMMEImageProcessor does not implement these image kwargs: {unsupported}")

        pixel_values: list[torch.Tensor] = []
        image_grid_hw: list[tuple[int, int]] = []

        # Process images separately because each produces its own patch grid.
        for image in images:
            if do_resize:
                image = self._resize_to_budget(image, max_side, max_pixels, min_pixels, resample)
            # Pad to a whole patch grid before rescaling, so padded pixels become -1 exactly like the
            # black canvas the reference implementation pastes onto.
            image, grid_height, grid_width = self._pad_to_patch_grid(image, patch_size)
            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)
            pixel_values.append(convert_image_to_patches(image, patch_size))  # (patches, patch_dim)
            image_grid_hw.append((grid_height, grid_width))

        return BatchFeature(
            data={
                "pixel_values": torch.cat(pixel_values, dim=0),  # (total_patches, patch_dim)
                "image_grid_hw": torch.tensor(image_grid_hw, dtype=torch.int64),  # (images, 2)
            },
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> int:
        """
        Return the number of image patches for one image, excluding row markers.

        The method applies the effective resize settings, then rounds each image dimension up to a whole patch.
        Values in `images_kwargs` override the processor settings.
        """
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        max_side = images_kwargs.get("max_side", self.max_side)
        max_pixels = images_kwargs.get("max_pixels", self.max_pixels)
        min_pixels = images_kwargs.get("min_pixels", self.min_pixels)
        _validate_image_dimensions(height, width)
        self._validate_size_settings(patch_size, max_side, max_pixels, min_pixels)
        if images_kwargs.get("do_resize", self.do_resize):
            height, width = get_resize_output_size(
                height,
                width,
                max_side,
                max_pixels,
                min_pixels,
            )
        return -(-height // patch_size) * (-(-width // patch_size))

    def _resize_to_budget(
        self,
        image: "torch.Tensor",
        max_side: int | None,
        max_pixels: int | None,
        min_pixels: int | None,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
    ) -> "torch.Tensor":
        height, width = image.shape[-2], image.shape[-1]
        resized_height, resized_width = get_resize_output_size(height, width, max_side, max_pixels, min_pixels)
        if (resized_height, resized_width) == (height, width):
            return image
        size = SizeDict(height=resized_height, width=resized_width)
        # Explicit antialiasing keeps the Torchvision and PIL resize paths aligned.
        return self.resize(image=image, size=size, resample=resample, antialias=True)

    def _pad_to_patch_grid(self, image: "torch.Tensor", patch_size: int) -> tuple["torch.Tensor", int, int]:
        height, width = image.shape[-2], image.shape[-1]
        grid_height, grid_width = -(-height // patch_size), -(-width // patch_size)
        pad_height = grid_height * patch_size - height
        pad_width = grid_width * patch_size - width
        if pad_height or pad_width:
            image = tvF.pad(image, [0, 0, pad_width, pad_height], fill=0)
        # image: (num_channels, grid_height * patch_size, grid_width * patch_size)
        return image, grid_height, grid_width


class NeoMMERMSNorm(Gemma4RMSNorm):
    pass


class NeoMMEEmbeddings(nn.Module):
    """Factorized (ALBERT-style) token embeddings: `vocab_size -> embedding_rank -> hidden_size`."""

    def __init__(self, config: NeoMMEConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_rank)
        self.embedding_projection = nn.Linear(config.embedding_rank, config.hidden_size, bias=False)

    def forward(
        self, input_ids: torch.LongTensor | None = None, inputs_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        return self.embedding_projection(inputs_embeds)

    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states through the tied factorized embedding for MLM logits."""
        projected = hidden_states @ self.embedding_projection.weight  # (batch, seq, embedding_rank)
        return projected @ self.word_embeddings.weight.t()  # (batch, seq, vocab_size)


class NeoMMEPatchEmbeddings(nn.Module):
    """Patch stem that maps flattened image patches to hidden size."""

    def __init__(self, config: NeoMMEConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.patch_dim)
        self.up_proj = nn.Linear(config.patch_dim, config.hidden_size * 2, bias=False)
        self.act_fn = nn.GELU()
        self.down_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.up_proj(self.norm(pixel_values))
        return self.down_proj(self.act_fn(hidden_states))


class NeoMMERotaryEmbedding(nn.Module):
    """Two-axis interleaved M-RoPE with per-layer-type frequency spectra."""

    def __init__(self, config: NeoMMEConfig, device=None):
        super().__init__()
        self.config = config
        self.layer_types = sorted(set(config.layer_types))
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.rope_init_fns: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {}
        self.rope_type: dict[str, str] = {}
        for layer_type in self.layer_types:
            rope_type = config.rope_parameters[layer_type]["rope_type"]
            self.rope_type[layer_type] = rope_type
            self.rope_init_fns[layer_type] = (
                self.compute_default_rope_parameters if rope_type == "default" else ROPE_INIT_FUNCTIONS[rope_type]
            )
            inv_freq, attention_scaling = self.rope_init_fns[layer_type](config, device=device, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            # `dynamic_rope_update` restores the unscaled spectrum from this copy when a sequence shrinks
            # back inside the original context, so it must survive the buffer being overwritten.
            self.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: NeoMMEConfig,
        device: torch.device | None = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Default inverse frequencies for a layer type."""
        partial_rotary_factor = config.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)
        rotary_dim = int(config.head_dim * partial_rotary_factor)
        theta = config.rope_parameters[layer_type]["rope_theta"]
        inv_freq = theta ** -(torch.arange(0, rotary_dim, 2, dtype=torch.float, device=device) / rotary_dim)
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.LongTensor, layer_type: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build cos/sin from two-axis `position_ids` of shape `(2, batch, seq_len)`."""
        input_shape = hidden_states.shape[:2]
        if position_ids.shape == input_shape:
            position_ids = position_ids.unsqueeze(0).expand(2, -1, -1)
        elif position_ids.shape != (2, *input_shape):
            raise ValueError(
                f"position_ids must have shape {tuple(input_shape)} or {(2, *input_shape)}, "
                f"got {tuple(position_ids.shape)}."
            )
        inv_freq = getattr(self, f"{layer_type}_inv_freq")  # (rotary_dim // 2,)
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        row_angles = position_ids[0].float().unsqueeze(-1) * inv_freq[0::2]  # (batch, seq, rotary_dim // 4)
        column_angles = position_ids[1].float().unsqueeze(-1) * inv_freq[1::2]  # (batch, seq, rotary_dim // 4)

        angles = torch.stack([row_angles, column_angles], dim=-1).flatten(-2)  # (batch, seq, rotary_dim // 2)
        cos = (angles.cos() * attention_scaling).to(hidden_states.dtype)
        sin = (angles.sin() * attention_scaling).to(hidden_states.dtype)
        return torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)


class NeoMMEAttention(nn.Module):
    """Bidirectional grouped-query attention with QK-norm, M-RoPE, and a sigmoid output gate.

    QK-norm runs before rotary embedding, value embeddings are added after rotation, and
    exclusive self-attention is applied before the output gate.
    """

    def __init__(self, config: NeoMMEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_norm = NeoMMERMSNorm(config.head_dim, config.norm_eps, with_scale=False)
        self.k_norm = NeoMMERMSNorm(config.head_dim, config.norm_eps, with_scale=False)

        # `config.layer_window_sizes` is a HALF-width (`abs(i - j) <= window`). The flash-attention path
        # builds an inclusive symmetric band of `sliding_window - 1` per side, hence the `+ 1`.
        window = config.layer_window_sizes[layer_idx]
        self.sliding_window = None if window is None else window + 1

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        # Fused K/V projection: rows `[:num_key_value_heads * head_dim]` are K, the rest are V.
        self.kv_proj = nn.Linear(config.hidden_size, 2 * config.num_key_value_heads * config.head_dim, bias=False)
        self.output_gate = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)
        # Exclusive Self-Attention: zero-init, so `tanh(alpha) == 0` makes it an exact no-op at step 0.
        self.alpha = nn.Parameter(torch.zeros(config.num_attention_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        value_embeds: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]  # (batch, seq)

        # query_states: (batch, seq, heads, head_dim)
        query_states = self.q_proj(hidden_states).view(*input_shape, self.num_attention_heads, self.head_dim)
        query_states = self.q_norm(query_states)

        # key_states and value_states: (batch, seq, 2, kv_heads, head_dim)
        key_states, value_states = (
            self.kv_proj(hidden_states).view(*input_shape, 2, self.num_key_value_heads, self.head_dim).unbind(-3)
        )
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)
        if value_embeds is not None:
            value_states = value_states + value_embeds.view(*input_shape, self.num_key_value_heads, self.head_dim)

        query_states = query_states.transpose(1, 2)  # (batch, heads, seq, head_dim)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        # attn_output is heads-last: (batch, seq, heads, head_dim)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = self._exclusive_self_attention(attn_output, value_states)
        attn_output = attn_output.reshape(*input_shape, -1)  # (batch, seq, heads * head_dim)
        gated_output = attn_output * torch.sigmoid(self.output_gate(hidden_states))
        return self.o_proj(gated_output), attn_weights  # (batch, seq, hidden_size)

    def _exclusive_self_attention(self, attn_output: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        """Exclusive self-attention correction along the value direction."""

        value_states = repeat_kv(value_states, self.num_key_value_groups)  # (batch, heads, seq, head_dim)
        value_states = value_states.transpose(1, 2)  # (batch, seq, heads, head_dim)
        value_unit = F.normalize(value_states.float(), dim=-1).to(attn_output.dtype)

        projection = (attn_output * value_unit).sum(-1, keepdim=True)  # (batch, seq, heads, 1)
        scale = torch.tanh(self.alpha).to(attn_output.dtype).view(1, 1, self.num_attention_heads, 1)
        return attn_output - (scale * projection) * value_unit


class NeoMMEMLP(NemotronMLP):
    pass


class NeoMMEEncoderLayer(GradientCheckpointingLayer):
    """Pre-norm encoder layer with initial-state mixing and muP depth scaling."""

    def __init__(self, config: NeoMMEConfig, layer_idx: int):
        super().__init__()
        self.self_attn = NeoMMEAttention(config, layer_idx)
        self.mlp = NeoMMEMLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.input_layernorm = NeoMMERMSNorm(config.hidden_size, config.norm_eps, with_scale=False)
        self.post_attention_layernorm = NeoMMERMSNorm(config.hidden_size, config.norm_eps, with_scale=False)
        self.residual_scale = config.residual_scale
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        initial_hidden_states: torch.Tensor,
        value_embeds: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Keep gradient-carrying inputs positional for reentrant checkpointing.
        mixed_states = self.lambdas[0] * hidden_states + self.lambdas[1] * initial_hidden_states
        normed_states = self.input_layernorm(mixed_states)
        attn_output, _ = self.self_attn(
            normed_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            value_embeds=value_embeds,
            **kwargs,
        )
        hidden_states = hidden_states + self.residual_scale * attn_output
        normed_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_states)
        return hidden_states + self.residual_scale * mlp_output


@auto_docstring
class NeoMMEPreTrainedModel(PreTrainedModel):
    config: NeoMMEConfig
    base_model_prefix = "model"
    _input_embed_layer = "word_embeddings"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["NeoMMEEncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    _can_record_outputs = {
        "hidden_states": NeoMMEEncoderLayer,
        "attentions": NeoMMEAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        # `apply` visits children before parents, so the NeoMME-specific parent initialization below runs last.
        super()._init_weights(module)

        if isinstance(module, NeoMMEEmbeddings):
            # The factorized table is scaled by its rank, not `initializer_range`, so the tied decode
            # logits stay O(1) at init (otherwise the initial cross-entropy is ~250 instead of ln V).
            init.normal_(module.word_embeddings.weight, mean=0.0, std=self.config.embedding_rank**-0.5)
        elif isinstance(module, NeoMMEAttention):
            init.zeros_(module.o_proj.weight)  # residual branch starts as an exact no-op
            init.zeros_(module.alpha)
        elif isinstance(module, NeoMMEMLP):
            init.zeros_(module.down_proj.weight)
        elif isinstance(module, NeoMMEEncoderLayer):
            init.copy_(module.lambdas, torch.tensor([1.0, 0.0]))
        elif isinstance(module, NeoMMEModel) and module.value_embeddings is not None:
            init.zeros_(module.value_embeddings.weight)
        elif isinstance(module, NeoMMERotaryEmbedding):
            for layer_type in module.layer_types:
                inv_freq, _ = module.rope_init_fns[layer_type](module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), inv_freq)

    def _resize_token_embeddings(
        self, new_num_tokens: int, pad_to_multiple_of: int | None = None, mean_resizing: bool = True
    ) -> nn.Embedding:
        """Resize word and value embedding tables together."""
        word_embeddings = super()._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        backbone = getattr(self, self.base_model_prefix, self)
        if getattr(backbone, "value_embeddings", None) is None:
            return word_embeddings

        resized = self._get_resized_embeddings(
            backbone.value_embeddings, word_embeddings.weight.shape[0], mean_resizing=mean_resizing
        )
        backbone.value_embeddings = nn.Embedding(resized.num_embeddings, resized.embedding_dim)
        backbone.value_embeddings.weight = resized.weight
        return word_embeddings


@auto_docstring(
    custom_intro="""
    The bare NeoMME model. It encodes text tokens and image patches with one bidirectional Transformer encoder.
    """
)
class NeoMMEModel(NeoMMEPreTrainedModel):
    def __init__(self, config: NeoMMEConfig):
        super().__init__(config)
        self.embeddings = NeoMMEEmbeddings(config)
        self.patch_embeddings = NeoMMEPatchEmbeddings(config)
        self.rotary_emb = NeoMMERotaryEmbedding(config)
        self.embedding_norm = NeoMMERMSNorm(config.hidden_size, config.norm_eps, with_scale=False)
        self.final_norm = NeoMMERMSNorm(config.hidden_size, config.norm_eps, with_scale=False)
        self.layers = nn.ModuleList(
            [NeoMMEEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        global_layers = [i for i, layer_type in enumerate(config.layer_types) if layer_type == "full_attention"]
        self.value_embeddings = (
            nn.Embedding(config.vocab_size, config.num_key_value_heads * config.head_dim)
            if config.use_value_embeds and global_layers
            else None
        )
        self.value_embedding_layers = (
            {global_layers[0], global_layers[-1]} if self.value_embeddings is not None else set()
        )
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        position_ids (`torch.LongTensor` of shape `(2, batch_size, sequence_length)` or `(batch_size, sequence_length)`, *optional*):
            Positions for the input tokens. [`NeoMMEProcessor`] returns two-axis positions for document images. A
            one-axis position tensor is used for text inputs.
        pixel_values (`torch.Tensor` of shape `(num_patches, 3 * patch_size ** 2)`, *optional*):
            Flattened image patches returned by [`NeoMMEProcessor`]. The model places these patches at image
            placeholders in `input_ids`.
        inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embedding_rank)`, *optional*):
            Token embeddings before projection to `hidden_size`. Use `input_ids` for image inputs because the model
            needs the image placeholders to place `pixel_values`.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is not None and self.value_embeddings is not None:
            logger.warning_once("inputs_embeds cannot apply value embeddings without token IDs")

        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)  # (batch, seq, hidden_size)
        if pixel_values is not None:
            if input_ids is None:
                raise ValueError("`pixel_values` requires `input_ids` to locate image placeholder tokens.")
            hidden_states = self._scatter_patch_embeddings(input_ids, hidden_states, pixel_values)

        batch_size, seq_len = hidden_states.shape[:2]
        if position_ids is None:
            # One axis: `NeoMMERotaryEmbedding` expands it onto both, which is what text-only inputs want.
            position_ids = torch.arange(seq_len, device=hidden_states.device).expand(batch_size, -1)

        # Each encoder layer can mix in this normalized input through its learned `lambdas`.
        hidden_states = initial_hidden_states = self.embedding_norm(hidden_states)

        attention_masks = self._build_attention_masks(hidden_states, attention_mask)
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }
        # Value embeddings require token IDs, so `inputs_embeds`-only calls omit them.
        value_embeds = None
        if self.value_embeddings is not None and input_ids is not None:
            value_embeds = self.value_embeddings(input_ids)  # (batch, seq, kv_heads * head_dim)

        for layer_idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                initial_hidden_states,
                value_embeds if layer_idx in self.value_embedding_layers else None,
                position_embeddings=position_embeddings[encoder_layer.attention_type],
                attention_mask=attention_masks[layer_idx],
                **kwargs,
            )

        # The backbone applies the final normalization.
        hidden_states = self.final_norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)

    def _scatter_patch_embeddings(
        self, input_ids: torch.LongTensor, hidden_states: torch.Tensor, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """Scatter patch embeddings into image placeholder tokens."""
        if pixel_values.shape[-1] != self.config.patch_dim:
            raise ValueError(
                f"pixel_values has patch width {pixel_values.shape[-1]} but the model expects "
                f"{self.config.patch_dim} (= 3 * patch_size ** 2 with patch_size={self.config.patch_size})"
            )
        previous_ids = F.pad(input_ids[:, :-1], (1, 0), value=self.config.pad_token_id or 0)  # token IDs shifted right
        image_mask = (
            (input_ids == self.config.image_token_id) & (previous_ids != self.config.document_token_id)
        ).unsqueeze(-1)

        num_image_tokens = image_mask.sum()
        torch_compilable_check(
            num_image_tokens == pixel_values.shape[0],
            lambda: f"Got {pixel_values.shape[0]} image patches for {int(num_image_tokens)} image placeholder tokens",
        )
        patch_embeds = self.patch_embeddings(pixel_values.to(hidden_states.dtype))  # (patches, hidden_size)
        return hidden_states.masked_scatter(image_mask, patch_embeds)

    def _build_attention_masks(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> list[torch.Tensor | None]:
        """Build one attention mask per layer."""
        if isinstance(attention_mask, dict):
            return [attention_mask[layer_type] for layer_type in self.config.layer_types]

        mask_kwargs = {"config": self.config, "inputs_embeds": hidden_states, "attention_mask": attention_mask}
        masks: dict[int | None, torch.Tensor | None] = {}
        for window in set(self.config.layer_window_sizes):
            if window is None:
                masks[window] = create_bidirectional_mask(**mask_kwargs)
            else:
                masks[window] = create_bidirectional_mask(
                    **mask_kwargs, and_mask_function=sliding_window_bidirectional_overlay(window)
                )
        return [masks[window] for window in self.config.layer_window_sizes]


@auto_docstring(
    custom_intro="""
    The NeoMME model with a masked language modeling head.
    """
)
class NeoMMEForMaskedLM(NeoMMEPreTrainedModel):
    def __init__(self, config: NeoMMEConfig):
        super().__init__(config)
        self.model = NeoMMEModel(config)
        self.post_init()

    def get_output_embeddings(self):
        """The decode is tied through the factorized embedding; there is no separate output layer."""
        return None

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MaskedLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for the masked-language-modeling loss. Indices should be in `[0, ..., config.vocab_size - 1]`
            or `-100`; only tokens with a label different from `-100` contribute.
        """
        outputs: BaseModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.model.embeddings.decode(outputs.last_hidden_state)  # (batch, seq, vocab_size)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Output type for [`NeoMMEForRetrieval`].
    """
)
@dataclass
class NeoMMEForRetrievalOutput(BaseModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
        Retrieval loss. This value is always `None`.
    embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, embedding_dim)`, *optional*):
        Normalized token embeddings for late-interaction retrieval. Padding rows are zeroed. Use
        [`NeoMMEProcessor.score_retrieval`] to score these embeddings with MaxSim.
    dense_embeddings (`torch.FloatTensor` of shape `(batch_size, hidden_size)` or `(batch_size, dense_dim)`, *optional*):
        A normalized mean-pooled embedding for each input. When `dense_dim` is set, the last dimension is `dense_dim`.
        Use [`NeoMMEProcessor.score_retrieval`] to score these embeddings with cosine similarity.
    """

    loss: torch.FloatTensor | None = None
    embeddings: torch.FloatTensor | None = None
    dense_embeddings: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    The NeoMME model with multi-vector and dense retrieval heads. One forward pass can return token embeddings for
    MaxSim scoring and mean-pooled embeddings for cosine similarity.
    """
)
class NeoMMEForRetrieval(NeoMMEPreTrainedModel):
    def __init__(self, config: NeoMMEConfig):
        super().__init__(config)
        self.model = NeoMMEModel(config)
        self.embedding_proj_layer = nn.Linear(config.hidden_size, config.embedding_dim, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_multivector: bool = True,
        output_dense: bool = True,
        dense_dim: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> NeoMMEForRetrievalOutput:
        r"""
        output_multivector (`bool`, *optional*, defaults to `True`):
            Whether to return token embeddings for late-interaction retrieval.
        output_dense (`bool`, *optional*, defaults to `True`):
            Whether to return one mean-pooled dense embedding per input.
        dense_dim (`int`, *optional*):
            Width of the Matryoshka prefix to return for dense embeddings. The model truncates the pooled vector
            before normalizing it.
        """
        if not (output_multivector or output_dense):
            raise ValueError("At least one of `output_multivector` or `output_dense` must be True")

        outputs: BaseModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)

        embeddings = self._forward_late_head(hidden_states, attention_mask) if output_multivector else None
        dense_embeddings = self._forward_dense_head(hidden_states, attention_mask, dense_dim) if output_dense else None
        return NeoMMEForRetrievalOutput(
            embeddings=embeddings,
            dense_embeddings=dense_embeddings,
            last_hidden_state=hidden_states,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _forward_late_head(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Project and normalize token embeddings, then zero padding positions."""
        proj_dtype = self.embedding_proj_layer.weight.dtype
        embeddings = self.embedding_proj_layer(hidden_states.to(proj_dtype))  # (batch, seq, embedding_dim)
        embeddings = F.normalize(embeddings, dim=-1)
        # Overwrite padding rows rather than multiply them out, so a non-finite value cannot survive.
        return embeddings.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

    def _forward_dense_head(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, dense_dim: int | None = None
    ) -> torch.Tensor:
        """Mean-pool non-padding states and normalize the requested Matryoshka prefix."""
        expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.shape).to(hidden_states.dtype)
        pooled = (hidden_states * expanded_mask).sum(1) / expanded_mask.sum(1).clamp_min(1e-9)
        if dense_dim is None:
            return F.normalize(pooled, dim=-1)

        # Reject widths that slicing would accept but that would return an unexpected vector size.
        if not 0 < dense_dim <= pooled.shape[-1]:
            raise ValueError(f"dense_dim must be in 1..{pooled.shape[-1]} (the pooled width), got {dense_dim}")
        return F.normalize(pooled[..., :dense_dim], dim=-1)


__all__ = [
    "NeoMMEForMaskedLM",
    "NeoMMEForRetrieval",
    "NeoMMEImageProcessor",
    "NeoMMEModel",
    "NeoMMEPreTrainedModel",
]
