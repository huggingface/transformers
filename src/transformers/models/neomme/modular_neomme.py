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

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from ... import initialization as init
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...masking_utils import create_bidirectional_mask, sliding_window_bidirectional_overlay
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MaskedLMOutput
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
from ...utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from ...utils.generic import can_return_tuple, maybe_autocast
from ...utils.output_capturing import capture_outputs
from ...utils.type_validators import positive_int
from ..gemma4.modeling_gemma4 import Gemma4RMSNorm
from ..gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
from ..laguna.modeling_laguna import LagunaRotaryEmbedding
from ..lfm2_vl.image_processing_lfm2_vl import convert_image_to_patches
from ..llama.modeling_llama import eager_attention_forward, repeat_kv
from ..nemotron.modeling_nemotron import NemotronMLP
from .configuration_neomme import NeoMMEConfig


logger = logging.get_logger(__name__)


def get_resize_output_size(height: int, width: int, max_side: int | None, size: SizeDict | None) -> tuple[int, int]:
    """Compute integer height and width from the configured image size targets."""
    min_pixels = size.min_pixels if size is not None else None
    max_pixels = size.max_pixels if size is not None else None
    scale = (min_pixels / (height * width)) ** 0.5 if min_pixels is not None and height * width < min_pixels else 1.0
    if max_side is not None:
        scale = min(scale, max_side / max(height, width))
    if max_pixels is not None:
        scale = min(scale, (max_pixels / (height * width)) ** 0.5)
    if scale == 1.0:
        return height, width

    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    exceeds_cap = (max_side is not None and max(resized_height, resized_width) > max_side) or (
        max_pixels is not None and resized_height * resized_width > max_pixels
    )
    if exceeds_cap:
        return max(1, math.floor(height * scale)), max(1, math.floor(width * scale))

    if min_pixels is not None and height * width < min_pixels:
        if resized_height * resized_width >= min_pixels:
            return resized_height, resized_width
        resized_height = max(1, math.ceil(height * scale))
        resized_width = max(1, math.ceil(width * scale))
        exceeds_cap = (max_side is not None and max(resized_height, resized_width) > max_side) or (
            max_pixels is not None and resized_height * resized_width > max_pixels
        )
        if not exceeds_cap:
            return resized_height, resized_width

    return resized_height, resized_width


class NeoMMEImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to `self.patch_size`):
        Side, in pixels, of one patch token. The image is padded to a whole multiple of it.
    max_side (`int`, *optional*):
        Longest-side cap in pixels. Unset means no longest-side resize.
    size (`dict[str, int]`, *optional*):
        Pixel-area resize bounds with `min_pixels` and `max_pixels` keys. Bounds apply before patch-grid padding.
        Unset means no area-based resize.
    """

    patch_size: Annotated[int, positive_int()]
    max_side: Annotated[int | None, positive_int()]


@auto_docstring
class NeoMMEImageProcessor(TorchvisionBackend):
    r"""
    Constructs an image processor for NeoMME.

    The processor converts each image to RGB and splits it into row-major patches. It preserves the original size by
    default, but `max_side` or a pixel-area `size` can resize images. It pads the bottom and right edges to complete
    the final patch. A batch returns one concatenated patch tensor and one grid height and width pair per image.
    """

    valid_kwargs = NeoMMEImageProcessorKwargs

    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    patch_size = 32
    # Checkpoints may set resize limits; unset limits preserve native resolution.
    max_side = None
    size = None

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
        unsupported = sorted(
            name
            for name in ("crop_size", "do_center_crop", "do_pad", "pad_size", "image_seq_length")
            if kwargs.get(name) not in (None, False)
        )
        if unsupported:
            raise ValueError(f"NeoMMEImageProcessor does not implement these image kwargs: {unsupported}")

        self._validate_size_settings(
            kwargs.get("patch_size", self.patch_size),
            kwargs.get("max_side", self.max_side),
            kwargs.get("size", self.size),
        )
        if kwargs.get("size") is None:
            # Generic resize validation requires `size`; NeoMME can resize from `max_side` alone.
            kwargs.pop("do_resize", None)
        return super()._validate_preprocess_kwargs(**kwargs)

    @staticmethod
    def _validate_size_settings(patch_size: int, max_side: int | None, size: SizeDict | None) -> None:
        if not isinstance(patch_size, int) or isinstance(patch_size, bool) or patch_size <= 0:
            raise ValueError(f"patch_size must be a positive integer, got {patch_size!r}.")
        if size is not None and set(dict(size)) != {"min_pixels", "max_pixels"}:
            raise ValueError("size must contain exactly min_pixels and max_pixels.")

        for name, value in (
            ("max_side", max_side),
            ("size.max_pixels", size.max_pixels if size is not None else None),
            ("size.min_pixels", size.min_pixels if size is not None else None),
        ):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None, got {value!r}.")

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        patch_size: int,
        max_side: int | None,
        size: SizeDict | None,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self._resize_to_budget(stacked_images, max_side, size, resample)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        image_grids_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Pad to a whole patch grid before rescaling, so padded pixels become -1 exactly like the
            # black canvas the reference implementation pastes onto.
            stacked_images, grid_height, grid_width = self._pad_to_patch_grid(stacked_images, patch_size)
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = convert_image_to_patches(stacked_images, patch_size)
            image_grids_grouped[shape] = torch.tensor(
                [[grid_height, grid_width]] * len(stacked_images), dtype=torch.int64
            )

        pixel_values = reorder_images(processed_images_grouped, grouped_images_index)
        image_grid_hw = reorder_images(image_grids_grouped, grouped_images_index)
        return BatchFeature(
            data={
                "pixel_values": torch.cat(pixel_values, dim=0),  # (total_patches, patch_dim)
                "image_grid_hw": torch.stack(image_grid_hw),  # (images, 2)
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
        size = self._standardize_kwargs(size=images_kwargs.get("size", self.size))["size"]

        if images_kwargs.get("do_resize", self.do_resize):
            height, width = get_resize_output_size(height, width, max_side, size)

        return -(-height // patch_size) * (-(-width // patch_size))

    def _resize_to_budget(
        self,
        image: "torch.Tensor",
        max_side: int | None,
        size: SizeDict | None,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
    ) -> "torch.Tensor":
        height, width = image.shape[-2], image.shape[-1]
        resized_height, resized_width = get_resize_output_size(height, width, max_side, size)
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


class NeoMMERotaryEmbedding(LagunaRotaryEmbedding):
    """Two-axis interleaved M-RoPE with per-layer-type frequency spectra."""

    @torch.no_grad()
    @dynamic_rope_update
    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.LongTensor, layer_type: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build cos/sin from two-axis `position_ids` of shape `(2, batch, seq_len)`."""
        inv_freq = getattr(self, f"{layer_type}_inv_freq")  # (rotary_dim // 2,)
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        device_type = (
            hidden_states.device.type
            if isinstance(hidden_states.device.type, str) and hidden_states.device.type != "mps"
            else "cpu"
        )
        with maybe_autocast(device_type=device_type, enabled=False):
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

        # `sliding_window` is a HALF-width (`abs(i - j) <= window`). The flash-attention path
        # builds an inclusive symmetric band of `sliding_window - 1` per side, hence the `+ 1`.
        window = None if self.attention_type == "full_attention" else config.per_layer_config[layer_idx].sliding_window
        self.sliding_window = None if window is None else window + 1

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
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

        key_states = self.k_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim)
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
        self.residual_multiplier = config.residual_multiplier
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
        hidden_states = hidden_states + self.residual_multiplier * attn_output

        normed_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_states)
        return hidden_states + self.residual_multiplier * mlp_output


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
            init.normal_(module.word_embeddings.weight, mean=0.0, std=self.config.embedding_rank**-0.5)
        elif isinstance(module, NeoMMEAttention):
            # Zero-init so the attention residual contributes nothing at initialization.
            init.zeros_(module.o_proj.weight)
            init.zeros_(module.alpha)
        elif isinstance(module, NeoMMEMLP):
            init.zeros_(module.down_proj.weight)
        elif isinstance(module, NeoMMEEncoderLayer):
            init.copy_(module.lambdas, torch.tensor([1.0, 0.0]))
        elif isinstance(module, NeoMMEModel):
            init.zeros_(module.value_embeddings.weight)
        elif isinstance(module, NeoMMERotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), inv_freq)

    def _resize_token_embeddings(
        self, new_num_tokens: int, pad_to_multiple_of: int | None = None, mean_resizing: bool = True
    ) -> nn.Embedding:
        """Resize word and value embedding tables together."""
        word_embeddings = super()._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        backbone = getattr(self, self.base_model_prefix, self)
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
        self.value_embeddings = nn.Embedding(config.vocab_size, config.num_key_value_heads * config.head_dim)
        self.value_embedding_layers = {global_layers[0], global_layers[-1]}
        self.gradient_checkpointing = False
        self.post_init()

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
        if inputs_embeds is not None:
            logger.warning_once("inputs_embeds cannot apply value embeddings without token IDs")

        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)  # (batch, seq, hidden_size)
        if pixel_values is not None:
            if input_ids is None:
                raise ValueError("`pixel_values` requires `input_ids` to locate image placeholder tokens.")
            image_outputs = self.get_image_features(pixel_values, return_dict=True)
            image_features = image_outputs.pooler_output
            image_mask = self.get_placeholder_mask(input_ids, image_features)
            hidden_states = hidden_states.masked_scatter(image_mask, image_features)

        batch_size, seq_len = hidden_states.shape[:2]
        if position_ids is None:
            # Text uses the token index for both M-RoPE axes.
            position_ids = torch.arange(seq_len, device=hidden_states.device).expand(batch_size, -1)
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(2, -1, -1)

        # Reuse this normalized input as `initial_hidden_states` in every encoder layer.
        hidden_states = initial_hidden_states = self.embedding_norm(hidden_states)

        attention_masks = self._build_attention_masks(hidden_states, attention_mask)
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }

        # Value embeddings are token-ID lookups and cannot be recovered from `inputs_embeds`.
        value_embeds = self.value_embeddings(input_ids) if input_ids is not None else None

        for layer_idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                initial_hidden_states,
                value_embeds if layer_idx in self.value_embedding_layers else None,
                position_embeddings=position_embeddings[encoder_layer.attention_type],
                attention_mask=attention_masks[layer_idx],
                **kwargs,
            )

        hidden_states = self.final_norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)

    @can_return_tuple
    @auto_docstring(custom_intro="Projects flattened image patches into the model hidden space.")
    def get_image_features(
        self, pixel_values: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutputWithPooling:
        if pixel_values.shape[-1] != self.config.patch_dim:  # trf-ignore: TRF041
            raise ValueError(
                f"pixel_values has patch width {pixel_values.shape[-1]} but the model expects "
                f"{self.config.patch_dim} (= 3 * patch_size ** 2 with patch_size={self.config.patch_size})"
            )
        image_features = self.patch_embeddings(pixel_values.to(self.patch_embeddings.norm.weight.dtype))
        return BaseModelOutputWithPooling(last_hidden_state=image_features, pooler_output=image_features)

    def get_placeholder_mask(self, input_ids: torch.LongTensor, image_features: torch.Tensor) -> torch.BoolTensor:
        """Find patch placeholders and validate that every image feature has a destination."""
        previous_ids = F.pad(input_ids[:, :-1], (1, 0), value=self.config.pad_token_id or 0)  # token IDs shifted right
        image_mask = (input_ids == self.config.image_token_id) & (previous_ids != self.config.document_token_id)

        num_image_tokens = image_mask.sum()
        torch_compilable_check(
            num_image_tokens == image_features.shape[0],
            lambda: f"Got {image_features.shape[0]} image patches for {int(num_image_tokens)} image placeholder tokens",
        )
        return image_mask.unsqueeze(-1).expand(-1, -1, image_features.shape[-1])

    def _build_attention_masks(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> list[torch.Tensor | None]:
        """Build one attention mask per layer."""
        if isinstance(attention_mask, dict):
            return [attention_mask[layer_type] for layer_type in self.config.layer_types]

        mask_kwargs = {"inputs_embeds": hidden_states, "attention_mask": attention_mask}
        masks: dict[int | None, torch.Tensor | None] = {}
        attention_masks = []
        for layer_type, layer_config in zip(self.config.layer_types, self.config.per_layer_config):
            window = None if layer_type == "full_attention" else layer_config.sliding_window
            if window not in masks:
                if window is None:
                    masks[window] = create_bidirectional_mask(config=layer_config, **mask_kwargs)
                else:
                    masks[window] = create_bidirectional_mask(
                        config=layer_config,
                        **mask_kwargs,
                        and_mask_function=sliding_window_bidirectional_overlay(window),
                    )
            attention_masks.append(masks[window])
        return attention_masks


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
        Normalized token embeddings for late-interaction retrieval. Padding rows are zeroed. Score them with
        MeanMaxSim, for example with `sentence_transformers.util.mean_maxsim`.
    dense_embeddings (`torch.FloatTensor` of shape `(batch_size, hidden_size)` or `(batch_size, dense_dim)`, *optional*):
        A normalized mean-pooled embedding for each input. When `dense_dim` is set, the last dimension is `dense_dim`.
        Score them with cosine similarity.
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
