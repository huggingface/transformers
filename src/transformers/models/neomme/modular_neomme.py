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
from ...masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MaskedLMOutput
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    no_inherit_decorator,
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
from ..muse_glimmer.modeling_muse_glimmer import MuseGlimmerPreTrainedModel, MuseGlimmerTextAttention
from ..nemotron.modeling_nemotron import NemotronMLP
from .configuration_neomme import NeoMMEConfig


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
        resized_height = max(1, math.floor(height * scale))
        resized_width = max(1, math.floor(width * scale))
        if max_pixels is not None and resized_height * resized_width > max_pixels:
            if resized_height >= resized_width:
                resized_height = max(1, max_pixels // resized_width)
            else:
                resized_width = max(1, max_pixels // resized_height)
        return resized_height, resized_width

    if min_pixels is not None and height * width < min_pixels:
        if resized_height * resized_width >= min_pixels:
            return resized_height, resized_width
        ceiled_height = max(1, math.ceil(height * scale))
        ceiled_width = max(1, math.ceil(width * scale))
        exceeds_cap = (max_side is not None and max(ceiled_height, ceiled_width) > max_side) or (
            max_pixels is not None and ceiled_height * ceiled_width > max_pixels
        )
        if not exceeds_cap:
            return ceiled_height, ceiled_width

    return resized_height, resized_width


class NeoMMEImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to `self.patch_size`):
        Side, in pixels, of one patch token. The image is padded to a whole multiple of it.
    max_side (`int`, *optional*):
        Longest-side cap in pixels. Unset means no longest-side resize.
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
        size = kwargs["size"]
        if size is not None and set(dict(size)) != {"min_pixels", "max_pixels"}:
            raise ValueError("size must contain exactly min_pixels and max_pixels.")

        if kwargs["size"] is None:
            # Generic resize validation requires `size`; NeoMME can resize from `max_side` alone.
            kwargs.pop("do_resize", None)
        return super()._validate_preprocess_kwargs(**kwargs)

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


class NeoMMERMSNorm(Gemma4RMSNorm):
    pass


class NeoMMEEmbeddings(nn.Module):
    """Factorized (ALBERT-style) token embeddings: `vocab_size -> embedding_rank -> hidden_size`."""

    def __init__(self, config: NeoMMEConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_rank)
        self.embedding_projection = nn.Linear(config.embedding_rank, config.hidden_size, bias=False)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.embedding_projection(self.word_embeddings(input_ids))


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
        self, x: torch.Tensor, position_ids: torch.LongTensor, layer_type: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build cos/sin from two-axis `position_ids` of shape `(2, batch, seq_len)`."""
        # Same axias 2D rope as in Qwen-VIT models, all will be standardized by @raushan
        inv_freq = getattr(self, f"{layer_type}_inv_freq")  # (rotary_dim // 2,)
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            row_angles = position_ids[0].float().unsqueeze(-1) * inv_freq[0::2]
            column_angles = position_ids[1].float().unsqueeze(-1) * inv_freq[1::2]

            angles = torch.stack([row_angles, column_angles], dim=-1).flatten(-2)
            cos = (angles.cos() * attention_scaling).to(x.dtype)
            sin = (angles.sin() * attention_scaling).to(x.dtype)
        return torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)


class NeoMMEExclusiveSelfAttention(nn.Module):
    def __init__(self, config: NeoMMEConfig):
        super().__init__()
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.alpha = nn.Parameter(torch.zeros(config.num_attention_heads))

    def forward(self, attn_output: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        value_states = repeat_kv(value_states, self.num_key_value_groups).transpose(1, 2)
        value_unit = F.normalize(value_states.float(), dim=-1).to(attn_output.dtype)
        projection = (attn_output * value_unit).sum(-1, keepdim=True)
        scale = torch.tanh(self.alpha).to(attn_output.dtype).view(1, 1, -1, 1)
        return attn_output - (scale * projection) * value_unit


class NeoMMESigmoidGatedProjection(nn.Module):
    def __init__(self, config: NeoMMEConfig):
        super().__init__()
        projection_size = config.num_attention_heads * config.head_dim
        self.gate_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.o_proj = nn.Linear(projection_size, config.hidden_size, bias=config.attention_bias)

    def forward(self, attn_output: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.o_proj(attn_output * torch.sigmoid(self.gate_proj(hidden_states)))


@no_inherit_decorator
class NeoMMEAttention(MuseGlimmerTextAttention):
    """Bidirectional grouped-query attention with QK-norm, M-RoPE, and a sigmoid output gate.

    QK-norm runs before rotary embedding, value embeddings are added after rotation, and
    exclusive self-attention is applied before the output gate.
    """

    def __init__(self, config: NeoMMEConfig, layer_idx: int):
        super().__init__()
        del self.qk_norm
        del self.qk_scale_factor
        del self.gate_proj
        del self.o_proj

        self.attention_type = config.layer_types[layer_idx]
        self.num_attention_heads = config.num_attention_heads
        self.is_causal = False
        self.q_norm = NeoMMERMSNorm(config.head_dim, config.norm_eps, with_scale=False)
        self.k_norm = NeoMMERMSNorm(config.head_dim, config.norm_eps, with_scale=False)

        # `sliding_window` is a HALF-width (`abs(i - j) <= window`). The flash-attention path
        # builds an inclusive symmetric band of `sliding_window - 1` per side, hence the `+ 1`.
        self.sliding_window = (
            config.per_layer_config[layer_idx].sliding_window + 1 if self.is_local_attention else None
        )
        self.exclusive_self_attention = NeoMMEExclusiveSelfAttention(config)
        self.output_projection = NeoMMESigmoidGatedProjection(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        value_embeds: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if value_embeds is not None:
            value_states = value_states + value_embeds.view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

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

        attn_output = self.exclusive_self_attention(attn_output, value_states)
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.output_projection(attn_output, hidden_states)
        return attn_output, attn_weights


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


class NeoMMEPreTrainedModel(MuseGlimmerPreTrainedModel):
    input_modalities = ("image", "text")
    _no_split_modules = ["NeoMMEEncoderLayer"]
    _can_record_outputs = {
        "hidden_states": NeoMMEEncoderLayer,
        "attentions": NeoMMEAttention,
    }

    def get_input_embeddings(self) -> nn.Embedding:
        backbone = getattr(self, self.base_model_prefix, self)
        return backbone.embed_tokens.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        backbone = getattr(self, self.base_model_prefix, self)
        backbone.embed_tokens.word_embeddings = value

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        # `apply` visits children before parents, so the NeoMME-specific parent initialization below runs last.
        PreTrainedModel._init_weights(self, module)

        if isinstance(module, NeoMMEEmbeddings):
            init.normal_(module.word_embeddings.weight, mean=0.0, std=self.config.embedding_rank**-0.5)
        elif isinstance(module, NeoMMEExclusiveSelfAttention):
            init.zeros_(module.alpha)
        elif isinstance(module, NeoMMESigmoidGatedProjection):
            # Zero-init so the attention residual contributes nothing at initialization.
            init.zeros_(module.o_proj.weight)
        elif isinstance(module, NeoMMEMLP):
            init.zeros_(module.down_proj.weight)
        elif isinstance(module, NeoMMEEncoderLayer):
            init.copy_(module.lambdas, torch.tensor([1.0, 0.0]))
        elif isinstance(module, NeoMMEModel):
            init.zeros_(module.value_embeddings.weight)
        elif isinstance(module, NeoMMEForMaskedLM):
            init.normal_(module.lm_head.weight, mean=0.0, std=self.config.embedding_rank**-0.5)
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
        self.embed_tokens = NeoMMEEmbeddings(config)
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

    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        position_ids (`torch.LongTensor` of shape `(2, batch_size, sequence_length)` or `(batch_size, sequence_length)`, *optional*):
            Positions for the input tokens. [`NeoMMEProcessor`] returns two-axis positions for document images. A
            one-axis position tensor is used for text inputs.
        pixel_values (`torch.Tensor` of shape `(num_patches, 3 * patch_size ** 2)`, *optional*):
            Flattened image patches returned by [`NeoMMEProcessor`]. The model places these patches at image
            placeholders in `input_ids`.
        """
        hidden_states = self.embed_tokens(input_ids)
        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, return_dict=True)
            image_mask = self.get_placeholder_mask(input_ids, image_outputs.pooler_output)
            hidden_states = hidden_states.masked_scatter(image_mask, image_outputs.pooler_output)

        batch_size, seq_len = hidden_states.shape[:2]
        # create 2D positions - text uses the token index for both M-RoPE axes
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).expand(batch_size, -1)
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(2, -1, -1)

        # Reuse this normalized input as `initial_hidden_states` in every encoder layer.
        hidden_states = initial_hidden_states = self.embedding_norm(hidden_states)

        if not isinstance(attention_mask_mapping := attention_mask, dict):
            attention_mask_mapping: dict[int, torch.Tensor | None] = {}
            mask_kwargs = {"inputs_embeds": hidden_states, "attention_mask": attention_mask}
            for layer_id in range(self.config.num_hidden_layers):
                per_layer_config = self.config.per_layer_config[layer_id]
                if per_layer_config.sliding_window is not None:
                    attention_mask_mapping[layer_id] = create_bidirectional_sliding_window_mask(
                        config=per_layer_config,
                        **mask_kwargs,
                    )
                else:
                    attention_mask_mapping[layer_id] = create_bidirectional_mask(
                        config=per_layer_config,
                        **mask_kwargs,
                    )

        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }

        value_embeds = self.value_embeddings(input_ids)

        for layer_idx, encoder_layer in enumerate(self.layers):
            # Pass gradient-carrying tensors positionally so reentrant checkpointing tracks them.
            hidden_states = encoder_layer(
                hidden_states,
                initial_hidden_states,
                value_embeds if layer_idx in self.value_embedding_layers else None,
                position_embeddings=position_embeddings[encoder_layer.attention_type],
                attention_mask=attention_mask_mapping[layer_idx],
                **kwargs,
            )

        hidden_states = self.final_norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring(
    custom_intro="""
    The NeoMME model with a factorized masked token decoder.
    """
)
class NeoMMEForMaskedLM(NeoMMEPreTrainedModel):
    _tied_weights_keys = {
        "lm_head.weight": "model.embed_tokens.word_embeddings.weight",
        "unembedding_projection.weight": "model.embed_tokens.embedding_projection.weight",
    }

    def __init__(self, config: NeoMMEConfig):
        super().__init__(config)
        self.model = NeoMMEModel(config)
        self.unembedding_projection = nn.Linear(config.embedding_rank, config.hidden_size, bias=False)
        self.lm_head = nn.Linear(config.embedding_rank, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
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
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        hidden_states = hidden_states @ self.unembedding_projection.weight
        logits = self.lm_head(hidden_states)

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
        Normalized token embeddings for late-interaction retrieval. Padding rows are zeroed. Score them with MeanMaxSim.
    dense_embeddings (`torch.FloatTensor` of shape `(batch_size, hidden_size)` or `(batch_size, dense_dim)`, *optional*):
        A normalized mean-pooled embedding for each input. When `dense_dim` is set, the last dimension is `dense_dim`.
        Score them with cosine similarity.
    """

    loss: torch.FloatTensor | None = None
    embeddings: torch.FloatTensor | None = None
    dense_embeddings: torch.FloatTensor | None = None


class NeoMMEMultiVectorHead(nn.Module):
    def __init__(self, config: NeoMMEConfig):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.embedding_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.proj(hidden_states.to(self.proj.weight.dtype))
        embeddings = F.normalize(embeddings, dim=-1)
        # Use masked_fill because multiplying NaN or Inf by zero would leave padding non-finite.
        return embeddings.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)


class NeoMMEDenseHead(nn.Module):
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, dense_dim: int | None = None
    ) -> torch.Tensor:
        expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.shape).to(hidden_states.dtype)
        pooled = (hidden_states * expanded_mask).sum(1) / expanded_mask.sum(1).clamp_min(1e-9)
        if dense_dim is None:
            return F.normalize(pooled, dim=-1)

        if not 0 < dense_dim <= pooled.shape[-1]:
            raise ValueError(f"dense_dim must be in 1..{pooled.shape[-1]} (the pooled width), got {dense_dim}")
        return F.normalize(pooled[..., :dense_dim], dim=-1)


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
        self.multi_vector_head = NeoMMEMultiVectorHead(config)
        self.dense_head = NeoMMEDenseHead()
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
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
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)

        embeddings = self.multi_vector_head(hidden_states, attention_mask) if output_multivector else None
        dense_embeddings = self.dense_head(hidden_states, attention_mask, dense_dim) if output_dense else None
        return NeoMMEForRetrievalOutput(
            embeddings=embeddings,
            dense_embeddings=dense_embeddings,
            last_hidden_state=hidden_states,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "NeoMMEForMaskedLM",
    "NeoMMEForRetrieval",
    "NeoMMEImageProcessor",
    "NeoMMEModel",
    "NeoMMEPreTrainedModel",
]
