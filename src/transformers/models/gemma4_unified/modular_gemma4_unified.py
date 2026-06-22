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
import math
from collections import UserDict
from dataclasses import dataclass
from typing import Any, Literal

import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from torch.nn import init
from torchvision.transforms.v2 import functional as tvF

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...image_processing_utils import BatchFeature
from ...image_utils import PILImageResampling
from ...masking_utils import create_causal_mask, create_masks_for_generate, create_sliding_window_causal_mask
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring, can_return_tuple, torch_compilable_check
from ..gemma2.modeling_gemma2 import Gemma2DecoderLayer
from ..gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig
from ..gemma4.image_processing_gemma4 import (
    _SUPPORTED_SOFT_TOKENS,
    Gemma4ImageProcessor,
    Gemma4ImageProcessorKwargs,
    convert_image_to_patches,
    pad_along_first_dim,
)
from ..gemma4.modeling_gemma4 import (
    Gemma4CausalLMOutputWithPast,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4Model,
    Gemma4ModelOutputWithPast,
    Gemma4MultimodalEmbedder,
    Gemma4PreTrainedModel,
    Gemma4RMSNorm,
    Gemma4TextAttention,
    Gemma4TextMLP,
    Gemma4TextModelOutputWithPast,
    Gemma4TextRotaryEmbedding,
    Gemma4TextScaledWordEmbedding,
    get_block_sequence_ids_for_mask,
)
from ..gemma4.processing_gemma4 import Gemma4Processor, Gemma4ProcessorKwargs
from ..gemma4.video_processing_gemma4 import (
    Gemma4VideoProcessor,
    Gemma4VideoProcessorKwargs,
    convert_video_to_patches,
    pad_to_max_patches,
)
from ..llama.modeling_llama import LlamaModel


class Gemma4UnifiedImageProcessorKwargs(Gemma4ImageProcessorKwargs):
    """
    patch_size (`int`, *optional*):
        Size of each teacher image patch in pixels (before merging).
    max_soft_tokens (`int`, *optional*):
        Maximum number of soft (vision) tokens per image after patch merging.
        Must be one of {70, 140, 280, 560, 1120}.
    pooling_kernel_size (`int`, *optional*):
        Kernel size for merging teacher patches into model patches.
    """


def patches_merge(
    patches: "torch.Tensor",
    positions_xy: "torch.Tensor",
    length: int,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Merge k×k groups of small patches into larger patches.

    Given `L` input patches of dimension `D = patch_size² × 3`, merge groups of
    `k×k` spatially adjacent patches into `length` output patches of dimension
    `(k × patch_size)² × 3`. The spatial grouping is determined by integer-dividing
    the XY positions by `k`.

    Args:
        patches: (*, L, D) — input patches.
        positions_xy: (*, L, 2) — integer XY positions for each patch (-1 for padding).
        length: target number of output patches. Must satisfy L = length × k².

    Returns:
        merged_patches: (*, length, k²×D) — merged patch features.
        merged_positions: (*, length, 2) — new XY positions for merged patches.
    """
    patch_size = math.isqrt(patches.shape[-1] // 3)
    if patches.shape[-1] != patch_size * patch_size * 3:
        raise ValueError(f"Patch dimension {patches.shape[-1]} is not a valid `patch_size * patch_size * 3`")

    k = math.isqrt(patches.shape[-2] // length)
    if k * k * length != patches.shape[-2]:
        raise ValueError(f"Cannot merge {patches.shape} to {length}")

    # Compute target ordering for reordering patches into kernel-grouped order.
    # This ensures patches within each k×k kernel are contiguous.
    max_x = positions_xy[..., 0].max(dim=-1, keepdim=True)[0] + 1
    kernel_idxs = torch.div(positions_xy, k, rounding_mode="floor")
    num_patches_from_top_left = k * k * kernel_idxs[..., 0] + k * max_x * kernel_idxs[..., 1]

    position_within_kernel = torch.remainder(positions_xy, k)
    num_patches_from_top_left_of_kernel = position_within_kernel[..., 0] + position_within_kernel[..., 1] * k
    target_ordering = num_patches_from_top_left_of_kernel + num_patches_from_top_left

    # Reorder patches by computing the inverse permutation via argsort,
    # then gathering patches into kernel-grouped order.
    perm = target_ordering.long().argsort(dim=-1)  # inverse permutation
    # Expand perm indices to match patch feature dimension for gathering
    perm_expanded = perm.unsqueeze(-1).expand_as(patches)
    kernel_ordered_patches = patches.gather(-2, perm_expanded)

    batch_shape = patches.shape[:-2]

    # Reshape: (*, length*k*k, patch_size*patch_size*3) → (*, length, (k*patch_size)*(k*patch_size)*3)
    kernel_ordered_patches = kernel_ordered_patches.reshape(*batch_shape, length, k * k, patch_size, patch_size, 3)
    # Rearrange (l, a*b, p, q, c) → (l, a*p, b*q, c)
    kernel_ordered_patches = kernel_ordered_patches.reshape(*batch_shape, length, k, k, patch_size, patch_size, 3)
    kernel_ordered_patches = kernel_ordered_patches.permute(
        *range(len(batch_shape)), -6, -5, -3, -4, -2, -1
    )  # (..., l, k, p, k, q, c)
    merged_patches = kernel_ordered_patches.reshape(*batch_shape, length, k * patch_size * k * patch_size * 3)

    # Compute new positions for merged patches
    perm_pos = perm.unsqueeze(-1).expand_as(positions_xy)
    kernel_ordered_positions = positions_xy.float().gather(-2, perm_pos.long())

    # Handle padding: preserve -1 positions
    padding = (positions_xy == -1).all(dim=-1, keepdim=True)  # (..., L, 1)
    kernel_ordered_positions = kernel_ordered_positions * (~padding).float() + positions_xy.float() * padding.float()

    # Reshape positions and take min within each kernel to get the merged position
    kernel_ordered_positions = kernel_ordered_positions.reshape(*batch_shape, length, k * k, 2)
    new_positions = torch.div(kernel_ordered_positions, k, rounding_mode="floor")
    # For each merged patch, take the minimum position across the kernel
    new_positions = new_positions.min(dim=-2)[0].to(torch.long)

    return merged_patches, new_positions


@auto_docstring(custom_intro="Constructs a Gemma4 unified image processor.")
class Gemma4UnifiedImageProcessor(Gemma4ImageProcessor):
    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        patch_size: int | None = None,
        max_soft_tokens: int | None = None,
        pooling_kernel_size: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}.")

        # Compute max_patches from max_soft_tokens and pooling_kernel_size
        max_patches = max_soft_tokens * pooling_kernel_size**2

        # Process each image individually: resize, rescale/normalize, patchify, pad.
        # Images have different aspect ratios and thus different resized dimensions,
        # so patchification and padding must happen per-image before stacking.
        pixel_values = []
        position_ids = []
        num_soft_tokens_per_image = []

        for image in images:
            # Step 1: Aspect-ratio-preserving resize
            if do_resize:
                image = self.aspect_ratio_preserving_resize(
                    image=image,
                    patch_size=patch_size,
                    max_patches=max_patches,
                    pooling_kernel_size=pooling_kernel_size,
                    resample=resample,
                )

            # Step 2: Rescale pixel values (typically to [0, 1]) and optionally identity normalize
            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)

            # Step 3: Patchify into teacher-size patches (16px)
            # (num_channels, height, width) → (num_teacher_patches, patch_size²*3)
            patch_height = image.shape[-2] // patch_size
            patch_width = image.shape[-1] // patch_size
            teacher_patches = convert_image_to_patches(image, patch_size)

            # Step 4: Compute teacher-level position IDs
            device = image.device
            patch_grid = torch.meshgrid(
                torch.arange(patch_width, device=device),
                torch.arange(patch_height, device=device),
                indexing="xy",
            )
            teacher_positions = torch.stack(patch_grid, dim=-1).reshape(teacher_patches.shape[0], 2)

            # Step 5: Merge k×k teacher patches into model patches via patches_merge
            # (num_teacher_patches, 768) → (num_model_patches, 6912)
            num_model_patches = teacher_patches.shape[0] // (pooling_kernel_size**2)
            merged_patches, merged_positions = patches_merge(
                teacher_patches.unsqueeze(0),  # add batch dim for patches_merge
                teacher_positions.unsqueeze(0),
                num_model_patches,
            )
            merged_patches = merged_patches.squeeze(0)  # remove batch dim
            merged_positions = merged_positions.squeeze(0)
            num_soft_tokens_per_image.append(merged_patches.shape[0])

            # Step 6: Pad merged patches and positions to max_soft_tokens
            merged_patches, merged_positions = pad_along_first_dim(merged_patches, merged_positions, max_soft_tokens)
            pixel_values.append(merged_patches)
            position_ids.append(merged_positions)

        # Stack into batch tensors
        pixel_values = torch.stack(pixel_values, dim=0)  # (batch, max_soft_tokens, model_patch_size²*3)
        position_ids = torch.stack(position_ids, dim=0)  # (batch, max_soft_tokens, 2)

        data = {
            "pixel_values": pixel_values,
            "image_position_ids": position_ids,
            "num_soft_tokens_per_image": num_soft_tokens_per_image,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


class Gemma4UnifiedVideoProcessorKwargs(Gemma4VideoProcessorKwargs):
    pass


class Gemma4UnifiedVideoProcessor(Gemma4VideoProcessor):
    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        do_resize: bool,
        resample: "tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        patch_size: int | None = None,
        max_soft_tokens: int | None = None,
        pooling_kernel_size: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}.")

        # Compute max_patches from max_soft_tokens and pooling_kernel_size
        max_patches = max_soft_tokens * pooling_kernel_size**2

        # Process each image individually: resize, rescale/normalize, patchify, pad.
        # Images have different aspect ratios and thus different resized dimensions,
        # so patchification and padding must happen per-image before stacking.
        pixel_values = []
        position_ids = []
        num_soft_tokens_per_video = []

        for video in videos:
            if do_resize:
                # Step 1: Aspect-ratio-preserving resize
                video = self.aspect_ratio_preserving_resize(
                    video=video,
                    patch_size=patch_size,
                    max_patches=max_patches,
                    pooling_kernel_size=pooling_kernel_size,
                    resample=resample,
                )

            # Step 2: Rescale pixel values (typically to [0, 1]) and optionally identity normalize
            video = self.rescale_and_normalize(video, do_rescale, rescale_factor, do_normalize, image_mean, image_std)

            # Step 3: Patchify into teacher-size patches (16px)
            num_frames = video.shape[0]
            patch_height = video.shape[-2] // patch_size
            patch_width = video.shape[-1] // patch_size
            patches = convert_video_to_patches(video, patch_size)

            # Step 4: Compute teacher-level position IDs
            device = video.device
            patch_grid = torch.meshgrid(
                torch.arange(patch_width, device=device),
                torch.arange(patch_height, device=device),
                indexing="xy",
            )
            stacked_grid = torch.stack(patch_grid, dim=-1)
            teacher_positions = stacked_grid.reshape(patches.shape[1], 2)
            teacher_positions = teacher_positions[None, ...].repeat(num_frames, 1, 1)

            # Step 5: Merge k×k teacher patches into model patches via patches_merge
            # (num_frames, num_teacher_patches, 768) → (num_frames, num_model_patches, 6912)
            num_model_patches = patches.shape[1] // (pooling_kernel_size**2)
            merged_patches, merged_positions = patches_merge(patches, teacher_positions, num_model_patches)
            num_soft_tokens_per_video.append(merged_patches.shape[1])

            # Step 6: Pad merged patches and positions to max_soft_tokens
            merged_patches, merged_positions = pad_to_max_patches(merged_patches, merged_positions, max_soft_tokens)
            pixel_values.append(merged_patches)
            position_ids.append(merged_positions)

        # Stack into batch tensors
        pixel_values = torch.stack(pixel_values, dim=0)
        position_ids = torch.stack(position_ids, dim=0)

        data = {
            "pixel_values_videos": pixel_values,
            "video_position_ids": position_ids,
            "num_soft_tokens_per_video": num_soft_tokens_per_video,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


class Gemma4UnifiedProcessorKwargs(Gemma4ProcessorKwargs):
    pass


class Gemma4UnifiedProcessor(Gemma4Processor):
    def replace_audio_token(self, audio_inputs: dict, audio_idx: int) -> str:
        """Replace the audio placeholder with the correct number of audio tokens.

        Unlike standard Gemma4 which has a conformer audio encoder with two stride-2
        convolution blocks (reducing token count ~4×), the unified model projects raw
        waveform frames directly through RMSNorm → Linear with **no downsampling**.
        So the number of output soft tokens equals the number of valid input frames.
        """
        mask = audio_inputs["input_features_mask"][audio_idx]
        return f"{self.boa_token}{self.audio_token * int(mask.sum())}{self.eoa_token}"

    def _compute_audio_num_tokens(self, audio_waveform, sampling_rate: int) -> int:
        """Compute the number of audio soft tokens for a single waveform.

        For the unified model, the audio pipeline is much simpler than the
        standard Gemma4 model. There is no mel spectrogram or conformer encoder.
        Raw audio samples are simply chunked into fixed-length frames of
        `audio_samples_per_token` samples each.

        The number of tokens is: ceil(num_samples / audio_samples_per_token)

        Args:
            audio_waveform: A 1-D numpy array or list containing the raw audio samples.
            sampling_rate: The sampling rate of the audio waveform in Hz (unused, kept
                for API compatibility with parent).

        Returns:
            The number of audio soft tokens to insert as placeholders.
        """
        num_samples = len(audio_waveform)
        audio_samples_per_token = getattr(self.feature_extractor, "audio_samples_per_token", 640)
        return math.ceil(num_samples / audio_samples_per_token)


@auto_docstring(checkpoint="google/gemma-4-12B-it")
@strict
class Gemma4UnifiedAudioConfig(PreTrainedConfig):
    r"""
    audio_embed_dim (`int`, defaults to 640):
        Dimension of audio features input to the multimodal embedder. Each audio soft
        token is a raw waveform frame of `audio_samples_per_token` samples, so
        `audio_embed_dim == audio_samples_per_token`.
    """

    model_type = "gemma4_unified_audio"

    audio_embed_dim: int = 640
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    @property
    def hidden_size(self):
        """
        Used by the MultimodalEmbedder to determine the input projection size.
        """
        return self.audio_embed_dim

    @property
    def output_proj_dims(self):
        """
        Output projection dimension (matches `hidden_size`).
        """
        return self.audio_embed_dim

    @property
    def audio_samples_per_token(self):
        """
        Number of raw audio samples per output soft token. At 16 kHz this equals 40ms
        of audio per token (16000 / 25 fps = 640). Same as `audio_embed_dim`
        """
        return self.audio_embed_dim

    # Setters are setting the `audio_embed_dim` in each case as it's shared across
    @hidden_size.setter
    def hidden_size(self, value):
        self.audio_embed_dim = value

    @output_proj_dims.setter
    def output_proj_dims(self, value):
        self.audio_embed_dim = value

    @audio_samples_per_token.setter
    def audio_samples_per_token(self, value):
        self.audio_embed_dim = value


@auto_docstring(checkpoint="google/gemma-4-12B-it")
@strict
class Gemma4UnifiedTextConfig(Gemma4TextConfig):
    r"""
    use_bidirectional_attention (`str`, *optional*):
        Controls bidirectional attention behavior. When set to `"vision"`, vision tokens
        attend bidirectionally while text tokens use causal attention. When set to `"all"`,
        all tokens use bidirectional attention.
    num_global_key_value_heads (`int`, *optional*):
        Number of key-value heads for global (full) attention layers. If `None`, defaults
        to `num_key_value_heads`.
    global_head_dim (`int`, defaults to 512):
        Dimension of each attention head in global (full) attention layers.
    attention_k_eq_v (`bool`, defaults to `False`):
        Whether keys and values share the same projection weights. When `True`, the key
        projection output is reused as the value projection.
    num_kv_shared_layers (`int`, defaults to 0):
        Number of consecutive decoder layers that share the same key-value projections.
        A value of 0 means no sharing (each layer has independent KV projections).
    use_double_wide_mlp (`bool`, defaults to `False`):
        Whether to use a double-width MLP with fused gate and up projections.
    """

    model_type = "gemma4_unified_text"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_ep_plan = None  # No MoE

    sliding_window: int = 1024
    use_bidirectional_attention: Literal["all", "vision"] | None = "vision"
    max_position_embeddings: int = 262_144

    # PLE and MoE is not used
    vocab_size_per_layer_input = AttributeError()
    hidden_size_per_layer_input = AttributeError()
    enable_moe_block = AttributeError()
    num_experts = AttributeError()
    top_k_experts = AttributeError()
    moe_intermediate_size = AttributeError()
    output_router_logits = AttributeError()
    router_aux_loss_coef = AttributeError()


@auto_docstring(checkpoint="google/gemma-4-12B-it")
@strict
class Gemma4UnifiedVisionConfig(PreTrainedConfig):
    r"""
    patch_size (`int`, defaults to 16):
        Size of the image patches in pixels. Images are first patchified at this resolution.
    pooling_kernel_size (`int`, defaults to 3):
        Kernel size for merging patches into model patches. A 3×3 merge produces
        model patches of size `patch_size * pooling_kernel_size = 48` pixels.
    mm_embed_dim (`int`, defaults to 3840):
        Hidden dimension for the patch embedding Dense projection (matches the text model `hidden_size`).
    mm_posemb_size (`int`, defaults to 1120):
        Size of the factorized 2D positional embedding table. The table has shape
        `(mm_posemb_size, 2, mm_embed_dim)` and is looked up per-axis.
    output_proj_dims (`int`, defaults to 3840):
        Output dimension of the multimodal embedder projection (maps to text hidden size).
        This is set by the composite config's text_config.hidden_size at runtime.
    """

    model_type = "gemma4_unified_vision"

    patch_size: int = 16
    pooling_kernel_size: int = 3
    mm_embed_dim: int = 3840
    mm_posemb_size: int = 1120
    rms_norm_eps: float = 1e-6
    output_proj_dims: int = 3840
    initializer_range: float = 0.02

    @property
    def model_patch_size(self):
        """
        Size of the merged model patches in pixels. Each merged patch has `model_patch_size² * 3`
        raw pixel channels.
        """
        return self.patch_size * self.pooling_kernel_size

    @model_patch_size.setter
    def model_patch_size(self, value):
        if value != self.patch_size * self.pooling_kernel_size:
            raise ValueError(
                f"`model_patch_size` needs to be equal to {self.patch_size = } * {self.pooling_kernel_size = }"
            )


@auto_docstring(checkpoint="google/gemma-4-12B-it")
@strict
class Gemma4UnifiedConfig(Gemma4Config):
    sub_configs = {
        "text_config": Gemma4UnifiedTextConfig,
        "vision_config": Gemma4UnifiedVisionConfig,
        "audio_config": Gemma4UnifiedAudioConfig,
    }

    text_config: Gemma4UnifiedTextConfig | dict[str, Any] | None = None
    vision_config: Gemma4UnifiedVisionConfig | dict[str, Any] | None = None
    audio_config: Gemma4UnifiedAudioConfig | dict[str, Any] | None = None


@auto_docstring
@dataclass
class Gemma4UnifiedAudioModelOutput(ModelOutput):
    r"""
    pooler_output (`torch.FloatTensor` of shape `(batch_size, ..., hidden_size)`):
        Last hidden state that went through the audio specific multimodal projectors.
    attention_mask (`torch.BoolTensor`, *optional*):
        A torch.BoolTensor of shape `(batch_size, num_frames)`. True for valid positions, False for padding.
    """

    pooler_output: torch.FloatTensor | None = None
    attention_mask: torch.BoolTensor | None = None


@auto_docstring
@dataclass
class Gemma4UnifiedVisionModelOutput(ModelOutput):
    r"""
    pooler_output (`torch.FloatTensor` of shape `(batch_size, ..., hidden_size)`):
        Last hidden state that went through the vision specific multimodal projectors.
    """

    pooler_output: torch.FloatTensor | None = None


class Gemma4UnifiedTextModelOutputWithPast(Gemma4TextModelOutputWithPast):
    pass


class Gemma4UnifiedCausalLMOutputWithPast(Gemma4CausalLMOutputWithPast):
    pass


class Gemma4UnifiedModelOutputWithPast(Gemma4ModelOutputWithPast):
    pass


class Gemma4UnifiedRMSNorm(Gemma4RMSNorm):
    pass


class Gemma4UnifiedTextRotaryEmbedding(Gemma4TextRotaryEmbedding):
    pass


class Gemma4UnifiedTextAttention(Gemma4TextAttention):
    pass


class Gemma4UnifiedTextMLP(Gemma4TextMLP):
    pass


class Gemma4UnifiedTextDecoderLayer(Gemma2DecoderLayer):
    def __init__(self, config: Gemma4UnifiedTextConfig | Gemma4UnifiedVisionConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.self_attn = Gemma4UnifiedTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4UnifiedTextMLP(config, layer_idx)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states


class Gemma4UnifiedTextScaledWordEmbedding(Gemma4TextScaledWordEmbedding):
    pass


class Gemma4UnifiedPreTrainedModel(Gemma4PreTrainedModel):
    _no_split_modules = ["Gemma4UnifiedTextDecoderLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Gemma4UnifiedTextRotaryEmbedding):
            for layer_type, rope_init_fn in module.rope_init_fns.items():
                rope_init_fn_kwargs = {"layer_type": layer_type}
                if layer_type == "full_attention" and module.rope_type[layer_type] == "proportional":
                    rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"

                curr_inv_freq, _ = rope_init_fn(module.config, **rope_init_fn_kwargs)
                getattr(module, f"{layer_type}_inv_freq").copy_(curr_inv_freq)
                getattr(module, f"{layer_type}_original_inv_freq").copy_(curr_inv_freq)
        elif isinstance(module, Gemma4UnifiedTextScaledWordEmbedding):
            init.constant_(module.embed_scale, module.scalar_embed_scale)
        elif isinstance(module, Gemma4UnifiedTextDecoderLayer):
            init.ones_(module.layer_scalar)
        elif isinstance(module, Gemma4UnifiedVisionEmbedder):
            init.normal_(module.pos_embedding, mean=0.0, std=self.config.vision_config.initializer_range)

    def get_per_layer_input_embeddings(self):
        raise AttributeError("PLE is not used")

    def set_per_layer_input_embeddings(self):
        raise AttributeError("PLE is not used")

    def resize_token_embeddings(self):
        raise AttributeError("PLE is not used")

    def _resize_per_layer_embeddings(self):
        raise AttributeError("PLE is not used")


@auto_docstring(custom_intro="The base Gemma 4 unified language model without a language modeling head.")
class Gemma4UnifiedTextModel(Gemma4UnifiedPreTrainedModel, LlamaModel):
    config: Gemma4UnifiedTextConfig
    input_modalities = ("text",)
    _can_record_outputs = {
        "hidden_states": Gemma4UnifiedTextDecoderLayer,
        "attentions": Gemma4UnifiedTextAttention,
    }

    def __init__(self, config: Gemma4UnifiedTextConfig):
        super().__init__(config)

        # Gemma4Unified downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma4UnifiedTextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.norm = Gemma4UnifiedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.unique_layer_types = set(self.config.layer_types)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Gemma4UnifiedTextModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        # Initialize as empty dict, or reuse past shared states. We use a UserDict instead of built-in dict (it behaves
        # the same) for fsdp2 support (otherwise, `_apply_to_tensors` rebuilds every dict it recurses into, and `shared_kv_states`
        # is not correctly shared, see https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/utils.py#L223-L255)
        shared_kv_states = kwargs.pop("shared_kv_states", UserDict())

        # decoder layers
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return Gemma4UnifiedTextModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            shared_kv_states=shared_kv_states if kwargs.get("return_shared_kv_states", False) else None,
        )


class Gemma4UnifiedForCausalLM(Gemma4ForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Gemma4UnifiedCausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Gemma4UnifiedForCausalLM

        >>> model = Gemma4UnifiedForCausalLM.from_pretrained("google/gemma-4-12B-it")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-12B-it")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: Gemma4UnifiedTextModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return Gemma4UnifiedCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            shared_kv_states=outputs.shared_kv_states,
        )


class Gemma4UnifiedVisionEmbedder(nn.Module):
    """Encoder-free vision embedder: projects raw merged pixel patches into LM space.

    This replaces the entire SigLIP-like vision tower. Instead of attention layers,
    it uses a simple Dense projection with LayerNorm and factorized positional
    embeddings.

    Pipeline: raw_patches → LN₁ → Dense → LN₂ → +factorized_posemb → LN₃ → RMSNorm → Linear
    """

    def __init__(self, vision_config: Gemma4UnifiedVisionConfig, text_config: Gemma4UnifiedTextConfig):
        super().__init__()
        patch_dim = vision_config.model_patch_size**2 * 3  # 48*48*3 = 6912
        mm_embed_dim = vision_config.mm_embed_dim

        # Patch embedding: LN₁ → Dense → LN₂
        # Matches mm_patch_embed_ln1, mm_patch_embed_dense, mm_patch_embed_ln2 in EFEmbedder
        self.patch_ln1 = nn.LayerNorm(patch_dim)
        self.patch_dense = nn.Linear(patch_dim, mm_embed_dim)
        self.patch_ln2 = nn.LayerNorm(mm_embed_dim)

        # Factorized 2D positional embedding
        # Matches mm_pos_embedding in EFEmbedder: shape (mm_posemb_size, 2, mm_embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(vision_config.mm_posemb_size, 2, mm_embed_dim))
        self.pos_norm = nn.LayerNorm(mm_embed_dim)

        # Final multimodal projection (same as for audio): RMSNorm → Linear
        self.multimodal_embedder = Gemma4UnifiedMultimodalEmbedder(vision_config, text_config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch, num_patches, model_patch_size²*3) — raw merged pixel patches.
            image_position_ids: (batch, num_patches, 2) — integer XY positions (-1 for padding).

        Returns:
            (batch, num_patches, mm_embed_dim) — embedded features (including padding positions).
        """
        # Step 1: Patch embedding (LN → Dense → LN)
        hidden_states = self.patch_ln1(pixel_values.to(self.patch_dense.weight.dtype))
        hidden_states = self.patch_dense(hidden_states)
        hidden_states = self.patch_ln2(hidden_states)

        # Step 2: Add factorized positional embeddings + LN
        clamped = image_position_ids.clamp(min=0).long()
        valid = (image_position_ids != -1).to(self.pos_embedding.dtype).unsqueeze(-1)
        axes = torch.arange(2, device=image_position_ids.device)
        pos_embs = (self.pos_embedding[clamped, axes] * valid).sum(-2)
        hidden_states = hidden_states + pos_embs
        hidden_states = self.pos_norm(hidden_states)

        # Step 3: Base multimodal embedder (RMSNorm → Dense)
        hidden_states = self.multimodal_embedder(hidden_states)

        return hidden_states


class Gemma4UnifiedMultimodalEmbedder(Gemma4MultimodalEmbedder):
    def __init__(
        self,
        multimodal_config: Gemma4UnifiedAudioConfig | Gemma4UnifiedVisionConfig,
        text_config: Gemma4UnifiedTextConfig,
    ):
        super().__init__(multimodal_config, text_config)

        # Both AudioConfig and VisionConfig define output_proj_dims directly
        self.multimodal_hidden_size = multimodal_config.output_proj_dims

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # Additional dtype casting
        inputs_embeds = inputs_embeds.to(self.embedding_projection.weight.dtype)
        embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(embs_normed)


class Gemma4UnifiedModel(Gemma4Model):
    """Encoder-free multimodal model.

    This model has **no vision tower** and **no audio tower**. Vision inputs are
    handled via the `Gemma4UnifiedVisionEmbedder` (raw patches → LN → Dense
    → LN → +posemb → LN → RMSNorm → Linear) and audio inputs are raw waveform frames
    projected directly through the multimodal embedder (RMSNorm → Linear).
    """

    def __init__(self, config: Gemma4UnifiedConfig):
        super().__init__(config)
        del self.audio_tower
        del self.vision_tower
        del self.vocab_size_per_layer_input

        self.embed_vision = (
            Gemma4UnifiedVisionEmbedder(config.vision_config, config.text_config)
            if config.vision_config is not None
            else None
        )

        self.embed_audio = (
            Gemma4UnifiedMultimodalEmbedder(config.audio_config, config.text_config)
            if config.audio_config is not None
            else None
        )

    def get_per_layer_input_embeddings(self):
        raise AttributeError("PLE is not used")

    def set_per_layer_input_embeddings(self):
        raise AttributeError("PLE is not used")

    @can_return_tuple
    @auto_docstring(custom_intro="Projects raw pixel patches into language model space via the unified pipeline.")
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> Gemma4UnifiedVisionModelOutput:
        r"""
        image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
            The patch positions as (x, y) coordinates in the image. Padding patches are indicated by (-1, -1).
        """
        vision_outputs = self.embed_vision(pixel_values, image_position_ids)

        # Strip padding patches before scattering into text sequence.
        # Padding patches have position_ids == -1 on both axes.
        # We only scatter non-padding patches into the placeholder token positions.
        padding_mask = (image_position_ids == -1).all(dim=-1).to(vision_outputs.device)  # (batch, num_patches)

        # Flatten valid patches: keep only non-padding patches across the batch
        vision_outputs = vision_outputs[~padding_mask]  # (total_valid_patches, text_hidden_size)

        return Gemma4UnifiedVisionModelOutput(
            pooler_output=vision_outputs,
        )

    @can_return_tuple
    @auto_docstring(custom_intro="Projects video frames into language model space via the unified pipeline.")
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> Gemma4UnifiedVisionModelOutput:
        r"""
        video_position_ids (`torch.LongTensor` of shape `(num_videos, num_frames, max_patches, 2)`, *optional*):
            2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding.
        """
        # Flatten video frames: (num_videos, num_frames, ...) → (num_videos*num_frames, ...)
        pixel_values_videos = pixel_values_videos.flatten(0, 1)
        video_position_ids = video_position_ids.flatten(0, 1)

        # Use the same unified pipeline as images
        return self.get_image_features(pixel_values_videos, video_position_ids, **kwargs)

    @can_return_tuple
    @auto_docstring(
        custom_intro="Projects raw audio waveform features directly into language model space (no audio tower)."
    )
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> Gemma4UnifiedAudioModelOutput:
        r"""
        input_features (`torch.FloatTensor` of shape `(num_audios, num_tokens, audio_embed_dim)`):
            Raw waveform features. Each token is a frame of `audio_samples_per_token` (640)
            raw audio samples. For unified models this replaces the mel spectrogram + conformer pipeline.
        input_features_mask (`torch.BoolTensor` of shape `(num_audios, num_tokens)`, *optional*):
            Mask indicating valid (non-padding) audio tokens. `True` = valid, `False` = padding.
        """
        if self.embed_audio is None:
            raise ValueError(
                "Audio features were requested, but the model was initialized without an audio_config. "
                "Cannot process audio without an audio embedder."
            )

        audio_outputs = self.embed_audio(inputs_embeds=input_features)

        return Gemma4UnifiedAudioModelOutput(
            pooler_output=audio_outputs,
            attention_mask=input_features_mask.to(audio_outputs.device) if input_features_mask is not None else None,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        image_position_ids: torch.LongTensor | None = None,
        video_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Gemma4UnifiedModelOutputWithPast:
        r"""
        input_features_mask (`torch.FloatTensor]` of shape `(num_images, seq_length)`):
            The attention mask for the input audio.
        image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
            2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        video_position_ids (`torch.LongTensor` of shape `(num_videos, num_frames, max_patches, 2)`, *optional*):
            2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        image_mask, video_mask, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds)
        multimodal_mask = image_mask | video_mask | audio_mask

        # Replace image id with PAD if the image token if OOV, to avoid index-errors
        llm_input_ids = None
        if inputs_embeds is None:
            llm_input_ids = input_ids.clone()
            llm_input_ids = torch.where(multimodal_mask, self.config.text_config.pad_token_id, llm_input_ids)
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_position_ids, return_dict=True).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

            # Confirm the number of soft tokens from the vision tower matches the number of slots in the embeddings.
            n_image_tokens = image_mask.sum()
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            torch_compilable_check(
                inputs_embeds[image_mask].numel() == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features:"
                f" {image_features.shape[0]}",
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.to(inputs_embeds.device), image_features.to(inputs_embeds.device)
            )

        if pixel_values_videos is not None:
            video_features = self.get_video_features(
                pixel_values_videos, video_position_ids, return_dict=True
            ).pooler_output
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)

            # Confirm the number of soft tokens from the vision tower matches the number of slots in the embeddings.
            n_video_tokens = video_mask.sum()
            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            torch_compilable_check(
                inputs_embeds[video_mask].numel() == video_features.numel(),
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, features:"
                f" {video_features.shape[0]}",
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask.to(inputs_embeds.device), video_features.to(inputs_embeds.device)
            )

        # Merge text and audio
        if input_features is not None and input_features_mask is not None:
            audio_output = self.get_audio_features(input_features, input_features_mask, return_dict=True)
            audio_features = audio_output.pooler_output
            audio_mask_from_encoder = audio_output.attention_mask  # True = valid

            # Strip padding tokens: only keep real (non-padding) audio soft tokens.
            # audio_mask_from_encoder is True for valid positions, False for padding tokens.
            # This mirrors the vision encoder's padding stripping (see Gemma4UnifiedVisionEncoder.forward).
            audio_features = audio_features[audio_mask_from_encoder.to(audio_features.device)]

            n_audio_tokens = audio_mask.sum()
            audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            torch_compilable_check(
                inputs_embeds[audio_mask].numel() == audio_features.numel(),
                f"Audio features and audio tokens do not match, tokens: {n_audio_tokens}, features:"
                f" {audio_features.shape[0] * audio_features.shape[1]}",
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                audio_mask.to(inputs_embeds.device), audio_features.to(inputs_embeds.device)
            )

        # It may already have been prepared by, e.g., `generate`
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config.get_text_config(),
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }

            # Larger Gemma 4 models use Gemma 3's bidirectional attention mask for vision inputs
            # Smaller Gemma models use a conventional casual attention mask
            if self.config.get_text_config().use_bidirectional_attention == "vision":
                block_sequence_ids = torch.full([*inputs_embeds.size()[:-1]], -1, device=inputs_embeds.device)
                if mm_token_type_ids is not None:
                    block_sequence_ids = get_block_sequence_ids_for_mask(
                        mm_token_type_ids, device=inputs_embeds.device
                    )

                mask_kwargs["block_sequence_ids"] = block_sequence_ids

            # Create the masks
            causal_mask_mapping = create_masks_for_generate(**mask_kwargs)

        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )

        return Gemma4UnifiedModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            audio_hidden_states=audio_features if input_features is not None else None,
            shared_kv_states=outputs.shared_kv_states,
        )


class Gemma4UnifiedForConditionalGeneration(Gemma4ForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        video_position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Gemma4UnifiedCausalLMOutputWithPast:
        r"""
        input_features_mask (`torch.FloatTensor]` of shape `(num_images, seq_length)`):
            The attention mask for the input audio.
        image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
            2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        video_position_ids (`torch.LongTensor` of shape `(num_videos, num_frames, max_patches, 2)`, *optional*):
            2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            input_features=input_features,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            image_position_ids=image_position_ids,
            video_position_ids=video_position_ids,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if (final_logit_softcapping := self.config.get_text_config().final_logit_softcapping) is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.get_text_config().vocab_size, **kwargs)

        return Gemma4UnifiedCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            audio_hidden_states=outputs.audio_hidden_states,
            shared_kv_states=outputs.shared_kv_states,
        )

    def get_per_layer_input_embeddings(self):
        raise AttributeError("PLE is not used")

    def set_per_layer_input_embeddings(self, value):
        raise AttributeError("PLE is not used")

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        pixel_values=None,
        pixel_values_videos=None,
        input_features=None,
        attention_mask=None,
        input_features_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # If we're in cached decoding stage, multimodal inputs are already cached and can be dropped
        if is_first_iteration or not use_cache:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["input_features"] = input_features
            model_inputs["input_features_mask"] = input_features_mask
        else:
            # Don't pass to not apply bidirectional mask on top
            model_inputs["mm_token_type_ids"] = None

        return model_inputs


__all__ = [
    "Gemma4UnifiedImageProcessor",
    "Gemma4UnifiedVideoProcessor",
    "Gemma4UnifiedProcessor",
    "Gemma4UnifiedAudioConfig",
    "Gemma4UnifiedConfig",
    "Gemma4UnifiedTextConfig",
    "Gemma4UnifiedVisionConfig",
    "Gemma4UnifiedForCausalLM",
    "Gemma4UnifiedForConditionalGeneration",
    "Gemma4UnifiedModel",
    "Gemma4UnifiedPreTrainedModel",
    "Gemma4UnifiedTextModel",
]
