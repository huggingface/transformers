from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from torchvision.transforms.v2 import functional as TVF
except ImportError:
    TVF = None


import re

from genesis.public.tensorstream.tensor_stream import (
    Event,
    Stream,
    TensorStream,
    TextType,
    VisionType,
    create_stream,
    group_streams,
)
from genesis.public.tensorstream.tensor_stream_utils import (
    compute_mrope_pos_tensor,
    modality_mask,
    reconstruct_tensor_stream_from_compact_dict,
    tensor_stream_token_view,
)
from genesis.public.tensorstream.tensor_stream_utils import (
    slice as ts_slice,
)

from ...cache_utils import Cache, SlidingWindowCache, StaticCache
from ...feature_extraction_utils import BatchFeature
from ...generation.utils import GenerationMixin
from ...image_processing_utils import BaseImageProcessor
from ...image_processing_utils_fast import BaseImageProcessorFast, SizeDict
from ...image_transforms import convert_to_rgb
from ...image_utils import (
    ImageInput,
    PILImageResampling,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...processing_utils import ImagesKwargs, ProcessorMixin, Unpack
from ...tokenization_utils import TensorType
from ...utils import auto_docstring, filter_out_non_signature_kwargs
from ...utils.import_utils import is_torchdynamo_compiling
from ..auto.image_processing_auto import AutoImageProcessor
from ..auto.modeling_auto import AutoModel
from ..auto.tokenization_auto import AutoTokenizer
from ..qwen2.tokenization_qwen2 import Qwen2Tokenizer
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3PreTrainedModel
from ..siglip2.configuration_siglip2 import Siglip2VisionConfig
from ..siglip2.modeling_siglip2 import Siglip2EncoderLayer as HFSiglip2EncoderLayer


# Vision preprocessing constants
VISION_MEAN = (0.5, 0.5, 0.5)
VISION_STD = (0.5, 0.5, 0.5)
VISION_SCALE = 1 / 255




def _normalize_rgb_values(
    values: float | Sequence[float] | tuple[float, ...],
    *,
    name: str,
) -> tuple[float, float, float]:
    """Coerce RGB normalization parameters into a 3-tuple of floats."""
    if isinstance(values, (list, tuple)):
        if len(values) == 3:
            return tuple(float(v) for v in values)
        if len(values) == 1:
            value = float(values[0])
            return (value, value, value)
        raise ValueError(f"`{name}` must have length 1 or 3 when provided as a sequence. Got length {len(values)}.")

    value = float(values)
    return (value, value, value)


def _make_writeable(arr: np.ndarray) -> np.ndarray:
    if arr.flags.writeable:
        return arr
    try:
        arr.setflags(write=True)
        return arr
    except ValueError:
        return arr.copy()



class IsaacVisionConfig(Siglip2VisionConfig):
    """Vision configuration for Isaac with Pixel Shuffle support.

    Extends Siglip2VisionConfig with additional fields for pixel shuffle.

    Args:
        pixel_shuffle_scale_factor (`int`, *optional*, defaults to 1):
            Spatial factor applied before pixel shuffle reduces the resolution.
        num_patches (`int`, *optional*, defaults to 256):
            Maximum number of learnable positional embeddings to initialize.
    """

    model_type = "isaac_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        pixel_shuffle_scale_factor: int = 1,
        num_patches: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Add our custom fields
        self.pixel_shuffle_scale_factor = pixel_shuffle_scale_factor
        self.num_patches = num_patches


class IsaacImageProcessorKwargs(ImagesKwargs):
    patch_size: int | None
    max_num_patches: int | None
    min_num_patches: int | None
    pixel_shuffle_scale: int | None
    do_rescale: bool | None
    rescale_factor: float | None
    do_normalize: bool | None
    image_mean: float | Sequence[float] | None
    image_std: float | Sequence[float] | None
    do_convert_rgb: bool | None


@auto_docstring
class IsaacImageProcessorFast(BaseImageProcessorFast):
    slow_image_processor_class = None
    r"""Fast torch-based image processor for Isaac vision inputs."""

    resample = PILImageResampling.BILINEAR
    model_input_names = ["patches", "token_grids"]
    valid_kwargs = IsaacImageProcessorKwargs
    unused_kwargs = ["size", "do_center_crop", "crop_size"]

    def __init__(
        self,
        *,
        patch_size: int = 16,
        max_num_patches: int = 256,
        min_num_patches: int | None = None,
        pixel_shuffle_scale: int = 1,
        do_rescale: bool = True,
        rescale_factor: float | None = None,
        do_normalize: bool = True,
        image_mean: float | Sequence[float] | None = None,
        image_std: float | Sequence[float] | None = None,
        do_convert_rgb: bool = True,
        **kwargs: Unpack[IsaacImageProcessorKwargs],
    ) -> None:
        super().__init__(**kwargs)

        if pixel_shuffle_scale < 1:
            raise ValueError("`pixel_shuffle_scale` must be >= 1")

        mean_values = _normalize_rgb_values(
            image_mean if image_mean is not None else VISION_MEAN, name="image_mean"
        )
        std_values = _normalize_rgb_values(
            image_std if image_std is not None else VISION_STD, name="image_std"
        )

        self.patch_size = patch_size
        self.max_num_patches = max_num_patches
        self.min_num_patches = min_num_patches
        self.pixel_shuffle_scale = pixel_shuffle_scale
        self.do_rescale = do_rescale
        self.rescale_factor = VISION_SCALE if rescale_factor is None else float(rescale_factor)
        self.do_normalize = do_normalize
        self.image_mean = list(mean_values)
        self.image_std = list(std_values)
        self.do_convert_rgb = do_convert_rgb

    def _validate_preprocess_kwargs(self, **kwargs):
        # Allow callers to omit resize-related placeholders that BaseImageProcessorFast checks for.
        kwargs.pop("do_resize", None)
        return super()._validate_preprocess_kwargs(**kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        patch_size: int,
        max_num_patches: int,
        interpolation: Any | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | Sequence[float] | None,
        image_std: float | Sequence[float] | None,
        return_tensors: str | TensorType | None,
        *,
        min_num_patches: int | None = None,
        pixel_shuffle_scale: int | None = None,
        do_convert_rgb: bool | None = None,
        **kwargs,
    ) -> BatchFeature:
        if TVF is None:
            raise ImportError("torchvision is required for IsaacImageProcessorFast but is not installed.")

        min_num_patches = min_num_patches if min_num_patches is not None else self.min_num_patches
        pixel_shuffle_scale = pixel_shuffle_scale if pixel_shuffle_scale is not None else self.pixel_shuffle_scale
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor

        mean_values = _normalize_rgb_values(
            image_mean if image_mean is not None else self.image_mean, name="image_mean"
        )
        std_values = _normalize_rgb_values(
            image_std if image_std is not None else self.image_std, name="image_std"
        )

        patches_list: list[torch.Tensor] = []
        token_grids: list[torch.Tensor] = []
        virtual_dims: list[list[int]] = []
        real_dims: list[list[int]] = []

        for image in images:
            if image.ndim != 3:
                raise ValueError("Expected channel-first image tensor with shape (C, H, W).")

            channels, original_height, original_width = image.shape
            if do_convert_rgb and channels == 1:
                image = image.repeat(3, 1, 1)
                channels = 3

            if original_height * original_width > MAX_PIXELS:
                raise ValueError(
                    f"Image (w={original_width}, h={original_height}) > MAX=`{MAX_PIXELS}`"
                )

            target_height, target_width = get_image_size_for_max_num_patches(
                original_height,
                original_width,
                patch_size,
                max_num_patches,
                min_num_patches=min_num_patches,
                pixel_shuffle_scale=pixel_shuffle_scale,
            )

            if do_resize:
                size_dict = SizeDict(height=target_height, width=target_width)
                image = self.resize(image=image, size=size_dict, interpolation=interpolation)
            else:
                if ((original_height % patch_size) != 0) or ((original_width % patch_size) != 0):
                    raise ValueError(
                        "Image dimensions must be divisible by patch_size when resize is disabled."
                    )

            # Apply rescaling and normalization as needed
            image = self.rescale_and_normalize(
                image,
                do_rescale,
                rescale_factor,
                do_normalize,
                list(mean_values),
                list(std_values),
            )

            # Convert to NHWC for residual P-frame adjustment and patch extraction
            nhwc_image = image.permute(1, 2, 0).unsqueeze(0)
            nhwc_image = _compute_residual_p_frames(nhwc_image, is_p_frame=[False])

            patches = patchify_vision(nhwc_image, patch_size=patch_size).squeeze(0)
            height_tokens, width_tokens, _ = patches.shape

            patches_list.append(patches.unsqueeze(0))
            token_grids.append(
                torch.tensor([height_tokens, width_tokens], dtype=torch.long, device=patches.device)
            )

            real_dims.append([1, height_tokens, width_tokens])
            if pixel_shuffle_scale > 1:
                if (height_tokens % pixel_shuffle_scale) or (width_tokens % pixel_shuffle_scale):
                    raise ValueError(
                        "Spatial dimensions must be divisible by pixel_shuffle_scale when pixel shuffle is enabled."
                    )
                virtual_dims.append(
                    [1, height_tokens // pixel_shuffle_scale, width_tokens // pixel_shuffle_scale]
                )
            else:
                virtual_dims.append([1, height_tokens, width_tokens])

        patches_tensor = torch.cat(patches_list, dim=0)
        token_grids_tensor = torch.stack(token_grids, dim=0)
        virtual_dims_tensor = torch.tensor(virtual_dims, dtype=torch.long, device=patches_tensor.device)
        real_dims_tensor = torch.tensor(real_dims, dtype=torch.long, device=patches_tensor.device)

        batch_feature = BatchFeature(
            data={
                "patches": patches_tensor,
                "token_grids": token_grids_tensor,
                "virtual_pixel_size": virtual_dims_tensor,
                "real_pixel_size": real_dims_tensor,
            },
            tensor_type=return_tensors,
        )
        return batch_feature




class IsaacImageProcessor(BaseImageProcessor):
    """Image processor that prepares RGB frames for the Isaac vision encoder."""

    model_input_names = ["patches", "token_grids"]

    def __init__(
        self,
        patch_size: int = 16,
        max_num_patches: int = 256,
        min_num_patches: int | None = None,
        pixel_shuffle_scale: int = 1,
        do_rescale: bool = True,
        rescale_factor: float | None = None,
        do_normalize: bool = True,
        image_mean: float | Sequence[float] | None = None,
        image_std: float | Sequence[float] | None = None,
        do_convert_rgb: bool = True,
        resize_mode: str = "bilinear",
        align_corners: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if pixel_shuffle_scale < 1:
            raise ValueError("`pixel_shuffle_scale` must be >= 1")

        rescale_value = VISION_SCALE if rescale_factor is None else float(rescale_factor)
        mean_value = VISION_MEAN if image_mean is None else image_mean
        std_value = VISION_STD if image_std is None else image_std

        self.patch_size = patch_size
        self.max_num_patches = max_num_patches
        self.min_num_patches = min_num_patches
        self.pixel_shuffle_scale = pixel_shuffle_scale
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_value
        self.do_normalize = do_normalize
        self.image_mean = _normalize_rgb_values(mean_value, name="image_mean")
        self.image_std = _normalize_rgb_values(std_value, name="image_std")
        self.do_convert_rgb = do_convert_rgb
        self.resize_mode = resize_mode
        self.align_corners = align_corners

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        patch_size: int | None = None,
        max_num_patches: int | None = None,
        min_num_patches: int | None = None,
        pixel_shuffle_scale: int | None = None,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        do_normalize: bool | None = None,
        image_mean: float | Sequence[float] | None = None,
        image_std: float | Sequence[float] | None = None,
        do_convert_rgb: bool | None = None,
        return_tensors: str | TensorType | None = None,
    ) -> BatchFeature:
        patch_size = patch_size if patch_size is not None else self.patch_size
        max_num_patches = max_num_patches if max_num_patches is not None else self.max_num_patches
        min_num_patches = min_num_patches if min_num_patches is not None else self.min_num_patches
        pixel_shuffle_scale = (
            pixel_shuffle_scale if pixel_shuffle_scale is not None else self.pixel_shuffle_scale
        )
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else _normalize_rgb_values(image_mean, name="image_mean")
        image_std = self.image_std if image_std is None else _normalize_rgb_values(image_std, name="image_std")
        do_convert_rgb = self.do_convert_rgb if do_convert_rgb is None else do_convert_rgb

        images = self.fetch_images(images)
        images = make_flat_list_of_images(images)

        if not images:
            raise ValueError("Received an empty list of images for preprocessing.")
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Expected PIL images, numpy arrays, or tensors convertible to numpy arrays."
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
        )

        patches_list = []
        token_grids = []
        virtual_dims = []
        real_dims = []

        for image in images:
            np_image = to_numpy_array(image)

            if np_image.ndim == 2:
                np_image = np.repeat(np_image[..., None], 3, axis=-1)

            height, width = np_image.shape[:2]
            if height * width > MAX_PIXELS:
                raise ValueError(f"Image (w={width}, h={height}) > MAX=`{MAX_PIXELS}`")

            torch_image = torch.from_numpy(_make_writeable(np_image))
            patches, vidims, rdims = self._process_single_image(
                torch_image,
                patch_size=patch_size,
                max_num_patches=max_num_patches,
                min_num_patches=min_num_patches,
                pixel_shuffle_scale=pixel_shuffle_scale,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
            )

            patches_list.append(patches)
            token_grids.append(torch.tensor([patches.size(1), patches.size(2)], dtype=torch.long))
            virtual_dims.append(vidims)
            real_dims.append(rdims)

        patches_tensor = torch.cat(patches_list, dim=0)
        token_grid_tensor = torch.stack(token_grids, dim=0)
        virtual_dims_tensor = torch.tensor(virtual_dims, dtype=torch.long)
        real_dims_tensor = torch.tensor(real_dims, dtype=torch.long)

        data = {
            "patches": patches_tensor,
            "token_grids": token_grid_tensor,
            "virtual_pixel_size": virtual_dims_tensor,
            "real_pixel_size": real_dims_tensor,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _process_single_image(
        self,
        image: torch.Tensor,
        *,
        patch_size: int,
        max_num_patches: int,
        min_num_patches: int | None,
        pixel_shuffle_scale: int,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: tuple[float, ...],
        image_std: tuple[float, ...],
    ) -> tuple[torch.Tensor, list[int], list[int]]:
        image_uint8 = image.unsqueeze(0)  # (1, H, W, C)
        image_chw = image_uint8.permute(0, 3, 1, 2)  # (1, C, H, W)

        _, _, orig_height, orig_width = image_chw.shape
        target_height, target_width = get_image_size_for_max_num_patches(
            orig_height,
            orig_width,
            patch_size,
            max_num_patches,
            min_num_patches=min_num_patches,
            pixel_shuffle_scale=pixel_shuffle_scale,
        )

        if self.resize_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            resized = F.interpolate(
                image_chw,
                size=(target_height, target_width),
                mode=self.resize_mode,
                align_corners=self.align_corners,
            )
        else:
            resized = F.interpolate(
                image_chw,
                size=(target_height, target_width),
                mode=self.resize_mode,
            )

        resized = resized.permute(0, 2, 3, 1)  # (1, H, W, C)

        scale = rescale_factor if do_rescale else 1.0
        mean = image_mean if do_normalize else (0.0, 0.0, 0.0)
        std = image_std if do_normalize else (1.0, 1.0, 1.0)
        resized = _prepare_image_tensor(resized, scale=scale, mean=mean, std=std)

        resized = _compute_residual_p_frames(resized, is_p_frame=[False])

        patches = patchify_vision(resized, patch_size=patch_size)
        _, h_patches, w_patches, _ = patches.shape

        real_dims = [1, h_patches, w_patches]
        if pixel_shuffle_scale > 1:
            if (h_patches % pixel_shuffle_scale) or (w_patches % pixel_shuffle_scale):
                raise ValueError(
                    "Spatial dimensions must be divisible by pixel_shuffle_scale when pixel shuffle is enabled."
                )
            virtual_dims = [1, h_patches // pixel_shuffle_scale, w_patches // pixel_shuffle_scale]
        else:
            virtual_dims = real_dims.copy()

        return patches, virtual_dims, real_dims


def _max_from_cu(cu: torch.Tensor | None, fallback: int) -> int:
    """Helper to compute max sequence length from cumulative sequence lengths."""
    if cu is None or len(cu) < 2:
        return fallback
    return int((cu[1:] - cu[:-1]).max().item())


def flash_attention_document_mask_forward(
    module: torch.nn.Module,
    q_lhd: torch.Tensor,  # (L, H, D)
    k_lhd: torch.Tensor,  # (L, H, D)
    v_lhd: torch.Tensor,  # (L, H, D)
    attention_mask: torch.Tensor | None = None,  # unused for FA path
    dropout: float = 0.0,
    scaling: float | None = None,
    cum_seq_q: torch.Tensor | None = None,
    cum_seq_k: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    is_causal: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """FlashAttention that consumes (L, H, D) directly to avoid layout churn."""
    L, H, D = q_lhd.shape

    # Compute max block length once (honor caller when provided)
    if max_seqlen is not None:
        max_q = max_k = int(max_seqlen)
    else:
        max_q = _max_from_cu(cum_seq_q, L)
        max_k = _max_from_cu(cum_seq_k, L)

    # Ensure contiguity only if needed
    if not q_lhd.is_contiguous():
        q_lhd = q_lhd.contiguous()
    if not k_lhd.is_contiguous():
        k_lhd = k_lhd.contiguous()
    if not v_lhd.is_contiguous():
        v_lhd = v_lhd.contiguous()

    out_lhd, *_ = torch.ops.aten._flash_attention_forward(
        query=q_lhd,  # (L, H, D)
        key=k_lhd,  # (L, H, D)
        value=v_lhd,  # (L, H, D)
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
        dropout_p=dropout,
        is_causal=is_causal,
        return_debug_mask=False,
        scale=scaling,
        window_size_left=-1,
        window_size_right=-1,
        alibi_slopes=None,
    )
    return out_lhd, None  # (L, H, D)


def sdpa_document_mask_forward(
    q_lhd: torch.Tensor,  # (L, H, D)
    k_lhd: torch.Tensor,  # (L, H, D)
    v_lhd: torch.Tensor,  # (L, H, D)
    dropout: float,
    scaling: float | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """SDPA with block-diagonal masking for variable-length sequences."""
    L, H, D = q_lhd.shape

    # Transpose to (1, H, L, D) format for SDPA
    Q = q_lhd.permute(1, 0, 2).unsqueeze(0)
    K = k_lhd.permute(1, 0, 2).unsqueeze(0)
    V = v_lhd.permute(1, 0, 2).unsqueeze(0)

    # Build block-diagonal mask for variable-length sequences
    attn_mask = None
    if cu_seqlens is not None:
        seq_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        seg_ids = torch.repeat_interleave(torch.arange(len(seq_sizes), device=q_lhd.device), seq_sizes)
        block_mask = seg_ids[:, None] != seg_ids[None, :]  # Cross-document attention blocked
        attn_mask = torch.where(block_mask, -torch.inf, 0.0).to(q_lhd.dtype).view(1, 1, L, L)

    Y = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask, dropout_p=dropout, scale=scaling)
    return Y.squeeze(0).permute(1, 0, 2)  # Back to (L, H, D)


class IsaacVisionEmbeddings(nn.Module):
    def __init__(self, config: IsaacVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def positional_embeddings(self, spatial_shapes: torch.Tensor) -> torch.Tensor:
        # Prepare positional embeddings grid: (1, embed_dim, h, w)
        positional_embeddings = (
            self.position_embedding.weight.reshape(self.position_embedding_size, self.position_embedding_size, -1)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        pos_embeds_list = []
        mode = "bilinear"
        align_corners = False
        for spatial_shape in spatial_shapes:
            height, width = spatial_shape
            # Guard to ensure height and width are positive for torch.compile
            if height > 0 and width > 0:
                resized_pos_embed = F.interpolate(
                    positional_embeddings,
                    size=(height, width),
                    mode=mode,
                    align_corners=align_corners,
                    antialias=True,
                )
                # Reshape from (1, embed_dim, height, width) to (height*width, embed_dim)
                resized_pos_embed = resized_pos_embed.reshape(self.embed_dim, height * width).transpose(0, 1)
            else:
                # Fallback - should never happen in practice
                raise RuntimeError(
                    "Encountered non-positive spatial dimensions while computing positional embeddings."
                )
            pos_embeds_list.append(resized_pos_embed)

        # Concatenate all positional embeddings along the sequence dimension
        pos_embeds = torch.cat(pos_embeds_list, dim=0)
        return pos_embeds

    def forward(self, seq_patches: torch.Tensor, spatial_shapes: torch.Tensor):
        # Apply patch embeddings
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(seq_patches.to(dtype=target_dtype))
        pos_embeds = self.positional_embeddings(spatial_shapes)

        # Add positional embeddings to patch embeddings
        embeddings = patch_embeds + pos_embeds
        return embeddings


class IsaacVisionAttention(nn.Module):
    """Custom attention that supports variable-length sequences with flash attention."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states, cu_seqlens=None, max_seqlen=None):
        # Expect packed sequences with batch_size == 1
        batch_size, L, _ = hidden_states.shape
        if batch_size != 1:
            raise ValueError("packed variable-length attention expects batch_size=1")
        x = hidden_states[0]  # (L, E)

        H = self.num_heads
        D = self.head_dim
        p_drop = self.dropout if self.training else 0.0

        # Project and reshape to (L, H, D)
        q = self.q_proj(x).view(L, H, D)
        k = self.k_proj(x).view(L, H, D)
        v = self.v_proj(x).view(L, H, D)

        attn_impl = getattr(self.config, "_attn_implementation", "flash_attention_3")

        if attn_impl in ("flash_attention_2", "flash_attention_3"):
            y_lhd, _ = flash_attention_document_mask_forward(
                self,
                q,
                k,
                v,
                attention_mask=None,
                dropout=p_drop,
                scaling=self.scale,
                cum_seq_q=cu_seqlens,
                cum_seq_k=cu_seqlens,
                max_seqlen=max_seqlen,
                is_causal=False,
            )
        else:
            y_lhd = sdpa_document_mask_forward(q, k, v, dropout=p_drop, scaling=self.scale, cu_seqlens=cu_seqlens)

        # Merge heads and project
        y = self.out_proj(y_lhd.reshape(L, self.embed_dim))
        return y.unsqueeze(0), None  # (1, L, E)


class IsaacVisionEncoderLayer(HFSiglip2EncoderLayer):
    """Isaac vision encoder layer with variable-length attention."""

    def __init__(self, config: IsaacVisionConfig):
        super().__init__(config)
        self.self_attn = IsaacVisionAttention(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
    ) -> tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


class IsaacVisionEncoder(nn.Module):
    """Encoder using Isaac encoder layers with variable-length attention support."""

    def __init__(self, config: IsaacVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([IsaacVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        output_hidden_states: bool = False,
    ):
        all_hidden_states = () if output_hidden_states else None

        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
                max_seqlen,
            )

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states, None


def create_pixel_shuffle_index_map(
    seq_sizes: torch.Tensor,
    token_grids: torch.Tensor,
    scale_factor: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build a gather-index map that tells us, for every *output* token after
    pixel-shuffle, which `scale_factor**2` *input* tokens are being merged.

    Args
    ----
    seq_sizes     : (num_images,)  - #patches in each image (row-major order)
    token_grids   : (num_images,2) - (height, width) for every image
    scale_factor  : spatial down-scale factor (≥2)
    device        : (optional) overrides `seq_sizes.device`

    Returns
    -------
    gather_idx : (new_total_seq_len, scale_factor**2) int64 tensor.
                 gather_idx[i, j] is the *flat* index into the *original*
                 packed sequence for the j-th sub-patch that forms the
                 i-th output token.
    """
    if device is None:
        device = seq_sizes.device

    scale_factor = int(scale_factor)
    if scale_factor < 2:
        raise ValueError("`scale_factor` must be ≥ 2")

    # Safety: all spatial dims must be divisible by the scale factor
    # Cannot run under torch compile fullgraph mode hence
    if not is_torchdynamo_compiling():
        if not (
            (token_grids[:, 0] % scale_factor == 0).all() and (token_grids[:, 1] % scale_factor == 0).all()
        ):
            raise AssertionError(
                "Every (H,W) in `token_grids` must be divisible by "
                f"scale_factor={scale_factor}, got {token_grids.tolist()}"
            )

    gather_chunks: list[torch.Tensor] = []
    tok_offset = 0

    for seq_len, (h, w) in zip(seq_sizes.tolist(), token_grids.tolist(), strict=False):
        # Build the (H, W) grid of flat indices for this image
        grid = torch.arange(seq_len, device=device, dtype=torch.int64) + tok_offset
        grid = grid.view(h, w)  # (H, W)

        # -------- identical ordering to your fixed-res routine --------
        # Step 1: split width into blocks of scale_factor
        grid = grid.view(h, w // scale_factor, scale_factor)  # (H, W/scale_factor, scale_factor)
        # Step 2: now split height into blocks of scale_factor
        grid = grid.view(h // scale_factor, scale_factor, w // scale_factor, scale_factor)
        # (H/scale_factor, scale_factor, W/scale_factor, scale_factor)
        # Step 3: final permutation to (H/scale_factor, W/scale_factor, scale_factor, scale_factor)
        grid = grid.permute(0, 2, 1, 3).contiguous()  # (H/scale_factor, W/scale_factor, scale_factor, scale_factor)
        # Step 4: each (scale_factor, scale_factor) block forms one output token
        gather_chunks.append(grid.reshape(-1, scale_factor * scale_factor))
        # (H*W / scale_factor**2, scale_factor**2)

        tok_offset += seq_len

    # Concatenate over all images in the packed batch
    gather_idx = torch.cat(gather_chunks, dim=0)  # (Σ_i HᵢWᵢ/scale_factor**2, scale_factor**2)
    return gather_idx


def pixel_shuffle_varlen(
    x: torch.Tensor,
    token_grids: torch.Tensor,
    scale_factor: int = 1,
) -> torch.Tensor:
    r"""Apply pixel shuffle to a packed vision sequence without unpacking per image.

    Args:
        x (`torch.Tensor`):
            Concatenated vision embeddings. Accepts `(seq_len, hidden_size)` or `(1, seq_len, hidden_size)` shapes
            produced by stacking image patches.
        token_grids (`torch.Tensor`):
            Integer tensor of shape `(num_images, 2)` whose rows give the `(height, width)` patch grid sizes
            corresponding to each image segment inside `x`.
        scale_factor (`int`, *optional*, defaults to 1):
            Spatial down-sampling factor specific to pixel shuffle. Values greater than one merge `scale_factor**2` neighboring patches into a
            single embedding channel-group.

    Returns:
        `torch.Tensor`: Pixel-shuffled embeddings with shape matching the input convention:
        `(seq_len, hidden_size * scale_factor**2)` when the input was 2D, or `(1, seq_len, hidden_size * scale_factor**2)`
        if the singleton batch dimension was present.

    Raises:
        ValueError: If more than one batch item is provided.
    """
    keep_batch_dim = x.dim() == 3
    if keep_batch_dim:
        if x.size(0) != 1:
            raise AssertionError("Packed sequence is expected to have batch_size == 1")
        x_ = x.squeeze(0)  # (seq, embed)
    else:
        x_ = x  # (seq, embed)

    embed_dim = x_.size(-1)
    scale_factor = int(scale_factor)

    # Calculate seq_sizes from token_grids
    seq_sizes = torch.prod(token_grids, dim=-1)

    # Build index map and gather in one go
    gather_idx = create_pixel_shuffle_index_map(
        seq_sizes=seq_sizes,
        token_grids=token_grids,
        scale_factor=scale_factor,
        device=x_.device,
    )  # (new_seq, scale_factor**2)

    # Gather → (new_seq, scale_factor**2, embed_dim)
    gathered = x_[gather_idx]  # fancy indexing keeps gradient

    # Merge the scale_factor**2 group dimension into channels to finish the shuffle
    out = gathered.reshape(gathered.size(0), embed_dim * scale_factor * scale_factor)

    # Restore batch dimension if needed
    if keep_batch_dim:
        out = out.unsqueeze(0)
    return out


class IsaacVisionTransformer(nn.Module):
    def __init__(self, config: IsaacVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = IsaacVisionEmbeddings(config)
        self.encoder = IsaacVisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pixel_shuffle_scale_factor = config.pixel_shuffle_scale_factor

    def forward(self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor]):
        seq_patches, token_grids = packed_seq_patches
        seq_sizes = torch.prod(token_grids, dim=-1)

        # Get embeddings from packed sequence
        hidden_states = self.embeddings(seq_patches, token_grids)

        # Add a pseudo batch dimension for the encoder
        hidden_states = hidden_states.unsqueeze(0)

        # Generate cumulative sequence lengths for variable-length attention
        cu_seqlens = torch.zeros(seq_sizes.size(0) + 1, dtype=torch.int32, device=hidden_states.device)
        cu_seqlens[1:] = seq_sizes.cumsum(0)
        max_seqlen = int(seq_sizes.max().item()) if seq_sizes.numel() > 0 else 0

        # Pass through encoder with variable-length attention parameters
        hidden_states, _, _ = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # Apply final layer normalization
        hidden_states = self.post_layernorm(hidden_states)

        if self.pixel_shuffle_scale_factor > 1:
            hidden_states = pixel_shuffle_varlen(
                x=hidden_states,
                token_grids=token_grids,
                scale_factor=self.pixel_shuffle_scale_factor,
            )
        # Remove the pseudo batch dimension we added earlier
        hidden_states = hidden_states.squeeze(0)

        # Return the full sequence of embeddings
        return hidden_states


# ============================================================================
# Configuration
# ============================================================================

MAX_PIXELS = 60_000_000  # 60‑megapixel ceiling ≈ 8200 × 7300 px

def get_scaled_image_size(
    scale: float,
    original_size: int,
    patch_size: int,
    pixel_shuffle_scale: int,
) -> int:
    scaled_size = scale * original_size
    divisor = patch_size * pixel_shuffle_scale
    scaled_size = math.ceil(scaled_size / divisor) * divisor
    scaled_size = max(divisor, scaled_size)
    return int(scaled_size)


def get_image_size_for_max_num_patches(
    image_height: int,
    image_width: int,
    patch_size: int,
    max_num_patches: int,
    min_num_patches: int | None = None,
    eps: float = 1e-5,
    pixel_shuffle_scale: int = 1,
) -> tuple[int, int]:
    r"""Compute a target resolution whose patch grid satisfies patching parametrization.

    Args:
        image_height (`int`):
            Height in pixels of the source image prior to any resizing.
        image_width (`int`):
            Width in pixels of the source image prior to any resizing.
        patch_size (`int`):
            Size of the square patch used by the vision encoder.
        max_num_patches (`int`):
            Upper bound on `(height / patch_size) * (width / patch_size)` after resizing.
        min_num_patches (`int`, *optional*):
            Lower bound on the number of patches. When provided the image will be scaled up if necessary.
        eps (`float`, *optional*, defaults to 1e-5):
            Convergence tolerance for the internal binary search to determing the target dimensions.
        pixel_shuffle_scale (`int`, *optional*, defaults to 1):
            Additional stride multiplier applied when pixel shuffle later reduces spatial resolution.

    Returns:
        `tuple[int, int]`: Height and width (in pixels) that are multiples of `patch_size * pixel_shuffle_scale`
        and respect both the maximum and optional minimum patch-count constraints.
    """

    # Ensure divisibility
    divisor = patch_size * pixel_shuffle_scale
    adjusted_height = math.ceil(image_height / divisor) * divisor
    adjusted_height = max(divisor, adjusted_height)
    adjusted_width = math.ceil(image_width / divisor) * divisor
    adjusted_width = max(divisor, adjusted_width)

    num_patches = (adjusted_height / patch_size) * (adjusted_width / patch_size)

    if min_num_patches is not None and num_patches < min_num_patches:
        # Scale up
        scale_min, scale_max = 1.0, 100.0
        while (scale_max - scale_min) >= eps:
            scale = (scale_min + scale_max) / 2
            target_height = get_scaled_image_size(scale, image_height, patch_size, pixel_shuffle_scale)
            target_width = get_scaled_image_size(scale, image_width, patch_size, pixel_shuffle_scale)
            num_patches = (target_height / patch_size) * (target_width / patch_size)
            if num_patches >= min_num_patches:
                scale_max = scale
            else:
                scale_min = scale
        scale = scale_max
        target_height = get_scaled_image_size(scale, image_height, patch_size, pixel_shuffle_scale)
        target_width = get_scaled_image_size(scale, image_width, patch_size, pixel_shuffle_scale)
        return target_height, target_width
    elif num_patches <= max_num_patches:
        return adjusted_height, adjusted_width
    else:
        # Scale down
        scale_min, scale_max = eps / 10, 1.0
        while (scale_max - scale_min) >= eps:
            scale = (scale_min + scale_max) / 2
            target_height = get_scaled_image_size(scale, image_height, patch_size, pixel_shuffle_scale)
            target_width = get_scaled_image_size(scale, image_width, patch_size, pixel_shuffle_scale)
            num_patches = (target_height / patch_size) * (target_width / patch_size)
            if num_patches <= max_num_patches:
                scale_min = scale
            else:
                scale_max = scale
        scale = scale_min
        target_height = get_scaled_image_size(scale, image_height, patch_size, pixel_shuffle_scale)
        target_width = get_scaled_image_size(scale, image_width, patch_size, pixel_shuffle_scale)
        return target_height, target_width


def patchify_vision(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    r"""Convert normalized images into flattened ViT-style patches.

    Args:
        image (`torch.Tensor`):
            Tensor of shape `(num_images, height, width, channels)`.
        patch_size (`int`):
            Edge length of the square patches

    Returns:
        `torch.Tensor`:
            Patch tensor where each position stores the flattened pixels belonging to that patch.

    Raises:
        ValueError: If `height` or `width` is not divisible by `patch_size`.
    """
    num_images, height, width, channels = image.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"Dimensions of images {image.shape} are not divisible by patch_size={patch_size}.")
    patches = image.reshape(num_images, height // patch_size, patch_size, width // patch_size, patch_size, channels)
    patches = patches.permute(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(num_images, height // patch_size, width // patch_size, channels * patch_size * patch_size)
    return patches


def precompute_cos_sin_3d(
    position_ids: torch.Tensor,  # shape (3, B, T)
    inv_freq: torch.Tensor,  # shape (dim//2,)
    mrope_half_section: list[int],  # sum to dim//2
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Generate 3D rotary embeddings for multi-axis positions.

    Args:
        position_ids (`torch.Tensor`):
            Tensor of shape `(3, batch_size, seq_len)` containing positional indices for the x/y/t axes.
        inv_freq (`torch.Tensor`):
            Precomputed inverse frequency vector used to derive rotary phases.
        mrope_half_section (`list[int]`):
            Sizes the axis-specific frequency blocks.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: Cosine and sine tensors, each of shape `(batch_size, seq_len, dim)`, ready
        to be passed into rotary attention layers.
    """
    B = position_ids.shape[1]
    T = position_ids.shape[2]
    dim_half = inv_freq.shape[0]
    device = position_ids.device

    # Initialize with full dimension (not half) to match LLaMA
    cos_3d = torch.zeros((B, T, dim_half * 2), dtype=torch.float32, device=device)
    sin_3d = torch.zeros((B, T, dim_half * 2), dtype=torch.float32, device=device)

    offset = 0
    for d in range(3):
        block_size = mrope_half_section[d]
        freq_slice = inv_freq[offset : offset + block_size]  # shape => (block_size,)
        # shape => (B, T, block_size)
        phase = position_ids[d].unsqueeze(-1).float() * freq_slice

        cos_part = phase.cos()
        sin_part = phase.sin()

        # Duplicate values for both halves of the dimension
        cos_3d[:, :, offset : offset + block_size] = cos_part
        cos_3d[:, :, dim_half + offset : dim_half + offset + block_size] = cos_part
        sin_3d[:, :, offset : offset + block_size] = sin_part
        sin_3d[:, :, dim_half + offset : dim_half + offset + block_size] = sin_part

        offset += block_size

    return cos_3d, sin_3d


class RopeScaling(TypedDict, total=False):
    rope_type: str
    factor: float
    mrope_section: list[int]
    mrope_interleaved: bool
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


class IsaacConfig(Qwen3Config):
    """Configuration class for Isaac multimodal model."""

    model_type = "isaac"
    sub_configs = {"vision_config": IsaacVisionConfig, "text_config": Qwen3Config}
    image_processor_type = "IsaacImageProcessor"

    def __init__(
        self,
        vision_config=None,
        text_config: Qwen3Config | dict | None = None,
        vision_patch_size: int = 16,
        vision_max_num_patches: int = 256,
        vision_min_num_patches: int | None = None,
        pixel_shuffle_scale: int = 1,
        vision_rescale_factor: float = VISION_SCALE,
        vision_mean: float | Sequence[float] = VISION_MEAN,
        vision_std: float | Sequence[float] = VISION_STD,
        max_sequence_length: int = 16384,
        vision_token: str = "<image>",
        vision_attn_implementation: str | None = None,
        **kwargs,
    ):
        resolved_text_config = kwargs.pop("text_config", text_config)
        if isinstance(resolved_text_config, Qwen3Config):
            text_config_kwargs = copy.deepcopy(resolved_text_config.to_dict())
        elif isinstance(resolved_text_config, dict):
            text_config_kwargs = copy.deepcopy(resolved_text_config)
        elif resolved_text_config is None:
            text_config_kwargs = {}
        else:
            raise TypeError("`text_config` must be a mapping or `Qwen3Config` instance when provided.")

        text_config_kwargs.update(kwargs)

        super().__init__(**text_config_kwargs)
        self.text_config = Qwen3Config(**text_config_kwargs)

        # Handle vision config - either dict or IsaacVisionConfig instance
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        # EventStreamProcessor parameters (for backward compatibility)
        self.video_patch_size = vision_patch_size
        self.vision_max_num_patches = vision_max_num_patches
        self.vision_min_num_patches = vision_min_num_patches
        self.pixel_shuffle_scale = pixel_shuffle_scale

        # Vision normalization parameters
        self.vision_rescale_factor = float(vision_rescale_factor)
        self.vision_mean = _normalize_rgb_values(vision_mean, name="vision_mean")
        self.vision_std = _normalize_rgb_values(vision_std, name="vision_std")

        # Processing parameters
        self.max_sequence_length = max_sequence_length
        self.vision_token = vision_token
        self.vision_attn_implementation = vision_attn_implementation

    def get_text_config(self, *_, **kwargs) -> Qwen3Config:
        # Accept optional decoder/encoder flags to align with HF composite configs
        kwargs.pop("decoder", None)
        kwargs.pop("encoder", None)
        return self.text_config


# ============================================================================
# Processor Components
# ============================================================================


def create_text_event(tokenizer: AutoTokenizer, text: str, time: float = 0.0) -> Event:
    r"""Wrap a text into an `Event` compatible with the multimodal TensorStream.

    Args:
        tokenizer (`AutoTokenizer`):
            Tokenizer used to convert text into model vocabulary ids.
        text (`str`):
            Plain-text fragment to encode.
        time (`float`, *optional*, defaults to 0.0):
            Timeline coordinate associated with the event. Both start and end times use the same value because text
            segments are instantaneous in the scheduler.

    Returns:
        `Event`: Event carrying a `(num_tokens, 1)` tensor of token ids with matching
        metadata so that downstream processors can compute modality-specific embeddings.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").squeeze(0)

    # Calculate dimensions for the event
    num_tokens = len(tokens)
    dims_virtual = [num_tokens, 1]  # [sequence_length, 1]
    dims_real = dims_virtual.copy()

    # Ensure tokens has the right shape for tensor_stream_token_view
    # It expects a 2D tensor where sum(dim=-1) gives the token IDs
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(-1)

    return Event(
        data=tokens,
        type=TextType.text,
        time=(time, time),
        dims_virtual=dims_virtual,
        dims_real=dims_real,
        idx_range=(0, num_tokens),
    )


# ============================================================================
# Processor
# ============================================================================


IsaacImageProcessorFast.slow_image_processor_class = IsaacImageProcessor



class IsaacProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = ("IsaacImageProcessor", "IsaacImageProcessorFast")
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self,
        image_processor: IsaacImageProcessor | IsaacImageProcessorFast | None = None,
        tokenizer: Qwen2Tokenizer | None = None,
        *,
        vision_token: str = "<image>",
        max_sequence_length: int = 16384,
        vision_patch_size: int = 16,
        vision_max_num_patches: int = 256,
        vision_min_num_patches: int | None = None,
        pixel_shuffle_scale: int = 1,
        rescale_factor: float | None = None,
        image_mean: float | Sequence[float] | None = None,
        image_std: float | Sequence[float] | None = None,
        vision_attn_implementation: str | None = None,
        config: IsaacConfig | dict | None = None,
        **kwargs,
    ) -> None:
        if tokenizer is None:
            raise ValueError("`tokenizer` must be provided to initialize IsaacProcessor.")

        if isinstance(config, dict):
            config = IsaacConfig(**config)

        if config is not None:
            vision_patch_size = config.video_patch_size
            vision_max_num_patches = config.vision_max_num_patches
            vision_min_num_patches = config.vision_min_num_patches
            pixel_shuffle_scale = config.pixel_shuffle_scale
            max_sequence_length = config.max_sequence_length
            vision_token = config.vision_token
            vision_attn_implementation = config.vision_attn_implementation
            rescale_factor = config.vision_rescale_factor
            image_mean = tuple(config.vision_mean)
            image_std = tuple(config.vision_std)

        resolved_rescale_factor = (
            float(rescale_factor) if rescale_factor is not None else float(VISION_SCALE)
        )
        resolved_image_mean = _normalize_rgb_values(
            image_mean if image_mean is not None else VISION_MEAN,
            name="image_mean",
        )
        resolved_image_std = _normalize_rgb_values(
            image_std if image_std is not None else VISION_STD,
            name="image_std",
        )

        if image_processor is None:
            image_processor = IsaacImageProcessor(
                patch_size=vision_patch_size,
                max_num_patches=vision_max_num_patches,
                min_num_patches=vision_min_num_patches,
                pixel_shuffle_scale=pixel_shuffle_scale,
                rescale_factor=resolved_rescale_factor,
                image_mean=resolved_image_mean,
                image_std=resolved_image_std,
            )
        else:
            vision_patch_size = getattr(image_processor, "patch_size", vision_patch_size)
            vision_max_num_patches = getattr(image_processor, "max_num_patches", vision_max_num_patches)
            vision_min_num_patches = getattr(image_processor, "min_num_patches", vision_min_num_patches)
            pixel_shuffle_scale = getattr(image_processor, "pixel_shuffle_scale", pixel_shuffle_scale)
            resolved_rescale_factor = getattr(image_processor, "rescale_factor", resolved_rescale_factor)
            resolved_image_mean = _normalize_rgb_values(
                getattr(image_processor, "image_mean", resolved_image_mean),
                name="image_mean",
            )
            resolved_image_std = _normalize_rgb_values(
                getattr(image_processor, "image_std", resolved_image_std),
                name="image_std",
            )

        if config is not None:
            config.vision_rescale_factor = resolved_rescale_factor
            config.vision_mean = resolved_image_mean
            config.vision_std = resolved_image_std

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self.config = config

        # Mirror tokenizer chat template so ProcessorMixin.apply_chat_template works.
        self.chat_template = getattr(self.tokenizer, "chat_template", None)

        self.vision_token = vision_token
        self.max_sequence_length = max_sequence_length
        self.vision_attn_implementation = vision_attn_implementation

        self.patch_size = getattr(self.image_processor, "patch_size", vision_patch_size)
        self.max_num_patches = getattr(self.image_processor, "max_num_patches", vision_max_num_patches)
        self.min_num_patches = getattr(self.image_processor, "min_num_patches", vision_min_num_patches)
        self.pixel_shuffle_scale = getattr(self.image_processor, "pixel_shuffle_scale", pixel_shuffle_scale)
        self.rescale_factor = getattr(self.image_processor, "rescale_factor", resolved_rescale_factor)
        self.image_mean = tuple(getattr(self.image_processor, "image_mean", resolved_image_mean))
        self.image_std = tuple(getattr(self.image_processor, "image_std", resolved_image_std))

    def build_event_stream_simple(
        self,
        text: str,
        images: list[PIL.Image.Image] | None = None,
    ) -> Stream:
        events = []
        # Process text and images
        # Find all occurrences of vision token

        pattern = re.escape(self.vision_token)
        parts = re.split(f"({pattern})", text)  # Keep the delimiter in the result

        image_idx = 0
        for current_time, part in enumerate(parts):
            if part == self.vision_token:
                # Replace vision token with image event
                if images is None or image_idx >= len(images):
                    raise ValueError("Encountered vision token without a corresponding image.")

                features = self.image_processor(
                    images=images[image_idx],
                    return_tensors=TensorType.PYTORCH,
                )

                patches = features["patches"][0]  # (H_tokens, W_tokens, embed)
                virtual_dims = features["virtual_pixel_size"][0].tolist()
                real_dims = features["real_pixel_size"][0].tolist()

                vision_event = Event(
                    data=patches.reshape(-1, patches.shape[-1]),
                    type=VisionType.image,
                    time=(current_time, current_time),
                    dims_virtual=virtual_dims,
                    dims_real=real_dims,
                    idx_range=(0, math.prod(virtual_dims)),
                )
                events.append(vision_event)
                image_idx += 1
            elif part:  # Non-empty text part
                # tokens = self.text_processor.tokenize(part, add_special_tokens=False)
                text_event = create_text_event(self.tokenizer, part, time=current_time)
                events.append(text_event)

        # Create stream without scheduling (events already in order)
        return create_stream(events, priority=[TextType.text, VisionType.image], schedule=True)

    def __call__(
        self,
        text: str | list[str],
        images: PIL.Image.Image | list[PIL.Image.Image] | None = None,
        return_tensors: str | TensorType | None = TensorType.PYTORCH,
        **kwargs,
    ) -> BatchFeature:
        """
        Process text and images into TensorStream format.
        Args:
            text: Input text or list of texts with vision tokens
            images: PIL image or list of images (optional)
            return_tensors: Format for output tensors

        Returns:
            BatchFeature with input_ids and tensor_stream
        """
        # Normalize inputs to lists
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        if images is not None:
            if isinstance(images, PIL.Image.Image):
                images_list = [images]
            else:
                images_list = images
        else:
            images_list = None

        if len(texts) != 1:
            raise ValueError("IsaacProcessor currently supports batch_size=1")
        if images_list is not None:
            # Count vision tokens in text to validate image count
            vision_token_count = texts[0].count(self.vision_token)
            if vision_token_count != len(images_list):
                raise ValueError(
                    f"Number of {self.vision_token} tokens in text ({vision_token_count}) "
                    f"must match number of images ({len(images_list)})"
                )

        # Build event stream
        stream = self.build_event_stream_simple(
            text=texts[0],
            images=images_list,
        )

        # Create TensorStream
        tensor_stream = TensorStream([stream])

        # Slice to max length if needed
        _, T = tensor_stream.shape
        if T > self.max_sequence_length:
            tensor_stream = ts_slice(tensor_stream, start=T - self.max_sequence_length, end=T)

        # Get token view
        tokens = tensor_stream_token_view(tensor_stream)
        if return_tensors in (TensorType.PYTORCH, "pt"):
            input_ids = torch.as_tensor(tokens, dtype=torch.long)
        else:
            input_ids = tokens

        data = {
            "input_ids": input_ids,
            "tensor_stream": tensor_stream,
        }

        return BatchFeature(data=data)


# ============================================================================
# Model
# ============================================================================


def compute_position_ids_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    r"""Create 3D positional indices for token input.

    Args:
        input_ids (`torch.Tensor`):
            Tensor of shape `(batch_size, seq_len)` containing token ids.

    Returns:
        `torch.Tensor`: Positional indices with shape `(batch_size, seq_len, 3)` where each channel duplicates the
        1D position so it can be consumed by the 3-axis MRoPE rotary embedding.
    """
    batch_size, seq_length = input_ids.shape
    position_ids = torch.arange(seq_length, device=input_ids.device)
    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
    position_ids = position_ids.unsqueeze(2).expand(-1, -1, 3)  # Add 3D for MRoPE
    return position_ids


class IsaacRotaryEmbedding(nn.Module):
    EXTRA_ROPE_KEYS = {"mrope_section", "mrope_interleaved"}

    def __init__(self, config: IsaacConfig, device=None):
        super().__init__()

        self.config = config
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        if rope_type not in ROPE_INIT_FUNCTIONS:
            raise ValueError(f"Unsupported rope_type '{rope_type}' for IsaacRotaryEmbedding")

        self.rope_type = rope_type
        rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]

        sanitized_scaling = {k: v for k, v in rope_scaling.items() if k not in self.EXTRA_ROPE_KEYS}
        if sanitized_scaling != rope_scaling:
            config_for_rope = copy.copy(config)
            config_for_rope.rope_scaling = sanitized_scaling
        else:
            config_for_rope = config

        init_device = device if device is not None and getattr(device, "type", None) != "meta" else None
        inv_freq, attention_scaling = rope_init_fn(config_for_rope, device=init_device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = self._normalize_scale(attention_scaling)

        rotary_half_dim = self.inv_freq.shape[0]
        self.mrope_section = self._resolve_mrope_section(rope_scaling.get("mrope_section"), rotary_half_dim)

    @staticmethod
    def _normalize_scale(scale: torch.Tensor | float) -> torch.Tensor | float:
        if isinstance(scale, torch.Tensor):
            return scale.detach().clone()
        return float(scale)

    @staticmethod
    def _resolve_mrope_section(section: list[int] | None, rotary_half_dim: int) -> list[int]:
        if section is None:
            weights = (2, 1, 1)
            base = [rotary_half_dim * w // sum(weights) for w in weights]
            base[0] += rotary_half_dim - sum(base)
            return base

        section = [int(v) for v in section]
        if len(section) != 3:
            raise ValueError("`mrope_section` must contain exactly three elements (temporal, height, width)")
        if sum(section) != rotary_half_dim:
            raise ValueError(
                f"`mrope_section` must sum to the rotary half-dimension ({rotary_half_dim}). Received {section}."
            )
        return section

    def forward(self, position_ids: torch.Tensor, modality_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            position_ids = position_ids.clone()
            not_spatial = modality_tensor != VisionType.image.value
            if not_spatial.any():
                data_1d = position_ids[not_spatial][..., 0].unsqueeze(-1)
                position_ids[not_spatial] = data_1d.expand(-1, position_ids.shape[-1])

            position_ids = position_ids.permute(2, 0, 1)
            cos, sin = precompute_cos_sin_3d(position_ids, self.inv_freq, self.mrope_section)
            scale = self.attention_scaling
            if isinstance(scale, torch.Tensor):
                scale = scale.to(device=cos.device, dtype=cos.dtype)
            elif scale != 1.0:
                scale = cos.new_tensor(scale)
            if isinstance(scale, torch.Tensor) or scale != 1.0:
                cos = cos * scale
                sin = sin * scale

        return cos, sin


class IsaacModel(Qwen3PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config: IsaacConfig):
        super().__init__(config)

        text_cfg_source = getattr(config, "get_text_config", lambda: config)()
        text_cfg = copy.deepcopy(text_cfg_source)
        text_cfg._attn_implementation = config._attn_implementation
        self.text_model = AutoModel.from_config(text_cfg)
        # Ensure downstream callers observe the composed config
        self.text_model.config = config

        self.rotary_emb = IsaacRotaryEmbedding(config, device=self.device)

        vision_cfg = config.vision_config
        # Use vision_attn_implementation if specified, otherwise fall back to general attn_implementation
        vision_cfg._attn_implementation = (
            config.vision_attn_implementation
            if config.vision_attn_implementation is not None
            else config._attn_implementation
        )
        if vision_cfg is None:
            raise ValueError("IsaacConfig should always have vision_config")

        hidden_dim = vision_cfg.hidden_size * (vision_cfg.pixel_shuffle_scale_factor**2)
        self.vision_embedding = nn.Sequential(
            IsaacVisionTransformer(vision_cfg),
            nn.Linear(
                hidden_dim,
                4 * hidden_dim,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, config.hidden_size, bias=False),
        )

        # Dispatch table for TensorStream balanced embedding (text + vision)
        self.embed_fns = {
            TextType: self.embed_text_tokens,
            VisionType: self.embed_vision,
        }

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.text_model.set_input_embeddings(value)

    @property
    def embed_tokens(self) -> nn.Module:
        return self.text_model.embed_tokens

    @embed_tokens.setter
    def embed_tokens(self, value: nn.Module) -> None:
        self.text_model.embed_tokens = value

    @property
    def layers(self) -> nn.ModuleList:
        return self.text_model.layers

    @property
    def norm(self) -> nn.Module:
        return self.text_model.norm

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func=None):
        self.text_model._set_gradient_checkpointing(
            enable=enable, gradient_checkpointing_func=gradient_checkpointing_func
        )

    def embed_text_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed text tokens, squeezing singleton dimensions."""
        # Text events are shaped as (..., 1); squeeze the singleton index dim
        h = self.text_model.embed_tokens(token_ids)
        if h.dim() >= 2 and h.size(-2) == 1:
            h = h[..., 0, :]
        return h

    def embed_vision(self, vision_tokens: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Embed vision tokens using the vision encoder."""
        # vision tokens is (seq_patches, token_grids)
        return self.vision_embedding(vision_tokens)

    def embed_stream(self, tensor_stream: TensorStream) -> torch.Tensor:
        """
        Embed each modality stream independently, preserving the original TensorStream
        structure.
        """
        flat_stream = tensor_stream.flat_stream()
        per_modality_stream = group_streams(flat_stream, group_fn=lambda ev: ev.type, schedule=False)
        per_modality_compact_stream = {k: v.compact() for k, v in per_modality_stream.items()}

        # Collect per-event grids for vision tokens (H, W like dims sans time)
        token_grids = defaultdict(list)
        for stream in tensor_stream.streams:
            for event in stream:
                token_grids[event.type].append(event.dims(virtual=False))

        embedded_compact = {}
        for stream_type, modality_payload_tensor in per_modality_compact_stream.items():
            if stream_type.modality == VisionType:
                # Build a (N_events, 2) grid tensor with spatial dims only
                grids = token_grids.get(stream_type, [])
                if len(grids) == 0:
                    input_tensor = modality_payload_tensor
                else:
                    token_grids_tensor = torch.tensor(grids, dtype=torch.long, device=tensor_stream.device)[:, 1:]
                    input_tensor = (modality_payload_tensor, token_grids_tensor)
                embedded_compact[stream_type] = self.embed_fns[stream_type.modality](input_tensor)
            else:
                embedded_compact[stream_type] = self.embed_fns[stream_type.modality](modality_payload_tensor)

        # Reconstruct a TensorStream with embedded payloads and compact
        embedded_ts = reconstruct_tensor_stream_from_compact_dict(tensor_stream, embedded_compact)
        h = embedded_ts.compact()  # (B, T, D)
        return h

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        tensor_stream: TensorStream | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        modality_tensor: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        """
        Forward pass with MRoPE position embeddings.

        Computes position embeddings once and passes them through all layers.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get inputs
        if tensor_stream is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both tensor_stream and inputs_embeds")
        elif tensor_stream is not None:
            # Embed TensorStream directly
            inputs_embeds = self.embed_stream(tensor_stream)
            # Create modality tensor if not provided
            if modality_tensor is None:
                modality_tensor = modality_mask(tensor_stream)
        elif input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            inputs_embeds = self.text_model.embed_tokens(input_ids)
            # Create text modality tensor if not provided
            if modality_tensor is None:
                batch_size, seq_length = input_ids.shape
                modality_tensor = torch.full(
                    (batch_size, seq_length), TextType.text.value, device=input_ids.device, dtype=torch.long
                )
        elif inputs_embeds is None:
            raise ValueError("You have to specify either tensor_stream, input_ids or inputs_embeds")

        # Create default position_ids if not provided
        if position_ids is None:
            if tensor_stream is not None:
                position_ids = compute_mrope_pos_tensor(tensor_stream)  # (B,L,3)
            else:
                position_ids = compute_position_ids_input_ids(input_ids)

        # Compute MRoPE position embeddings if we have custom rotary_emb
        cos, sin = self.rotary_emb(position_ids, modality_tensor)
        cos = cos.to(inputs_embeds.dtype)
        sin = sin.to(inputs_embeds.dtype)

        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, False
            )

        # Initialize hidden states
        hidden_states = inputs_embeds

        for decoder_layer in self.text_model.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=(cos, sin),
                **kwargs,
            )

            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        # Final layer norm
        hidden_states = self.text_model.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen3. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen3Config,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen3Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class IsaacForConditionalGeneration(Qwen3ForCausalLM, GenerationMixin):
    """Isaac multimodal model for conditional generation."""

    config_class = IsaacConfig

    def __init__(self, config: IsaacConfig):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = IsaacModel(config)  # Use our custom model
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Tracks rotary position offsets computed during a full forward pass so decode steps can reuse them.
        self.rope_deltas = None

        self.config = config

    def get_rope_index(
        self,
        input_ids: torch.Tensor | None,
        tensor_stream: TensorStream | None,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MRoPE position ids from a TensorStream (or 1D fallback).

        Returns (position_ids, rope_deltas). position_ids is (B,L,3) for MRoPE.
        rope_deltas is (B,1) used to advance positions in decode.
        """
        # tensor_stream present: compute 3D coords
        if tensor_stream is None and input_ids is None:
            raise ValueError("`tensor_stream` or `input_ids` must be provided to compute rope indices")

        if tensor_stream is not None:
            pos_3d = compute_mrope_pos_tensor(tensor_stream)  # (B,L,3)
        else:
            pos_3d = compute_position_ids_input_ids(input_ids)
        B, L, _ = pos_3d.shape

        # Max position per batch across the 3 planes and sequence dimension: (B,)
        m_per_batch = pos_3d.amax(dim=(1, 2))

        # Sequence lengths per batch: (B,)
        if attention_mask is None:
            seq_lens = torch.full_like(m_per_batch, L)
        else:
            seq_lens = attention_mask.eq(1).sum(dim=-1).to(dtype=m_per_batch.dtype, device=m_per_batch.device)

        rope_deltas = (m_per_batch + 1 - seq_lens).to(dtype=pos_3d.dtype).unsqueeze(1)
        return pos_3d, rope_deltas

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        tensor_stream: TensorStream | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        """
        Forward pass for conditional generation supporting both standard inputs and TensorStream.
        Uses our embed_stream approach for multimodal inputs.
        """

        # Don't compute embeddings here - let the model handle it
        if tensor_stream is not None:
            input_ids = None
        if input_ids is None and inputs_embeds is None and tensor_stream is None:
            raise ValueError("Either input_ids, inputs_embeds, or tensor_stream must be provided.")

        # Build position ids (MRoPE) if needed and tensor_stream is available
        # During decode we reuse `self.rope_deltas` computed on the initial forward pass; `rope_delta` captures how far
        # cached rotary phases have progressed so we can advance `position_ids` without rebuilding the TensorStream.
        if position_ids is None and tensor_stream is not None:
            position_ids, self.rope_deltas = self.get_rope_index(input_ids, tensor_stream, attention_mask)
        elif position_ids is None and input_ids is not None:
            # For text inputs build position ids and modality tensor
            position_ids = compute_position_ids_input_ids(input_ids)
            if cache_position is not None and self.rope_deltas is not None:
                # Combine the incremental decode step (`cache_position`) with cached offsets so hidden states continue
                # rotating in lockstep across generation steps.
                rope_delta = (cache_position[0] + self.rope_deltas).to(input_ids.device)
            else:
                rope_delta = 0
            if cache_position is not None and not isinstance(rope_delta, int):  # otherwise `deltas` is an int `0`
                batch_size = input_ids.shape[0]
                rope_delta = rope_delta.repeat_interleave(batch_size // rope_delta.shape[0], dim=0)
            position_ids = position_ids.add(rope_delta)

        if tensor_stream is not None:
            modality_tensor = modality_mask(tensor_stream)
        else:
            batch_size, seq_len = input_ids.shape
            modality_tensor = torch.empty(batch_size, seq_len, device=position_ids.device).fill_(TextType.text.value)

        outputs = self.model(
            input_ids=input_ids,
            tensor_stream=tensor_stream,
            attention_mask=attention_mask,
            position_ids=position_ids,
            modality_tensor=modality_tensor,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: list[torch.FloatTensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        tensor_stream: TensorStream | None = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Prepare inputs for generation, handling TensorStream inputs properly.
        """
        # Call parent preparation
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )

        # Handle TensorStream for first forward pass only
        if tensor_stream is not None and (cache_position is None or cache_position[0] == 0):
            model_inputs["tensor_stream"] = tensor_stream
        # Let forward rebuild position_ids using cached deltas during decode
        model_inputs["position_ids"] = None
        # Drop tensor_stream after step 0
        if cache_position is not None and cache_position[0] != 0:
            model_inputs["tensor_stream"] = None
        return model_inputs

    def can_generate(self) -> bool:
        return True


AutoImageProcessor.register(
    IsaacConfig,
    slow_image_processor_class=IsaacImageProcessor,
    fast_image_processor_class=IsaacImageProcessorFast,
    exist_ok=True,
)


__all__ = [
    "IsaacConfig",
    "IsaacModel",
    "IsaacForConditionalGeneration",
    "IsaacImageProcessor",
    "IsaacImageProcessorFast",
    "IsaacProcessor",
]
def _prepare_image_tensor(image: torch.Tensor, scale: float, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    """Mirror the prepare_image_tensor utility used in the training pipelines."""
    if not torch.is_floating_point(image):
        image = image.float()

    rescaled = image * scale
    mean_tensor = torch.tensor(mean, dtype=torch.float32, device=rescaled.device).view(1, 1, 1, -1)
    std_tensor = torch.tensor(std, dtype=torch.float32, device=rescaled.device).view(1, 1, 1, -1)
    normalized = (rescaled - mean_tensor) / std_tensor
    return normalized


def _compute_residual_p_frames(frames: torch.Tensor, is_p_frame: list[bool]) -> torch.Tensor:
    """Compute residuals for P-frames to stay in sync with the training pipeline."""
    if not any(is_p_frame):
        return frames

    frame_indices = torch.arange(len(is_p_frame), device=frames.device)
    i_frame_mask = torch.tensor([not flag for flag in is_p_frame], device=frames.device)
    last_i_indices = torch.cummax((i_frame_mask * (1 + frame_indices)), dim=0).values.long() - 1
    p_indices = frame_indices[torch.tensor(is_p_frame, device=frames.device)]
    frames[p_indices] = frames[p_indices] - frames[last_i_indices[p_indices]]
    return frames
