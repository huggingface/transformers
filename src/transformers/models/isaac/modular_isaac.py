from __future__ import annotations

from collections import defaultdict
from typing import Any, Union, TypedDict

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image


from ...utils import logging
from ...processing_utils import ProcessorMixin, BatchFeature
from ...tokenization_utils_base import PreTrainedTokenizerBase
from ..auto import AutoTokenizer
from ..qwen3.configuration_qwen3 import Qwen3Config
from ..qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3PreTrainedModel,
    Qwen3DecoderLayer,
    Qwen3Model
)
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...tokenization_utils import TensorType
import re

from ..siglip2.modeling_siglip2 import Siglip2MLP
from ..siglip2.configuration_siglip2 import Siglip2VisionConfig

from perceptron.tensorstream import (
    Event,
    Stream,
    TensorStream,
    TextType,
    VisionType,
    create_stream,
    group_streams,
)
from perceptron.tensorstream.ops import (
    compute_mrope_pos_tensor,
    modality_mask,
    reconstruct_tensor_stream_from_compact_dict,
    slice as ts_slice,
    tensor_stream_token_view,
)

logger = logging.get_logger(__name__)


class PixelShuffleSiglip2VisionConfig(Siglip2VisionConfig):
    """Vision configuration for Isaac with Pixel Shuffle support.

    Extends Siglip2VisionConfig with additional fields for pixel shuffle.
    """

    model_type = "pixel_shuffle_siglip2"
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


def create_cumulative_seq_lengths(seq_sizes: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int]:
    """Create cumulative sequence lengths for variable-length attention."""
    cu_seqlens = torch.zeros(len(seq_sizes) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = seq_sizes.cumsum(0)
    max_seqlen = int(seq_sizes.max().item()) if len(seq_sizes) > 0 else 0
    return cu_seqlens, max_seqlen


class Siglip2VariableSequenceEmbeddings(nn.Module):
    def __init__(self, config: PixelShuffleSiglip2VisionConfig):
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

    def positional_embeddings(
        self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        # Prepare positional embeddings grid: (1, embed_dim, h, w)
        positional_embeddings = (
            self.position_embedding.weight.reshape(self.position_embedding_size, self.position_embedding_size, -1)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        _seq_patches, _seq_sizes, spatial_shapes = packed_seq_patches
        pos_embeds_list = []
        mode = "bilinear"
        align_corners = False
        antialias = True
        for spatial_shape in spatial_shapes:
            height, width = spatial_shape
            # Guard to ensure height and width are positive for torch.compile
            if height > 0 and width > 0:
                resized_pos_embed = F.interpolate(
                    positional_embeddings,
                    size=(height, width),
                    mode=mode,
                    align_corners=align_corners,
                    antialias=antialias,
                )
                # Reshape from (1, embed_dim, height, width) to (height*width, embed_dim)
                resized_pos_embed = resized_pos_embed.reshape(self.embed_dim, height * width).transpose(0, 1)
            else:
                # Fallback - should never happen in practice
                resized_pos_embed = positional_embeddings.reshape(
                    self.embed_dim, self.position_embedding_size * self.position_embedding_size
                ).transpose(0, 1)[: height * width]
            pos_embeds_list.append(resized_pos_embed)

        # Concatenate all positional embeddings along the sequence dimension
        pos_embeds = torch.cat(pos_embeds_list, dim=0)
        return pos_embeds

    def forward(self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        seq_patches, _seq_sizes, _spatial_shapes = packed_seq_patches

        # Apply patch embeddings
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(seq_patches.to(dtype=target_dtype))
        pos_embeds = self.positional_embeddings(packed_seq_patches)

        # Add positional embeddings to patch embeddings
        embeddings = patch_embeds + pos_embeds
        return embeddings


class Siglip2VariableLengthAttention(nn.Module):
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
        batch_size, seq_len, _ = hidden_states.size()

        # For variable-length attention, we need to reshape to (total_tokens, embed_dim)
        if batch_size != 1:
            raise ValueError("Variable-length attention expects batch_size=1 for packed sequences")
        hidden_states = hidden_states.squeeze(0)  # Remove batch dimension: (seq_len, embed_dim)

        # Store original dtype
        orig_dtype = hidden_states.dtype

        # 1. Linear projections
        Q = self.q_proj(hidden_states)  # (seq_len, embed_dim)
        K = self.k_proj(hidden_states)  # (seq_len, embed_dim)
        V = self.v_proj(hidden_states)  # (seq_len, embed_dim)

        # 2. Reshape for multi-head attention: (seq_len, n_heads, head_dim)
        Q = Q.view(-1, self.num_heads, self.embed_dim // self.num_heads)
        K = K.view(-1, self.num_heads, self.embed_dim // self.num_heads)
        V = V.view(-1, self.num_heads, self.embed_dim // self.num_heads)

        # 3. Apply variable-length attention using flash attention
        attn_output, _, _, _, _ = torch.ops.aten._flash_attention_forward(
            query=Q,
            key=K,
            value=V,
            cum_seq_q=cu_seqlens,
            cum_seq_k=cu_seqlens,
            max_q=max_seqlen,
            max_k=max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            return_debug_mask=False,
            scale=self.scale,
            window_size_left=-1,
            window_size_right=-1,
            alibi_slopes=None,
        )

        # 4. Reshape attention output from (seq_len, n_heads, head_dim) to (seq_len, embed_dim)
        attn_output = attn_output.reshape(seq_len, self.embed_dim)

        # 5. Convert back to original dtype if needed
        if attn_output.dtype != orig_dtype:
            attn_output = attn_output.to(orig_dtype)

        # 6. Project output
        attn_output = self.out_proj(attn_output)  # (seq_len, embed_dim)

        # 7. Add back batch dimension for compatibility
        attn_output = attn_output.unsqueeze(0)  # (1, seq_len, embed_dim)

        return attn_output, None


class IsaacSiglip2EncoderLayer(nn.Module):
    """Siglip2 encoder layer with variable-length attention."""

    def __init__(self, config: PixelShuffleSiglip2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Siglip2VariableLengthAttention(config)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)  # Use HF's Siglip2MLP
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

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


class IsaacEncoder(nn.Module):
    """Encoder using Isaac encoder layers with variable-length attention support."""

    def __init__(self, config: PixelShuffleSiglip2VisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([IsaacSiglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])

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

    r = int(scale_factor)
    if r < 2:
        raise ValueError("`scale_factor` must be ≥ 2")

    # Safety: all spatial dims must be divisible by r
    # Cannot run under torch compile fullgraph mode hence
    if not torch.compiler.is_compiling():
        if not ((token_grids[:, 0] % r == 0).all() and (token_grids[:, 1] % r == 0).all()):
            raise AssertionError(
                f"Every (H,W) in `token_grids` must be divisible by scale_factor={r}, got {token_grids.tolist()}"
            )

    gather_chunks: list[torch.Tensor] = []
    tok_offset = 0

    for seq_len, (h, w) in zip(seq_sizes.tolist(), token_grids.tolist(), strict=False):
        # Build the (H, W) grid of flat indices for this image
        grid = torch.arange(seq_len, device=device, dtype=torch.int64) + tok_offset
        grid = grid.view(h, w)  # (H, W)

        # -------- identical ordering to your fixed-res routine --------
        # Step 1: split width into blocks of r
        grid = grid.view(h, w // r, r)  # (H, W/r, r)
        # Step 2: now split height into blocks of r
        grid = grid.view(h // r, r, w // r, r)  # (H/r, r, W/r, r)
        # Step 3: final permutation to (H/r, W/r, r, r)
        grid = grid.permute(0, 2, 1, 3).contiguous()  # (H/r, W/r, r, r)
        # Step 4: each (r, r) block forms one output token
        gather_chunks.append(grid.reshape(-1, r * r))  # (H*W / r², r²)

        tok_offset += seq_len

    # Concatenate over all images in the packed batch
    gather_idx = torch.cat(gather_chunks, dim=0)  # (Σ_i HᵢWᵢ/r², r²)
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
    r = int(scale_factor)

    # Calculate seq_sizes from token_grids
    seq_sizes = torch.prod(token_grids, dim=-1)

    # Build index map and gather in one go
    gather_idx = create_pixel_shuffle_index_map(
        seq_sizes=seq_sizes,
        token_grids=token_grids,
        scale_factor=r,
        device=x_.device,
    )  # (new_seq, r²)

    # Gather → (new_seq, r², embed_dim)
    gathered = x_[gather_idx]  # fancy indexing keeps gradient

    # Merge the r² group dimension into channels to finish the shuffle
    out = gathered.reshape(gathered.size(0), embed_dim * r * r)

    # Restore batch dimension if needed
    if keep_batch_dim:
        out = out.unsqueeze(0)
    return out


class Siglip2SequenceVisionTransformer(nn.Module):
    def __init__(self, config: PixelShuffleSiglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = Siglip2VariableSequenceEmbeddings(config)
        self.encoder = IsaacEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pixel_shuffle_scale_factor = config.pixel_shuffle_scale_factor

    def forward(self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor]):
        seq_patches, token_grids = packed_seq_patches
        seq_sizes = torch.prod(token_grids, dim=-1)

        # Get embeddings from packed sequence
        hidden_states = self.embeddings((seq_patches, seq_sizes, token_grids))

        # Add a pseudo batch dimension for the encoder
        hidden_states = hidden_states.unsqueeze(0)

        # Generate cumulative sequence lengths for variable-length attention
        cu_seqlens, max_seqlen = create_cumulative_seq_lengths(seq_sizes, hidden_states.device)

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

MAX_PIXELS = 60_000_000  # 60-megapixel ceiling ≈ 8200 × 7300 px

# Vision preprocessing constants
VISION_MEAN = (0.5, 0.5, 0.5)
VISION_STD = (0.5, 0.5, 0.5)
VISION_SCALE = 1 / 255


def _make_writeable(arr: np.ndarray) -> np.ndarray:
    """Return *arr* itself if it is already writeable, otherwise try to flip the
    write flag in-place and finally fall back to `arr.copy()`.
    This guarantees the buffer handed to `torch.from_numpy()` is always
    writeable, silencing the PyTorch warning about undefined behaviour.
    """
    if arr.flags.writeable:
        return arr

    # First, try the cheap path — in-place flag toggle (works for mmap'd arrays
    # and some shared memory buffers):
    try:
        arr.setflags(write=True)
        return arr  # success: no data copy
    except ValueError:
        # Buffer is inherently read-only (e.g. backed by PyAV / PIL): make copy
        return arr.copy()


def extract_image_pil(image: PIL.Image.Image) -> torch.Tensor | None:
    if image.width * image.height > MAX_PIXELS:
        raise ValueError(f"Image (w={image.width}, h={image.height}) > MAX=`{MAX_PIXELS}`")
    img = image if image.mode == "RGB" else image.convert("RGB")
    arr = np.asarray(img)
    arr = _make_writeable(arr)
    return torch.from_numpy(arr)


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

    def get_scaled_image_size(scale, original_size, patch_size, pixel_shuffle_scale):
        scaled_size = scale * original_size
        divisor = patch_size * pixel_shuffle_scale
        scaled_size = math.ceil(scaled_size / divisor) * divisor
        scaled_size = max(divisor, scaled_size)
        return int(scaled_size)

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


_MEAN_TENSOR = torch.tensor(VISION_MEAN, dtype=torch.float32).view(1, 1, 1, -1)
_STD_TENSOR = torch.tensor(VISION_STD, dtype=torch.float32).view(1, 1, 1, -1)


def prepare_image_tensor(
    image: torch.Tensor,
    scale: float = VISION_SCALE,
) -> torch.Tensor:
    r"""Standardize RGB images prior to patch extraction via rescaling and whitening.

    Args:
        image (`torch.Tensor`):
            Tensor with shape `(..., height, width, 3)` containing RGB values. The tensor is converted to floating
            point if needed.
        scale (`float`, *optional*, defaults to `VISION_SCALE`):
            Scalar multiplier applied before normalization.
    Returns:
        `torch.Tensor`: Normalized tensor with the same shape as the input and dtype `torch.float32`.
    """
    if not torch.is_floating_point(image):
        image = image.float()
    rescaled = image * scale

    # Use precomputed tensors and move to the correct device if needed
    mean_tensor = _MEAN_TENSOR.to(image.device)
    std_tensor = _STD_TENSOR.to(image.device)

    normalized = (rescaled - mean_tensor) / std_tensor
    return normalized


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


def process_vision_for_patches(
    images: torch.Tensor,
    patch_size: int,
    max_num_patches: int,
    min_num_patches: int | None = None,
    pixel_shuffle_scale: int = 1,
) -> tuple[torch.Tensor, list[int]]:
    r"""Resize, normalize, and patchify RGB images for the vision encoder.

    Args:
        images (`torch.Tensor`):
            Either `(height, width, channels)` for a single image or `(num_images, height, width, channels)` for a
            batch. Channels are expected to be RGB.
        patch_size (`int`):
            Edge length of square patches; implictly controls resize grid granularity.
        max_num_patches (`int`):
            Maximum number of patches allowed after resizing.
        min_num_patches (`int`, *optional*):
            Minimum number of patches. If provided, the routine upsamples images as needed to satisfy the lower bound.
        pixel_shuffle_scale (`int`, *optional*, defaults to 1):
            pixel shuffle scale factor; influences the target grid that the function produces.

    Returns:
        `tuple[torch.Tensor, list[int]]`: A pair `(patches, dims_virtual)` where `patches` has shape
        `(num_images, target_h / patch_size, target_w / patch_size, channels * patch_size**2)` and `dims_virtual`
        encodes effective `(images, height, width)` dimensions after optional pixel shuffling.
    """
    # Add batch dim if single image
    if images.dim() == 3:
        images = images.unsqueeze(0)

    # Permute to channel first for resize
    images = images.permute(0, 3, 1, 2)

    # Get target dimensions
    _, _, orig_height, orig_width = images.shape
    target_height, target_width = get_image_size_for_max_num_patches(
        orig_height,
        orig_width,
        patch_size,
        max_num_patches,
        min_num_patches=min_num_patches,
        pixel_shuffle_scale=pixel_shuffle_scale,
    )

    # Resize
    images = F.interpolate(
        images,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )

    # Back to channel last
    images = images.permute(0, 2, 3, 1)

    # Normalize
    images = prepare_image_tensor(images)

    # Patchify
    patches = patchify_vision(images, patch_size=patch_size)

    # Calculate dimensions for the patches
    n_images, h_patches, w_patches, _ = patches.shape
    dims_virtual = (
        [1, h_patches, w_patches]
        if pixel_shuffle_scale == 1
        else [1, h_patches // pixel_shuffle_scale, w_patches // pixel_shuffle_scale]
    )

    return patches, dims_virtual


def precompute_inv_freq(theta: float, dim: int) -> torch.Tensor:
    """
    Returns shape (dim//2,).
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    return inv_freq  # type: ignore[return-value]


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
    sub_configs = {"vision_config": PixelShuffleSiglip2VisionConfig}

    def __init__(
        self,
        vision_config=None,
        vision_patch_size: int = 16,
        vision_max_num_patches: int = 256,
        vision_min_num_patches: int | None = None,
        pixel_shuffle_scale: int = 1,
        max_sequence_length: int = 16384,
        vision_token: str = "<image>",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Handle vision config - either dict or PixelShuffleSiglip2VisionConfig instance
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

        # Processing parameters
        self.max_sequence_length = max_sequence_length
        self.vision_token = vision_token


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


class IsaacProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("AutoTokenizer",)

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        config: IsaacConfig,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config

        # Use vision token from config
        self.vision_token = config.vision_token

        # Processing parameters
        self.max_sequence_length = config.max_sequence_length

        # Vision processing parameters
        self.patch_size = config.video_patch_size
        self.max_num_patches = config.vision_max_num_patches
        self.min_num_patches = config.vision_min_num_patches
        self.pixel_shuffle_scale = config.pixel_shuffle_scale

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> Any:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, **kwargs
        )

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
                if image_idx < len(images):
                    # Create vision event from PIL image
                    image_tensor = extract_image_pil(images[image_idx])
                    if image_tensor is not None:
                        # Create a vision event with the image tensor
                        vision_event = Event(
                            data=image_tensor.unsqueeze(0),  # HWC format from extract_image_pil
                            type=VisionType.image,  # I-frame
                            time=(current_time, current_time),
                        )
                        events.append(vision_event)
                    image_idx += 1
            elif part:  # Non-empty text part
                # tokens = self.text_processor.tokenize(part, add_special_tokens=False)
                text_event = create_text_event(self.tokenizer, part, time=current_time)
                events.append(text_event)

        # Process vision events if any
        if any(event.type == VisionType.image for event in events):
            # Separate text and vision events for processing
            text_events = [event for event in events if event.type == TextType.text]
            vision_events = [event for event in events if event.type == VisionType.image]

            # Process vision events using functional approach
            processed_vision_events = []
            for vision_event in vision_events:
                # Process the vision data
                patches, dims_virtual = process_vision_for_patches(
                    vision_event.data.squeeze(0),  # Remove the extra dimension
                    patch_size=self.patch_size,
                    max_num_patches=self.max_num_patches,
                    min_num_patches=self.min_num_patches,
                    pixel_shuffle_scale=self.pixel_shuffle_scale,
                )

                # Update event with processed data
                vision_event.data = patches.unsqueeze(1)  # Add back frame dimension
                vision_event.dims_virtual = dims_virtual
                vision_event.dims_real = (
                    dims_virtual
                    if self.pixel_shuffle_scale == 1
                    else [
                        dims_virtual[0],
                        dims_virtual[1] * self.pixel_shuffle_scale,
                        dims_virtual[2] * self.pixel_shuffle_scale,
                    ]
                )
                vision_event.idx_range = (0, math.prod(dims_virtual))

                # Flatten the patches
                vision_event.data = vision_event.data.reshape(-1, vision_event.data.shape[-1])
                processed_vision_events.append(vision_event)

            events = text_events + processed_vision_events

        # Create stream without scheduling (events already in order)
        return create_stream(events, priority=[TextType.text, VisionType.image], schedule=True)

    def __call__(
        self,
        text: Union[str, list[str]],
        images: Union[PIL.Image.Image, list[PIL.Image.Image], None] = None,
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
    def __init__(self, config: IsaacConfig, device=None):
        super().__init__()

        # Extract dimensions from config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        # Get rope_scaling config - use direct access when available
        rope_scaling = getattr(config, "rope_scaling", None) or {}

        # Read RopeScaling parameters
        self.rope_type = rope_scaling.get("rope_type", "default")

        self.mrope_section = [
            self.head_dim // 4,  # 2x more for temporal dim
            self.head_dim // 8,
            self.head_dim // 8,
        ]

        rope_base = getattr(config, "rope_theta", 10000.0)
        inv_freq = precompute_inv_freq(rope_base, self.head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor, modality_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # Ensure non-spatial tokens have 1D rotation equivalence
            not_spatial = ~(modality_tensor == VisionType.image.value)
            # shape is [N, 1]
            data_1d = position_ids[not_spatial][..., 0].unsqueeze(-1)
            # now broadcast it from [N, 1] -> [N, D] so it matches pos[not_spatial] exactly
            data_1d = data_1d.expand(-1, position_ids.shape[-1])  # expand along the last dim
            position_ids = position_ids.clone()  # Clone to avoid warning about in-place operations on expanded tensors
            position_ids[not_spatial] = data_1d
            position_ids = position_ids.permute(2, 0, 1)  # pos dim first -> (3, B, L)
            cos, sin = precompute_cos_sin_3d(position_ids, self.inv_freq, self.mrope_section)

        return cos, sin


class IsaacModel(Qwen3Model):
    def __init__(self, config: IsaacConfig):
        super().__init__(config)
        self.layers = torch.nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = IsaacRotaryEmbedding(config, device=self.device)

        vision_cfg = config.vision_config
        if vision_cfg is None:
            raise ValueError("IsaacConfig should always have vision_config")

        hidden_dim = vision_cfg.hidden_size * (vision_cfg.pixel_shuffle_scale_factor**2)
        self.vision_embedding = nn.Sequential(
            Siglip2SequenceVisionTransformer(vision_cfg),
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

    def embed_text_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed text tokens, squeezing singleton dimensions."""
        # Text events are shaped as (..., 1); squeeze the singleton index dim
        h = self.embed_tokens(token_ids)
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
            inputs_embeds = self.embed_tokens(input_ids)
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

        for decoder_layer in self.layers:
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

            hidden_states = layer_outputs[0]

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


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


__all__ = [
    "IsaacConfig",
    "IsaacModel",
    "IsaacForConditionalGeneration",
    "IsaacProcessor",
]