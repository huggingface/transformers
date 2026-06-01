# Copyright 2026 NVIDIA CORPORATION and The HuggingFace Inc. team. All rights reserved.
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
"""LocateAnything: a visual-grounding VLM (MoonViT vision tower + Qwen2 language model)."""

import math
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from ... import initialization as init
from ...activations import PytorchGELUTanh
from ...cache_utils import DynamicCache
from ...generation import GenerationMixin
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
    auto_docstring,
    is_flash_attn_2_available,
    logging,
    torch_compilable_check,
)
from ..auto import AutoModelForCausalLM
from .configuration_locateanything import LocateAnythingConfig, MoonViTConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None

logger = logging.get_logger(__name__)


def multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: torch.Tensor | None = None,
    k_cu_seqlens: torch.Tensor | None = None,
):
    """Multi-head attention using flash attention 2.

    Args:
        q, k, v: tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
        q_cu_seqlens (torch.Tensor): cumulative sequence lengths of q.
            The first element should be 0 and the last element should be q.shape[0].
        k_cu_seqlens (torch.Tensor): cumulative sequence lengths of k.
            The first element should be 0 and the last element should be k.shape[0].

    Returns:
        output: shape (batch_size, seqlen, dim) or (tot_seqlens, dim) if packing,
            where dim = num_heads * head_dim
    """
    if flash_attn_varlen_func is None:
        logger.warning_once("flash_attn is not available for MoonViT; falling back to sdpa attention.")
        return sdpa_attention(
            q,
            k,
            v,
            q_cu_seqlens=q_cu_seqlens,
            k_cu_seqlens=k_cu_seqlens,
        )

    # Unified format legal check
    if not (q.dim() == k.dim() == v.dim() == 3):
        raise ValueError("q, k, v must have 3 dims")
    if q_cu_seqlens[-1] != q.shape[0]:
        raise ValueError("q_cu_seqlens must sum to q.shape[0]")
    if not (k_cu_seqlens[-1] == k.shape[0] == v.shape[0]):
        raise ValueError("k_cu_seqlens must sum to k.shape[0]")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"unsupported dtype {q.dtype} for multihead attn")

    max_seqlen_q = (q_cu_seqlens[1:] - q_cu_seqlens[:-1]).max().item()
    max_seqlen_k = (k_cu_seqlens[1:] - k_cu_seqlens[:-1]).max().item()
    attn_out = flash_attn_varlen_func(
        q,
        k,
        v,
        q_cu_seqlens,
        k_cu_seqlens,
        max_seqlen_q,
        max_seqlen_k,
        causal=False,
    )
    attn_out = attn_out.flatten(start_dim=-2)

    return attn_out


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: torch.Tensor | None = None,
    k_cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """SDPA attention.

    Args:
        q, k, v: tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
    """
    seq_length = q.shape[0]
    attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[
            ...,
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
        ] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


def eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: torch.Tensor | None = None,
    k_cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    seq_length = q.shape[0]
    attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[
            ...,
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
        ] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn_weight += attention_mask
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)

    attn_output = attn_weight @ v
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


VL_VISION_ATTENTION_FUNCTIONS = {
    "flash_attention_2": multihead_attention,
    "sdpa": sdpa_attention,
    "eager": eager_attention,
}


def _apply_rope_input_validation(x, freqs_cis):
    if x.ndim != freqs_cis.ndim + 1:
        raise ValueError(f"Invalid shapes: {x.shape}, {freqs_cis.shape}")
    if x.shape[:-2] != freqs_cis.shape[:-1]:
        raise ValueError(f"Invalid shapes: {x.shape}, {freqs_cis.shape}")
    if x.shape[-1] != 2 * freqs_cis.shape[-1]:
        raise ValueError(f"Invalid shapes: {x.shape}, {freqs_cis.shape}")
    if freqs_cis.dtype != torch.complex64:
        raise ValueError(f"freqs_cis must be complex64, got {freqs_cis.dtype}")


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)

    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
    # ..., num_heads, head_dim/2
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Learnable2DInterpPosEmb(nn.Module):
    def __init__(self, height: int, width: int, dim: int, interpolation_mode: str = "bicubic") -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for shape in grid_hws.tolist():
            if shape == self.weight.shape[:-1]:
                pos_embs.append(self.weight.flatten(end_dim=1))
            else:
                pos_embs.append(
                    F.interpolate(
                        self.weight.permute((2, 0, 1)).unsqueeze(0),
                        size=shape,
                        mode=self.interpolation_mode,
                    )
                    .squeeze(0)
                    .permute((1, 2, 0))
                    .flatten(end_dim=1)
                )
        out = x + torch.cat(pos_embs)
        return out


class MoonVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
    ):
        super().__init__()
        if not isinstance(patch_size, (int, Sequence)):
            raise TypeError(f"Invalid patch_size type: {type(patch_size)}")
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if len(patch_size) != 2:
            raise ValueError(f"Expected patch_size to be a tuple of 2, got {patch_size}")
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size)

        self.pos_emb = Learnable2DInterpPosEmb(height=pos_emb_height, width=pos_emb_width, dim=out_dim)

    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (L, Channels): input tensor
            grid_hws (N, 2): grid height and width

        Returns:
            (L, Cout) tensor
        """
        x = self.proj(x).view(x.size(0), -1)
        # apply positional embedding
        x = self.pos_emb(x, grid_hws)
        return x


class Rope2DPosEmb(nn.Module):
    """2D rotary position embedding with multi-resolution support.

    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.

    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py

    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
        device (str): the device to store the precomputed cis
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        if self.dim % 4 != 0:
            raise ValueError("dim must be divisible by 4")
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

        self.freqs_cis = None

    def extra_repr(self):
        return (
            f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"
        )

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.

        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_hws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_hws (torch.Tensor): grid height and width

        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if self.freqs_cis is None:
            self.freqs_cis = self._precompute_freqs_cis(grid_hws.device)

        shapes = grid_hws.tolist()
        if not all(1 <= h <= self.max_height and 1 <= w <= self.max_width for h, w in shapes):
            raise ValueError(f"grid shapes {shapes} exceed the maximum ({self.max_height}, {self.max_width})")
        freqs_cis = torch.cat(
            [self.freqs_cis[:h, :w].reshape(-1, self.dim // 2) for h, w in shapes],
            dim=0,
        )
        return freqs_cis


class MLP2(nn.Module):
    """
    Args:
        dims: [in_dim, hidden_dim, out_dim]
        bias: whether to use bias in linear layer.
    """

    def __init__(self, dims: list[int], activation, bias=True):
        super().__init__()
        if len(dims) != 3:
            raise ValueError(f"MLP2 expects dims=[in_dim, hidden_dim, out_dim], got {dims}")
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation
        for m in [self.fc0, self.fc1]:
            nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        x = self.activation(x)
        return self.fc1(x)


class MoonVitEncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        *,
        attn_implementation: str = "eager",
        activation=F.gelu,
        attn_bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = self.hidden_dim // self.num_heads
        self.attn_implementation = attn_implementation

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], activation)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
    ):
        """
        Args:
            x (torch.Tensor): (batch_size, seqlen, hidden_dim)
            cu_seqlens (torch.Tensor):
        """
        xqkv = self.wqkv(x)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_heads,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (batch_size, seqlen, 3, nheads, headdim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        attn_func = VL_VISION_ATTENTION_FUNCTIONS[self.attn_implementation]
        attn_out = attn_func(xq, xk, xv, q_cu_seqlens=cu_seqlens, k_cu_seqlens=cu_seqlens)

        attn_out = self.wo(attn_out)
        return attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: non-packed (B, N, D) or packed (L, D). if non-packed, seqlens should be None, if packed, seqlens should be set

        Returns:
            output: same shape of input, non-packed (B, N, D) for non-packed input, (L, D) for packed input
        """
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        attn_out = self.attention_qkvpacked(hidden_states, cu_seqlens, rope_freqs_cis=rope_freqs_cis)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.mlp(self.norm1(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states


class MoonVitEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
    ) -> None:
        super().__init__()

        self.rope_2d = Rope2DPosEmb(block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512)
        self.blocks = nn.ModuleList([MoonVitEncoderLayer(**block_cfg) for _ in range(num_layers)])
        self.final_layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_hws=grid_hws)

        lengths = torch.cat(
            (
                torch.zeros(1, device=hidden_states.device, dtype=grid_hws.dtype),
                grid_hws[:, 0] * grid_hws[:, 1],
            )
        )
        cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)

        for _, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cu_seqlens, rope_freqs_cis=rope_freqs_cis)

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


def patch_merger(
    x: torch.Tensor,
    grid_hws: torch.Tensor,
    merge_kernel_size: list[int, int] = (2, 2),
) -> list[torch.Tensor]:
    d_model = x.size(-1)

    outputs = []
    pre_sum = 0
    for x_shape in grid_hws.tolist():
        height, width = x_shape[0], x_shape[1]
        # Get the current sequence
        seq = x[pre_sum : pre_sum + height * width]
        # Reshape along self.merge_kernel_size and concat to the last dimension
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = height // kernel_height, width // kernel_width
        reshaped_seq = seq.view(new_height, kernel_height, new_width, kernel_width, d_model)
        reshaped_seq = reshaped_seq.permute(0, 2, 1, 3, 4).contiguous()
        padded_seq = reshaped_seq.view(new_height * new_width, -1)
        outputs.append(padded_seq)
        pre_sum += height * width

    return outputs


@auto_docstring(custom_intro="The MoonViT vision encoder used as the vision tower of LocateAnything.")
class MoonViTPreTrainedEncoder(PreTrainedModel):
    config_class = MoonViTConfig
    model_type = "moonvit"
    _no_split_modules = ["PackingTransformer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: MoonViTConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        config = deepcopy(config)
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.patch_embed = MoonVisionPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
        )

        self.encoder = MoonVitEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "mlp_dim": config.intermediate_size,
                "activation": PytorchGELUTanh(),
                "attn_bias": True,
                "attn_implementation": config._attn_implementation,
            },
        )
        self.post_init()

    def forward(self, pixel_values: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): The input pixel values.
            grid_hws (torch.Tensor): The grid height and width.

        Returns:
            torch.Tensor: The output tokens.
        """
        hidden_states = self.patch_embed(pixel_values, grid_hws)
        hidden_states = self.encoder(hidden_states, grid_hws)
        hidden_states = patch_merger(hidden_states, grid_hws, merge_kernel_size=self.merge_kernel_size)
        return hidden_states


def build_window_attention_mask(
    q_len: int,
    kv_len: int,
    past_len: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
    causal_attn: bool = False,
) -> torch.Tensor:
    """Build the 4D block-diffusion attention mask for one multi-token-prediction (MTP) window.

    The base is a standard causal mask (a query at global position `past_len + i` may attend to
    key positions `<= past_len + i`). On top of that, the trailing `block_size` queries -- the
    diffusion window of duplicated/masked tokens -- attend bidirectionally to the trailing
    `block_size` keys, and the duplicated token column is masked out so it is not attended twice.

    Args:
        q_len (`int`): Number of query tokens forwarded this step (the not-yet-cached suffix).
        kv_len (`int`): Total key/value length (cached prefix + `q_len`).
        past_len (`int`): Length of the cached prefix (global offset of the first query).
        block_size (`int`): Size of the diffusion window.
        dtype (`torch.dtype`): Floating dtype of the mask.
        device (`torch.device`): Device of the mask.
        causal_attn (`bool`, *optional*, defaults to `False`): Keep strict causal masking inside
            the window instead of making it bidirectional.

    Returns:
        `torch.Tensor`: A `(1, 1, q_len, kv_len)` additive mask (`0.0` for allowed, large negative
        for masked positions).
    """
    min_val = torch.finfo(dtype).min
    q_pos = torch.arange(q_len, device=device) + past_len
    k_pos = torch.arange(kv_len, device=device)
    allowed = k_pos[None, :] <= q_pos[:, None]
    mask = torch.zeros((q_len, kv_len), dtype=dtype, device=device)
    mask.masked_fill_(~allowed, min_val)
    if not causal_attn:
        mask[-block_size:, -block_size:] = 0.0
    # Mask the duplicated last committed token so it is not attended to inside the window.
    mask[-block_size:, -block_size - 1] = min_val
    return mask[None, None]


def get_token_ids_from_config(config) -> dict[str, int]:
    """Extract all token IDs from the configuration object.

    Args:
        config: Configuration object (LocateAnythingConfig or similar)

    Returns:
        Dictionary containing all token IDs
    """
    token_ids = {}

    # Get from main config
    token_ids["box_start_token_id"] = getattr(config, "box_start_token_id", 151668)
    token_ids["box_end_token_id"] = getattr(config, "box_end_token_id", 151669)
    token_ids["coord_start_token_id"] = getattr(config, "coord_start_token_id", 151677)
    token_ids["coord_end_token_id"] = getattr(config, "coord_end_token_id", 152677)
    token_ids["ref_start_token_id"] = getattr(config, "ref_start_token_id", 151672)
    token_ids["ref_end_token_id"] = getattr(config, "ref_end_token_id", 151673)
    token_ids["none_token_id"] = getattr(config, "none_token_id", 4064)

    # Get from text_config
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        token_ids["null_token_id"] = getattr(text_config, "null_token_id", 152678)
        token_ids["im_end_token_id"] = getattr(text_config, "eos_token_id", 151645)
        token_ids["switch_token_id"] = getattr(text_config, "switch_token_id", 152679)
        token_ids["default_mask_token_id"] = getattr(text_config, "text_mask_token_id", 151676)
    else:
        token_ids["null_token_id"] = 152678
        token_ids["im_end_token_id"] = 151645
        token_ids["switch_token_id"] = 152679
        token_ids["default_mask_token_id"] = 151676

    return token_ids


def top_p_logits(logits: torch.Tensor, top_p: float | None = None) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits: torch.Tensor, top_k: int | None = None) -> torch.Tensor:
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def apply_repetition_penalty(
    logits: torch.Tensor, input_ids: torch.Tensor, repetition_penalty: float = 1.0
) -> torch.Tensor:
    """
    Apply repetition penalty to logits.

    Args:
        logits: Shape [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
        input_ids: Previously generated token ids, shape [batch_size, seq_len]
        repetition_penalty: Penalty factor. > 1.0 penalizes repetition, < 1.0 encourages it.

    Returns:
        Modified logits with repetition penalty applied.
    """
    if repetition_penalty == 1.0:
        return logits

    # Convert to 3D for vectorized computation
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)  # [B, 1, V]
        squeeze_back = True
    else:
        squeeze_back = False

    batch_size, seq_len, vocab_size = logits.shape

    # Construct [B, V] bool mask marking tokens that have appeared in each batch
    device = logits.device
    token_mask = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=device)
    for b in range(batch_size):
        # Apply penalty only based on tokens already generated in this batch
        unique_tokens = input_ids[b].unique()
        # Prevent out-of-bounds: only keep IDs within vocab range
        valid_tokens = unique_tokens[(unique_tokens >= 0) & (unique_tokens < vocab_size)]
        if valid_tokens.numel() > 0:
            token_mask[b, valid_tokens] = True

    # Expand to [B, L, V] to align with logits
    token_mask = token_mask.unsqueeze(1).expand(-1, seq_len, -1)

    # Divide positive values by penalty, multiply negative values by penalty
    positive = logits > 0
    negative = ~positive

    # Apply penalty only at mask positions
    logits = torch.where(token_mask & positive, logits / repetition_penalty, logits)
    logits = torch.where(token_mask & negative, logits * repetition_penalty, logits)

    if squeeze_back:
        logits = logits.squeeze(1)

    return logits


def sample_tokens(
    logits: torch.Tensor,
    generated: torch.Tensor,
    token_ids: dict[str, int],
    **generate_kwargs,
):
    batch_size, seq_len, vocab_size = logits.shape

    repetition_penalty = generate_kwargs.get("repetition_penalty", 1.0)
    temperature = generate_kwargs.get("temperature", 0)
    top_p = generate_kwargs.get("top_p")
    top_k = generate_kwargs.get("top_k")

    # Apply repetition penalty based on all previously generated tokens
    if repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated, repetition_penalty)

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if seq_len == 1:
        return probs, confidence, x0, None

    box_avg = []
    fallback_box = torch.zeros(1, dtype=x0.dtype, device=x0.device)

    for b in range(batch_size):
        decoded_box = decode_bbox_avg(
            logits[b],
            probs[b],
            token_ids,
            keep_k=generate_kwargs.get("keep_k_avg", 4),
            generation_mode=generate_kwargs.get("generation_mode", "hybrid"),
        )
        if decoded_box is not None:
            box_avg.append(decoded_box)
        else:
            out_ref = decode_ref(logits[b], probs[b], token_ids)
            if out_ref is not None:
                box_avg.append(torch.tensor(out_ref, dtype=x0.dtype, device=x0.device))
            else:
                box_avg.append(fallback_box)

    box_avg = torch.stack(box_avg)

    return probs, confidence, x0, box_avg


def is_valid_box_frame(
    probs,
    token_ids: dict[str, int],
    start_thresh=0.6,
    end_thresh=0.2,
    topk=5,
):
    box_start_token_id = token_ids["box_start_token_id"]
    box_end_token_id = token_ids["box_end_token_id"]
    null_token_id = token_ids["null_token_id"]
    im_end_token_id = token_ids["im_end_token_id"]
    none_token_id = token_ids["none_token_id"]  # none

    p_start = probs[0, box_start_token_id]
    if p_start >= start_thresh:
        if (
            probs[1, none_token_id] > 0.2
            and probs[2, box_end_token_id] > 0.2
            and probs[3, null_token_id] > 0.1
            and probs[4, null_token_id] > 0.1
        ):
            return "empty_box"

    end_target_ids = torch.tensor([box_end_token_id, null_token_id, im_end_token_id], device=probs.device)
    end_score = probs[5, end_target_ids].sum()

    if end_score >= end_thresh:
        return "legal_box"

    return "illegal_box"


def decode_bbox_avg(
    logits,
    probs,
    token_ids: dict[str, int],
    keep_k=5,
    start_thresh=0.7,
    end_thresh=0.2,
    generation_mode: str = "hybrid",
):
    """
    Decode bounding box coordinates using top-k weighted average.

    Args:
        logits: Logits of shape (6, vocab_size)
        probs: Probability distribution of shape (6, vocab_size)
        token_ids: Dictionary containing all token IDs
        keep_k: Number of top-k candidate tokens to keep at each position
        start_thresh: Confidence threshold for box start token
        end_thresh: Confidence threshold for box end token

    Returns:
        Decoded bounding box coordinate list in format [box_start, x1, x2, y1, y2, box_end],
        or None if decoding fails
    """
    coord_start_token_id = token_ids["coord_start_token_id"]
    coord_end_token_id = token_ids["coord_end_token_id"]
    box_start_token_id = token_ids["box_start_token_id"]
    box_end_token_id = token_ids["box_end_token_id"]
    none_token_id = token_ids["none_token_id"]

    device = logits.device

    box_type = is_valid_box_frame(probs, token_ids, start_thresh=start_thresh, end_thresh=end_thresh, topk=keep_k)
    if box_type == "empty_box":
        # Handle the <box>none</box> case first
        return torch.tensor(
            [
                box_start_token_id,
                none_token_id,
                box_end_token_id,
                token_ids["null_token_id"],
                token_ids["null_token_id"],
                token_ids["null_token_id"],
            ],
            dtype=torch.long,
            device=probs.device,
        )
    elif box_type == "illegal_box":
        return None

    # Extract probabilities at positions 1-4 and compute Top-K for all 4 positions at once
    pos_probs, pos_ids = torch.topk(probs[1:5], k=keep_k, dim=-1)
    mask = (pos_ids >= coord_start_token_id) & (pos_ids <= coord_end_token_id)
    has_valid = mask.any(dim=-1)  # shape: [4]
    if not has_valid.all():
        return None  # not a box, exit...

    first_valid_idx = mask.long().argmax(dim=-1, keepdim=True)  # [4, 1]
    # Extract highest-probability valid_probs[0] and corresponding valid_ids[0]
    first_valid_probs = pos_probs.gather(-1, first_valid_idx).squeeze(-1)  # [4]
    first_valid_ids = pos_ids.gather(-1, first_valid_idx).squeeze(-1)  # [4]
    if generation_mode == "hybrid":
        valid_counts = mask.sum(dim=-1)  # [4]
        # Compute max/min of valid ids: fill invalid positions with extreme values to avoid interfering with max/min
        LARGE_NUM, SMALL_NUM = 999999, -999999
        valid_ids_for_max = torch.where(mask, pos_ids, torch.tensor(SMALL_NUM, device=device))
        valid_ids_for_min = torch.where(mask, pos_ids, torch.tensor(LARGE_NUM, device=device))

        valid_max = valid_ids_for_max.max(dim=-1)[0]
        valid_min = valid_ids_for_min.min(dim=-1)[0]

        is_abnormal = (first_valid_probs < 0.9) & (valid_counts > 1) & ((valid_max - valid_min) > 60)
        # is_abnormal = (first_valid_probs < 0.7) & (valid_counts > 1) & ((valid_max - valid_min) > 80)

        # Normal positions take top-1 (first_valid_ids); abnormal positions are replaced with 0
        final_coords = torch.where(is_abnormal, torch.tensor(0, device=pos_ids.device), first_valid_ids)
    elif generation_mode == "fast":
        final_coords = first_valid_ids

    start_t = torch.tensor([box_start_token_id], dtype=final_coords.dtype, device=device)
    end_t = torch.tensor([box_end_token_id], dtype=final_coords.dtype, device=device)

    return torch.cat([start_t, final_coords, end_t])


def decode_ref(
    logits,
    probs,
    token_ids: dict[str, int],
    keep_k=5,
    start_thresh=0.6,
):
    ref_start_token_id = token_ids.get("ref_start_token_id")
    coord_start_token_id = token_ids["coord_start_token_id"]
    coord_end_token_id = token_ids["coord_end_token_id"]
    device = probs.device
    probs.size(0)

    # 1. Check if the first position is <ref> and its probability meets start_thresh
    # Note: we directly use the probability of the ref token at position 0 for the check
    if probs[0, ref_start_token_id] < start_thresh:
        return None

    # 2. Extract Top-K probabilities and token IDs for all subsequent positions
    pos_probs, pos_ids = torch.topk(probs[1:], k=keep_k, dim=-1)  # shape: [L-1, keep_k]

    # 3. Build mask: identify coordinate tokens (<0> ~ <1000>)
    is_coord = (pos_ids >= coord_start_token_id) & (pos_ids <= coord_end_token_id)
    # Invert: valid tokens are non-coordinate tokens
    is_valid = ~is_coord  # shape: [L-1, keep_k]

    # Ensure each position has at least one non-coordinate valid token in its Top-K
    has_valid = is_valid.any(dim=-1)  # shape: [L-1]
    if not has_valid.all():
        return None

    # 4. Get the highest-probability valid token
    # Since topk results are sorted in descending order of probability,
    # argmax returns the first index where is_valid is True, i.e., the index of the most probable valid token
    first_valid_idx = is_valid.long().argmax(dim=-1, keepdim=True)  # shape: [L-1, 1]

    # Extract the final token IDs
    final_text_ids = pos_ids.gather(-1, first_valid_idx).squeeze(-1)  # shape: [L-1]

    start_t = torch.tensor([ref_start_token_id], dtype=final_text_ids.dtype, device=device)

    return torch.cat([start_t, final_text_ids])


def handle_pattern(x0, token_ids: dict[str, int], generation_mode: str = "hybrid"):
    """
    Args:
        x0: Token ID list of length 6
        token_ids: Dictionary containing all token IDs
    """
    null_token_id = token_ids["null_token_id"]
    im_end_token_id = token_ids["im_end_token_id"]
    box_start_token_id = token_ids["box_start_token_id"]
    box_end_token_id = token_ids["box_end_token_id"]
    none_token_id = token_ids["none_token_id"]
    coord_start_token_id = token_ids["coord_start_token_id"]
    coord_end_token_id = token_ids["coord_end_token_id"]
    ref_end_token_id = token_ids["ref_end_token_id"]

    x0 = x0.tolist()

    if x0[0] == null_token_id:
        return {
            "type": "im_end",
            "tokens": [im_end_token_id],
            "need_switch_to_ar": False,
            "is_terminal": True,
        }
    elif x0[0] == im_end_token_id:
        return {
            "type": "im_end",
            "tokens": [im_end_token_id],
            "need_switch_to_ar": False,
            "is_terminal": True,
        }
    elif x0[:2] == [box_start_token_id, none_token_id]:
        return {
            "type": "empty_box",
            "tokens": [box_start_token_id, none_token_id, box_end_token_id],
            "need_switch_to_ar": False,
            "is_terminal": False,
        }
    elif x0[0] == box_start_token_id:
        coord_ix = 1
        for coord in x0[1:5]:
            if coord_start_token_id <= coord <= coord_end_token_id:
                coord_ix += 1
            else:
                break

        # Standard 4-coordinate bbox: <box><x1><x2><y1><y2></box>
        if coord_ix == 5 and x0[5] == box_end_token_id:
            return {
                "type": "coord_box",
                "tokens": x0,
                "need_switch_to_ar": False,
                "is_terminal": False,
            }
        # Two-coordinate pointing: <box><x><y></box>
        # Convention: the first two coordinates are valid coord tokens, the third token is box_end.
        # Remaining positions (if any) are not part of the pattern; truncate at box_end.
        elif coord_ix == 3 and x0[3] == box_end_token_id:
            return {
                "type": "point_box",
                "tokens": x0[:4],
                "need_switch_to_ar": False,
                "is_terminal": False,
            }
        else:
            if generation_mode == "fast":
                # fast mode: treat as coord_box, stay in MTP
                return {
                    "type": "coord_box",
                    "tokens": x0,
                    "need_switch_to_ar": False,
                    "is_terminal": False,
                }
            else:
                # hybrid mode: error_box, switch to AR
                return {
                    "type": "error_box",
                    "tokens": x0[:coord_ix],
                    "need_switch_to_ar": True,
                    "is_terminal": False,
                }

    else:
        for i, token in enumerate(x0):
            if token == null_token_id:
                x0 = x0[:i]
                break

        if len(x0) >= 2 and x0[-1] == x0[-2] == ref_end_token_id:
            x0 = x0[:-1]

        return {
            "type": "ref_object",
            "tokens": x0,
            "need_switch_to_ar": False,
            "is_terminal": False,
        }


@auto_docstring
class LocateAnythingPreTrainedModel(PreTrainedModel):
    config_class = LocateAnythingConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer", "Qwen3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_quantized_cache = True
    _supports_sdpa = True

    @classmethod
    def _autoset_attn_implementation(cls, config, *args, **kwargs):
        if getattr(config, "_attn_implementation", None) == "magi":
            return config
        return super()._autoset_attn_implementation(config, *args, **kwargs)

    def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check=False, *args, **kwargs):
        if attn_implementation == "magi":
            return "magi"
        return super()._check_and_adjust_attn_implementation(attn_implementation, is_init_check, *args, **kwargs)

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", None) or self.config.text_config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                init.zeros_(module.weight[module.padding_idx])


@auto_docstring(
    custom_intro="The LocateAnything model: a MoonViT vision tower and an MLP projector on top of a causal language model."
)
class LocateAnythingForConditionalGeneration(LocateAnythingPreTrainedModel, GenerationMixin):
    config_class = LocateAnythingConfig

    def __init__(self, config: LocateAnythingConfig):
        super().__init__(config)

        self.template = config.template
        self.mlp_checkpoint = config.mlp_checkpoint
        self.downsample_ratio = config.downsample_ratio
        self.loss_version = config.loss_version

        if config.vision_config.model_type != "moonvit":
            raise ValueError(
                f"Unsupported vision model type: {config.vision_config.model_type}. Only moonvit is supported."
            )
        vision_attn_impl = getattr(config.vision_config, "_attn_implementation", None) or "flash_attention_2"
        if vision_attn_impl == "flash_attention_2" and not is_flash_attn_2_available():
            logger.warning_once("flash_attn is not available for MoonViT inference; falling back to sdpa.")
            vision_attn_impl = "sdpa"
        config.vision_config._attn_implementation = vision_attn_impl
        self.vision_model = MoonViTPreTrainedEncoder(config.vision_config)

        text_attn_impl = (
            getattr(config.text_config, "_attn_implementation", None)
            or getattr(config, "_attn_implementation", None)
            or "sdpa"
        )
        if text_attn_impl == "flash_attention_2" and not is_flash_attn_2_available():
            logger.warning_once("flash_attn is not available for LocateAnything text inference; falling back to sdpa.")
            text_attn_impl = "sdpa"
        config.text_config._attn_implementation = text_attn_impl
        config.text_config._attn_implementation_internal = text_attn_impl

        self.language_model = AutoModelForCausalLM.from_config(config.text_config)

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        # MLP for moonvit (without pixel_shuffle_back, direct mapping)
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * 4),
            nn.Linear(vit_hidden_size * 4, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        self.image_token_index = config.image_token_index

        self.token_ids = get_token_ids_from_config(config)

        # Initialize weights and set up tied-weight bookkeeping.
        self.post_init()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_hws: torch.Tensor | None = None,
        image_flags: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        """
        Encodes image pixels with the vision tower and projects them into the language model hidden size.
        """
        image_features = self.extract_feature(pixel_values, image_grid_hws)

        if image_flags is not None and image_flags.sum() > 0:
            filtered_image_features = []
            feature_index = 0
            for flag in image_flags:
                num_images = flag.item()
                if num_images != 0:
                    filtered_image_features.extend(image_features[feature_index : feature_index + num_images])
                    feature_index += num_images
                else:
                    feature_index += 1
            image_features = filtered_image_features

        if not image_features:
            return torch.empty(
                0, self.config.text_config.hidden_size, device=pixel_values.device, dtype=pixel_values.dtype
            )

        image_features = torch.cat(image_features, dim=0)
        return self.mlp1(image_features)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ) -> torch.BoolTensor:
        """
        Obtains the image placeholder mask and checks that the number of placeholder tokens matches image features.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.image_token_index, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.image_token_index

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        torch_compilable_check(
            inputs_embeds[special_image_mask].numel() == image_features.numel(),
            f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
        )
        return special_image_mask

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_grid_hws: torch.Tensor | None = None,
        image_flags: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values, image_grid_hws=image_grid_hws, image_flags=image_flags
            )
            image_features = image_features.to(input_embeds.device, input_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=input_embeds, image_features=image_features
            )
            input_embeds = input_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def extract_feature(self, pixel_values, image_grid_hws):
        vit_embeds = self.vision_model(pixel_values=pixel_values, grid_hws=image_grid_hws)

        return vit_embeds

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        visual_features: torch.FloatTensor | None = None,
        image_grid_hws: torch.Tensor | None = None,
        tokenizer=None,
        n_future_tokens: int = 6,
        **generate_kwargs,
    ) -> str:
        r"""
        Parallel Box Decoding (PBD) generation. Three modes are supported:

        - `"slow"`: pure auto-regressive decoding.
        - `"fast"`: block-wise multi-token prediction (MTP) only.
        - `"hybrid"` (default): MTP first, falling back to AR on uncertain boxes.

        The language model is the standard library decoder; the MTP block-diffusion attention
        mask is built by [`~models.locateanything.modeling_locateanything.build_window_attention_mask`]
        and passed in as a 4D mask, and the speculative window is rolled back from the
        [`~cache_utils.DynamicCache`] after each MTP step.
        """
        # `verbose` is accepted for backward compatibility but no longer drives any timing logic.
        generate_kwargs.pop("verbose", False)

        pixel_values = pixel_values.to(self.language_model.dtype)
        if isinstance(image_grid_hws, np.ndarray):
            image_grid_hws = torch.from_numpy(image_grid_hws).to(pixel_values.device, dtype=torch.int32)

        batch_size, seq_len = input_ids.shape
        if batch_size != 1:
            raise ValueError("LocateAnything generation only supports batch size 1 for now.")
        if not generate_kwargs.get("use_cache", False):
            raise ValueError("LocateAnything generation only supports `use_cache=True`.")

        generation_mode = generate_kwargs.get("generation_mode", "hybrid")
        if generation_mode not in ("fast", "slow", "hybrid"):
            raise ValueError(f"Unsupported generation_mode='{generation_mode}'. Use 'fast', 'slow', or 'hybrid'.")

        device = input_ids.device
        embed_tokens = self.language_model.get_input_embeddings()

        # Build the prompt embeddings once, scattering the projected image features into the
        # image placeholder positions (same merge used by `forward`).
        if visual_features is not None:
            vit_embeds = visual_features
        elif pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values, image_grid_hws)
            vit_embeds = torch.cat(vit_embeds, dim=0)
            vit_embeds = self.mlp1(vit_embeds)
        else:
            vit_embeds = None

        prompt_embeds = embed_tokens(input_ids)
        if vit_embeds is not None:
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=prompt_embeds, image_features=vit_embeds
            )
            prompt_embeds = prompt_embeds.masked_scatter(special_image_mask, vit_embeds.to(prompt_embeds.dtype))

        generated = input_ids.clone()
        total_gen_length = min(tokenizer.model_max_length, seq_len + generate_kwargs.get("max_new_tokens", 2048))
        past_key_values = None

        use_mtp = generation_mode in ("fast", "hybrid")
        block_size = n_future_tokens
        default_mask_token_id = self.token_ids["default_mask_token_id"]
        pre_mask_tokens = torch.full(
            (batch_size, n_future_tokens - 1), default_mask_token_id, dtype=generated.dtype, device=device
        )
        max_possible_len = total_gen_length + n_future_tokens
        full_position_ids = torch.arange(0, max_possible_len, device=device).unsqueeze(0)

        def _embed_new(sequence, past_len):
            """Embed the not-yet-cached suffix of `sequence`, reusing merged prompt embeds for the prefill."""
            if past_len == 0:
                # First forward: the prompt (with merged image features) plus any appended tokens.
                if sequence.size(1) > seq_len:
                    return torch.cat([prompt_embeds, embed_tokens(sequence[:, seq_len:])], dim=1)
                return prompt_embeds
            return embed_tokens(sequence[:, past_len:])

        while generated.size(1) < total_gen_length:
            commit_len = generated.size(1)
            past_len = past_key_values.get_seq_length() if past_key_values is not None else 0

            if use_mtp:
                sequence = torch.cat((generated, generated[:, -1:], pre_mask_tokens), dim=1)
                position_ids = full_position_ids[:, past_len : sequence.size(1)].clone()
                position_ids[0, -n_future_tokens:] -= 1
                q_len = sequence.size(1) - past_len
                attn = build_window_attention_mask(
                    q_len, sequence.size(1), past_len, block_size, prompt_embeds.dtype, device
                )
            else:
                sequence = generated
                position_ids = full_position_ids[:, past_len : sequence.size(1)]
                attn = torch.ones((batch_size, sequence.size(1)), dtype=torch.long, device=device)

            inputs_embeds = _embed_new(sequence, past_len)
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                position_ids=position_ids,
                past_key_values=past_key_values if past_key_values is not None else DynamicCache(),
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            # Roll back the speculative window so only committed tokens stay cached.
            past_key_values.crop(commit_len)

            if use_mtp:
                next_token_logits = outputs.logits[:, -n_future_tokens:, :]
                _, _, x0, box_avg = sample_tokens(
                    next_token_logits, generated, self.token_ids, keep_k=5, **generate_kwargs
                )
                is_box_empty = (box_avg[0] == 0).all()
                new_tokens = x0[0] if is_box_empty else box_avg[0]
                out_pattern = handle_pattern(new_tokens, self.token_ids, generation_mode)
                out_type = out_pattern["type"]
                out_token = torch.tensor(out_pattern["tokens"], dtype=generated.dtype, device=device)
            else:
                next_token_logits = outputs.logits[:, -1:, :]
                _, _, x0, _ = sample_tokens(next_token_logits, generated, self.token_ids, **generate_kwargs)
                out_token = x0[0]
                out_type = self._classify_ar_token(out_token[0].item(), generation_mode)

            generated = torch.cat([generated, out_token.unsqueeze(0)], dim=1)

            if out_type == "im_end":
                break
            if generation_mode == "hybrid":
                if out_type == "error_box":
                    use_mtp = False
                elif out_type == "box_end_ar":
                    use_mtp = True

        response = tokenizer.batch_decode(generated[:, seq_len:], skip_special_tokens=False)
        return response[0]

    def _classify_ar_token(self, token_val: int, generation_mode: str) -> str:
        """Classify a single AR token to drive hybrid mode switching and termination."""
        if generation_mode == "hybrid":
            if token_val == self.token_ids["box_end_token_id"]:
                return "box_end_ar"
            if (
                self.token_ids["coord_start_token_id"] <= token_val <= self.token_ids["coord_end_token_id"]
                or token_val == self.token_ids["none_token_id"]
            ):
                return "coord_ar"
            return "im_end"
        return "im_end" if token_val == self.token_ids["im_end_token_id"] else "continue_ar"


__all__ = ["LocateAnythingForConditionalGeneration", "LocateAnythingPreTrainedModel", "MoonViTPreTrainedEncoder"]
