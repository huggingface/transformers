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
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torchvision.transforms.v2 import functional as tvF

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...image_processing_backends import PilBackend, TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...integrations import (
    use_kernel_forward_from_hub,
    use_kernel_func_from_hub_with_fallback,
    use_kernelized_func,
)
from ...integrations.accelerate import force_accelerate_hooks
from ...masking_utils import create_recurrent_attention_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast, MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
    torch_compilable_check,
)
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ...video_utils import VideoMetadata, group_videos_by_shape, reorder_videos
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE, DeepseekV3TopkRouter
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection
from ..exaone4_5.modeling_exaone4_5 import Exaone4_5_Model
from ..glm46v.modeling_glm46v import Glm46VForConditionalGeneration
from ..glm46v.processing_glm46v import Glm46VProcessor
from ..glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from ..glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention, GlmMoeDsaDecoderLayer, GlmMoeDsaIndexer
from ..glm_ocr.configuration_glm_ocr import GlmOcrVisionConfig
from ..glm_ocr.modeling_glm_ocr import (
    GlmOcrVisionBlock,
    GlmOcrVisionMlp,
    GlmOcrVisionModel,
    GlmOcrVisionPatchMerger,
)
from ..glmga.image_processing_glmga import GlmgaImageProcessor, GlmgaImageProcessorKwargs
from ..glmga.image_processing_pil_glmga import GlmgaImageProcessorPil
from ..glmga.video_processing_glmga import GlmgaVideoProcessor, GlmgaVideoProcessorInitKwargs
from ..inkling.modeling_inkling import causal_conv1d_fn, causal_conv1d_update
from ..llama.modeling_llama import LlamaRMSNorm, eager_attention_forward
from ..minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLExperts
from ..mixtral.modeling_mixtral import load_balancing_loss_func
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
from ..qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated
from ..qwen3_next.modeling_qwen3_next import apply_mask_to_padding_states


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="zai-org/GLM-5.3-Flash")
@strict
class Glm5NextTextConfig(GlmMoeDsaConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of routed expert groups.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer feed-forward schedule. Values are `"dense"` or `"sparse"`.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of sparse-attention positions selected by the DSA indexer.
    index_head_dim (`int`, *optional*, defaults to 128):
        DSA indexer projection head dimension.
    index_n_heads (`int`, *optional*, defaults to 32):
        Number of DSA indexer heads.
    layer_types (`list[str]`, *optional*):
        Per-layer attention cache schedule. Values are `"linear_attention"` for
        KDA layers and `"deepseek_sparse_attention"` for MLA (DSA) layers.
    indexer_types (`list[str]`, *optional*):
        Per-layer DSA indexer mode. Values are `"full"` (run the indexer) or `"shared"`
        (reuse the previous full layer's top-k selection).
    swiglu_limit (`float`, *optional*, defaults to 10.0):
        Clamp limit applied to SwiGLU gate/up projections.
    linear_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each head in linear attention.
    linear_num_heads (`int`, *optional*, defaults to 64):
        Number of heads used in linear attention layers.
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel size of the convolution used in linear attention layers.
    linear_lower_bound (`float`, *optional*, defaults to -5.0):
        Whether the forget gate has a lower bound to apply to the decay.
    hc_mult (`int`, *optional*, defaults to 4):
        Number of MHC residual streams.
    hc_eps (`float`, *optional*, defaults to 1e-6):
        Numerical floor used by MHC Sinkhorn normalization.
    hc_sinkhorn_iters (`int`, *optional*, defaults to 20):
        Number of Sinkhorn iterations used by MHC routing.
    index_kpool (`int`, *optional*, defaults to 16):
        Pool size of the compressed token groups selected by the DSA indexer.
    index_kpool_always_select_tail (`bool`, *optional*, defaults to `True`):
        Whether the incomplete KPool tail is always included in sparse attention.
    """

    model_type = "glm5_next_text"
    base_config_key = "text_config"

    num_hidden_layers: int = 45
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    head_dim: int = 0
    max_position_embeddings: int = 1048576
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    moe_intermediate_size: int = 2048
    num_experts_per_tok: int = 8
    n_routed_experts: int = 288
    swiglu_limit: float = 10.0
    linear_head_dim: int = 128
    linear_num_heads: int = 64
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float | None = -5.0
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001

    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = 154820

    index_kpool: int = 16
    index_kpool_always_select_tail: bool = True

    mlp_bias = AttributeError()
    rope_parameters = AttributeError()
    first_k_dense_replace = AttributeError()

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(3, self.num_hidden_layers) + ["sparse"] * (
                self.num_hidden_layers - 3
            )

        if self.layer_types is None:
            kda_layers = [idx for idx in range(self.num_hidden_layers) if idx % 4 != 3]
            self.layer_types = [
                "linear_attention" if layer_idx in kda_layers else "deepseek_sparse_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]
        self.layer_types = [
            "deepseek_sparse_attention" if layer_type == "full_attention" else layer_type
            for layer_type in self.layer_types
        ]

        # Per-layer indexer mode: a pattern (e.g. `"FSSF..."`) overrides the freq/offset schedule.
        if self.indexer_types is None:
            pattern = kwargs.get("index_topk_pattern")
            if pattern is not None:
                self.indexer_types = (
                    [{"F": "full", "S": "shared"}[c] for c in pattern] if isinstance(pattern, str) else list(pattern)
                )
            else:
                freq = max(kwargs.get("index_topk_freq", 1), 1)
                offset = kwargs.get("index_skip_topk_offset", 2)
                self.indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared" for i in range(self.num_hidden_layers)
                ]

        # Convert dict to attributes (if given)
        if (linear_attn_dict := kwargs.get("linear_attn_config")) is not None:
            self.linear_head_dim = linear_attn_dict.get("head_dim", self.linear_head_dim)
            self.linear_num_heads = linear_attn_dict.get("num_heads", self.linear_num_heads)
            self.linear_conv_kernel_dim = linear_attn_dict.get("short_conv_kernel_size", self.linear_conv_kernel_dim)
            self.linear_lower_bound = linear_attn_dict.get("gate_lower_bound", self.linear_lower_bound)

            # Additional lower bound logic as per original dict
            if linear_attn_dict.get("safe_gate", True) and self.linear_lower_bound is None:
                self.linear_lower_bound = -5.0

        # NOTE: this forces an intentional override as we have the convention of head_dim being the RoPE based dim
        kwargs.pop("head_dim", None)
        self.head_dim = self.qk_rope_head_dim
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim

        PreTrainedConfig.__post_init__(self, **kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.num_attention_heads != self.num_key_value_heads:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be the same as "
                f"num_key_value_heads ({self.num_key_value_heads})."
            )

        if self.index_kpool < 1:
            raise ValueError(f"index_kpool must be positive, got {self.index_kpool}.")

        if self.index_topk % self.index_kpool != 0:
            raise ValueError(f"index_topk ({self.index_topk}) must be divisible by index_kpool ({self.index_kpool}).")

        if self.q_lora_rank is None:
            raise ValueError("For DSA usage in the attention layers, the `q_lora_rank` is strictly required!")

        if self.qk_rope_head_dim > 0:
            raise ValueError(
                f"Expecting NoPE for the DSA attention layers, but got {self.qk_rope_head_dim} as RoPE dim."
            )


@auto_docstring(checkpoint="zai-org/GLM-5.3-Flash")
@strict
class Glm5NextVisionConfig(GlmOcrVisionConfig):
    r"""
    out_hidden_size (`int`, *optional*, defaults to 1536):
        The output hidden size of the vision model.
    projection_intermediate_size (`int`, *optional*, defaults to 10240):
        The projection_intermediate_size size for the vision patch merger.
    swiglu_limit (`float`, *optional*, defaults to 10.0):
        Clamp limit applied to the vision SwiGLU gate/up projections.
    """

    model_type = "glm5_next_vision"
    projection_intermediate_size: int = 10240
    swiglu_limit: float = 10.0


@auto_docstring(checkpoint="zai-org/GLM-5.3-Flash")
@strict
class Glm5NextConfig(PreTrainedConfig):
    r"""
    image_token_id (`int`, *optional*, defaults to 154854):
        The image token index to encode the image prompt.
    video_token_id (`int`, *optional*, defaults to 154855):
        The video token index to encode the video prompt.
    image_start_token_id (`int`, *optional*, defaults to 154830):
        The image start token index to encode the start of image.
    image_end_token_id (`int`, *optional*, defaults to 154831):
        The image end token index to encode the end of image.
    video_start_token_id (`int`, *optional*, defaults to 154832):
        The video start token index to encode the start of video.
    video_end_token_id (`int`, *optional*, defaults to 154833):
        The video end token index to encode the end of video.

    ```python
    >>> from transformers import Glm5NextConfig

    >>> # Initializing a GLM-5.3-Flash style configuration
    >>> configuration = Glm5NextConfig()
    ```"""

    model_type = "glm5_next"
    sub_configs = {"vision_config": Glm5NextVisionConfig, "text_config": Glm5NextTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 154854
    video_token_id: int = 154855
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            # Flat (text-only) GLM-5.3-Flash checkpoints store the text fields at the
            # top level; forward them so `text_config` is populated for BC.
            self.text_config = self.sub_configs["text_config"](**kwargs)

        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        super().__post_init__(**kwargs)


class Glm5NextTextRMSNorm(LlamaRMSNorm):
    pass


class Glm5NextTextMLP(Qwen2MoeMLP):
    def __init__(self, config, intermediate_size=None):
        super().__init__(config)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Key difference using clamping
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextTextExperts(MiniMaxM3VLExperts):
    def __init__(self, config):
        super().__init__(config)
        del self.limit
        del self.swiglu_alpha
        self.intermediate_dim = config.moe_intermediate_size

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        # Simple swiglu instead of alpha
        return F.silu(gate) * up


class Glm5NextTextTopkRouter(DeepseekV3TopkRouter):
    pass


class Glm5NextTextMoE(DeepseekV3MoE):
    def __init__(self, config: Glm5NextConfig):
        super().__init__(config)
        self.experts = Glm5NextTextExperts(config)
        self.gate = Glm5NextTextTopkRouter(config)
        self.shared_experts = Glm5NextTextMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )


class Glm5NextTextHyperConnection(DeepseekV4HyperConnection):
    pass


class Glm5NextTextHyperHead(nn.Module):
    """Final GLM-5.3-Flash HC-stream collapse. Unlike DeepSeek-V4, this is an unweighted mean."""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)


class Glm5NextTextForgetGate(nn.Module):
    def __init__(self, config: Glm5NextTextConfig):
        super().__init__()
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.qkv_dim = self.head_dim * self.num_heads

        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.qkv_dim))
        self.A_log = nn.Parameter(torch.empty(self.num_heads))

        self.safe_gate_lower_bound = config.linear_lower_bound

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_shape = (*hidden_states.shape[:2], -1, self.head_dim)

        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (forget_gate.float() + self.dt_bias.float().view(1, 1, -1)).view(hidden_shape)
        A_log = self.A_log.float().view(1, 1, self.num_heads, 1)
        decay_rate = torch.exp(A_log)

        # Safe lower bound decay
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * torch.sigmoid(decay_rate * g)

        # Softplus "log(1 + exp(x))" with uper bound restraint to avoid overflows
        # NOTE: Softplus for larger values (e.g. 20+), Softplus(x) == x
        g_softplus = torch.where(g > 20.0, g, torch.log(1.0 + torch.exp(g)))

        return -decay_rate * g_softplus


@use_kernel_forward_from_hub("RMSNormGated")
class Glm5NextTextRMSNormGated(Qwen3_5RMSNormGated):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__(hidden_size, eps, kwargs)
        self.activation = "sigmoid"

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype

        # Strict FP32 norm (do not downcast on the weights)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states

        # Apply gating
        hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))

        return hidden_states.to(input_dtype)


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """
    This function is intended to align with the l2norm implementation in the FLA library.

    # NOTE: FLA compares against `F.normalize` but does + eps instead of max(..., eps) leading to a slight differences
    """
    # main difference to qwen's gdn variation: intentionally use sqrt and / to match original triton
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


@use_kernel_func_from_hub_with_fallback("fused_recurrent_kda", "fla")
def recurrent_kimi_delta_attention(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    # calculations happen in float as states are more susceptible to rounding errors
    initial_dtype = query.dtype
    query, key, value, g, beta = [x.to(torch.float32) for x in (query, key, value, g, beta)]

    # important: FLA calculates these in fp32 so we do this after the float casts
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # shapes and other metadata
    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, sequence_length, num_heads, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    # recurrent iteration
    for i in range(sequence_length):
        q_i = query[:, i]
        k_i = key[:, i]
        v_i = value[:, i]
        g_i = g[:, i][..., None].exp()
        b_i = beta[:, i][..., None]

        last_recurrent_state = last_recurrent_state * g_i
        kv_mem = (last_recurrent_state * k_i[..., None]).sum(dim=-2)
        delta = (v_i - kv_mem) * b_i

        last_recurrent_state = last_recurrent_state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, i] = (last_recurrent_state * q_i.unsqueeze(-1)).sum(dim=-2)

    return core_attn_out.to(initial_dtype), last_recurrent_state if output_final_state else None


@use_kernel_func_from_hub_with_fallback("chunk_kda", "fla")
def chunk_kimi_delta_attention(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    # calculations happen in float as states are more susceptible to rounding errors
    initial_dtype = query.dtype

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    # important: FLA calculates these in fp32 so we do this after the float casts
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # shapes and other metadata
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    total_sequence_length = sequence_length + pad_size

    # prepare all the relevant input
    query = F.pad(query, (0, 0, 0, pad_size)) * scale
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # reshape to chunks
    query, key, value, g, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, g, k_beta, v_beta)
    ]
    beta = beta.reshape(beta.shape[0], beta.shape[1], -1, chunk_size)

    # Intra chunk
    # Main difference to GDN is the per head application of `g` which was broadcasted across heads instead
    g = g.cumsum(dim=-2)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp().float()
    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(total_sequence_length // chunk_size):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        g_i = g[:, :, i]

        # Inter chunk
        attn_inter = (q_i * g_i.exp()) @ last_recurrent_state
        # Intra chunk
        attn_intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i]).sum(dim=-1).masked_fill(mask, 0)
        # New update rule
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime

        core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new
        last_recurrent_state = (
            last_recurrent_state * g_i[:, :, -1].exp().unsqueeze(-1)
            + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)

    return core_attn_out, last_recurrent_state


@use_kernelized_func(
    [chunk_kimi_delta_attention, recurrent_kimi_delta_attention, causal_conv1d_fn, causal_conv1d_update]
)
class Glm5NextTextLinearAttention(nn.Module):
    """Kimi-style KDA (Kimi Linear Attention) for GLM-5.3-Flash."""

    def __init__(
        self,
        config: Glm5NextTextConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.qkv_dim = self.head_dim * self.num_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        self.q_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)

        self.conv_dim = self.qkv_dim * 3
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.forget_gate = Glm5NextTextForgetGate(config)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.o_norm = Glm5NextTextRMSNormGated(self.head_dim, eps=self.layer_norm_epsilon)
        self.o_proj = nn.Linear(self.qkv_dim, self.hidden_size, bias=False)

        self.layer_type = config.layer_types[layer_idx]

    @force_accelerate_hooks("conv1d")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        # Zero out padding
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        mixed_qkv = torch.cat(
            [
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            ],
            dim=-1,
        ).transpose(1, 2)

        # Acts for normal prefill but also for multi-token prefill continue
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0]

        # Single token decode path
        if use_precomputed_states and seq_len == 1:
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        # Multi token prefill or simple "full" prefill
        else:
            # Concatenated state for prefill
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )

            # Cut out any tail
            mixed_qkv = mixed_qkv[:, :, -seq_len:]

        query, key, value = torch.split(
            mixed_qkv.transpose(1, 2),
            [self.qkv_dim] * 3,
            dim=-1,
        )

        query = query.view(hidden_shape)
        key = key.view(hidden_shape)
        value = value.view(hidden_shape)

        # Forget gate and input gate
        g = self.forget_gate(hidden_states)
        beta = torch.sigmoid(self.b_proj(hidden_states))

        # KDA
        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = recurrent_kimi_delta_attention(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                **kwargs,
            )
        else:
            core_attn_out, last_recurrent_state = chunk_kimi_delta_attention(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                **kwargs,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state.to(torch.float32), self.layer_idx)

        # Final gated norm and proj
        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
        output = self.o_norm(core_attn_out, gate).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output


class Glm5NextTextIndexer(GlmMoeDsaIndexer):
    """
    DeepSeek Sparse Attention (DSA) indexer with k-pool compression for GLM-5.3-Flash.

    The indexer uses lightweight projections (`wq_b`, `wk`) separate from the main MLA
    attention path. It scores compressed k-pool candidates, expands selected pools back
    into raw cache indices, and optionally appends the current incomplete tail pool.

    **Cache strategy**: the indexer state cache lives on the per-layer `DynamicIndexedLayer`
    (or the `StaticIndexedLayer` for static caches) inside the shared cache, accessed via
    `past_key_values.update_indexer()`.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.index_kpool = config.index_kpool
        self.index_kpool_always_select_tail = config.index_kpool_always_select_tail

        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(self.index_kpool, self.head_dim))
        self.index_kpool_compress_gate = nn.Parameter(torch.zeros(self.head_dim, self.hidden_size))

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        attention_mask: torch.BoolTensor,
        past_key_values: Cache | None,
    ) -> torch.LongTensor:
        """
        Selects the top-k tokens per query for DeepSeek Sparse Attention (DSA) based on grouping pools (and tails).

        Args:
            hidden_states: Input hidden states `[B, S, hidden_size]`.
            q_resid: Query residual from `q_a_layernorm(q_a_proj(x))`, shape `[B, S, q_lora_rank]`.
            attention_mask: Local boolean padding mask of shape `[B, S]`.
            past_key_values: Cache object containing the indexer state cache for this layer.

        Returns:
            `torch.Tensor`: the `int32` top-k token indices of shape `[B, S, topk]` (or `[B, S, 2*topk - 1]` with tail).
            The eager / SDPA paths turn these into an additive sparse mask.
        """
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        q = self.wq_b(q_resid).view(hidden_shape)
        k = self.k_norm(self.wk(hidden_states)).view(hidden_shape).squeeze(2)

        gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
        valid_channel = attention_mask.to(k.dtype)[..., None]

        packed_states = torch.cat([k, gate_scores, valid_channel], dim=-1)

        kv_len = seq_len
        current_length = seq_len
        if past_key_values is not None:
            cache_layer = past_key_values.layers[self.layer_idx]

            packed_states = past_key_values.update_indexer(packed_states, self.layer_idx)
            # Only different on static caches where key is a static full tensor to max len
            kv_len = cache_layer.keys.shape[-2]
            current_length = cache_layer.get_seq_length()

        # Get pools based on the valid key entries (based on padding / causality)
        valid_keys = packed_states[..., -1].bool()
        visible_tokens = self.get_visible_tokens(
            valid_keys=valid_keys,
            q_length=seq_len,
            current_length=current_length,
        )

        # Key difference: Score across pools, not on a per token basis
        pool_keys, pool_indices, pool_valid = self.get_pooled_states(packed_states=packed_states)
        scores = torch.matmul(q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores * self.softmax_scale)

        # Weight per head and sum across heads: [B, S, 1, H] @ [B, S, H, P] -> [B, S, P]
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        # Clamp invalid / static pool ends
        pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
        pool_visible = visible_tokens.gather(
            dim=-1,
            index=pool_end[:, None, :].expand(batch_size, seq_len, -1),
        )
        # A pool is selectable only if its final token is visible to the query
        valid_candidates = pool_visible & pool_valid[:, None]

        index_scores = index_scores.masked_fill(
            ~valid_candidates,
            torch.finfo(index_scores.dtype).min,
        )

        # Similar budgeting as in original but compressed by its pool size
        select_k = min(self.index_topk // self.index_kpool, index_scores.shape[-1])

        # Selection is based on 2 steps
        #   1. The actual scores (selected indices)
        #   2. And removing invalid rows based on padding (selected valid)
        selected = index_scores.topk(select_k, dim=-1).indices
        batch_idx = torch.arange(batch_size, device=hidden_states.device)[:, None, None]

        selected_valid = valid_candidates.gather(-1, selected)
        selected_indices = pool_indices[batch_idx, selected]

        # Convert selected pools back into the raw tokens
        # [B, S, K, P] -> [B, S, K * P]
        topk_indices = selected_indices.flatten(-2)
        topk_indices = topk_indices.masked_fill(
            ~selected_valid[..., None].expand_as(selected_indices).flatten(-2),
            -1,
        )

        output_width = self.index_topk
        if self.index_kpool_always_select_tail:
            topk_indices = self.append_visible_tail(topk_indices, visible_tokens, valid_keys)
            output_width += self.index_kpool - 1  # expanded tail size maximum

        # Pad so we fill up with invalid entries instead of gathered selections
        topk_indices = F.pad(topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1)

        topk_indices = topk_indices[..., :output_width]
        topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)

        return topk_indices.to(torch.int32)

    def get_visible_tokens(
        self,
        valid_keys: torch.BoolTensor,
        q_length: int,
        current_length: int,
    ) -> torch.BoolTensor:
        """
        Check whether a token is visible for Q based on
            - Causality
            - Padding status
                - Valid keys which is a (cached) variation of the attention mask
        """
        device = valid_keys.device

        kv_positions = torch.arange(valid_keys.shape[-1], device=device)
        q_positions = current_length - q_length + torch.arange(q_length, device=device)
        causal = kv_positions[None, None, :] <= q_positions[None, :, None]

        return causal & valid_keys[:, None, :]

    def get_pooled_states(
        self,
        packed_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.BoolTensor]:
        """
        (Re-)Build compressed k-pool candidates from the packed state.

        Pooling starts at the first real token, not raw slot 0. This is the part
        that makes:

            [P, P, A, B, C, D, ...]

        behave like:

            [A, B, C, D, ...]

        for k-pool grouping.
        """
        # All states we need
        #   1. The actual keys
        #   2. The compressed gate scores
        #   3. The valid keys based on padding and causality
        keys, gate_scores, valid_keys = torch.split(
            packed_states,
            [self.head_dim, self.head_dim, 1],
            dim=-1,
        )
        valid_keys = valid_keys.bool().squeeze(-1)

        # Metadata
        batch_size, seq_len = keys.shape[:2]
        number_of_pools = (seq_len + self.index_kpool - 1) // self.index_kpool
        device = keys.device

        # Example, index_kpool=4:
        #   [P, P, A, B, C, D, E, F]
        #          ^
        #      first_key
        #
        #   pool 0 -> first_key + [0, 1, 2, 3] -> [A, B, C, D]
        #   pool 1 -> first_key + [4, 5, 6, 7] -> [E, F, out, out]
        first_key = torch.where(
            valid_keys.any(-1),
            valid_keys.long().argmax(-1),
            torch.full((batch_size,), seq_len, dtype=torch.long, device=device),
        )
        pool_offsets = torch.arange(number_of_pools * self.index_kpool, device=device)
        pool_offsets = pool_offsets.view(1, number_of_pools, self.index_kpool)
        pool_indices = first_key[:, None, None] + pool_offsets

        batch_idx = torch.arange(batch_size, device=device)[:, None, None]
        safe_indices = pool_indices.clamp(0, seq_len - 1)

        grouped_keys = keys[batch_idx, safe_indices]
        grouped_gate_scores = gate_scores[batch_idx, safe_indices]
        grouped_valid_keys = valid_keys[batch_idx, safe_indices]

        # Only allow those within range (clamp)
        grouped_valid_keys = grouped_valid_keys & (pool_indices < seq_len)
        pool_valid = grouped_valid_keys.all(-1)
        pool_indices = pool_indices.masked_fill(~grouped_valid_keys, -1)

        # Learn a weighted average over the tokens inside each complete pool
        logits = grouped_gate_scores.float() + self.index_kpool_compress_ape.float()[None, None]
        logits = logits.masked_fill(~grouped_valid_keys[..., None], float("-inf"))
        probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(
            grouped_keys.dtype
        )  # nan to num for full invalid pools
        pool_keys = (probabilities * grouped_keys).sum(dim=2)

        # Avoids static cache allocated positions
        keep = pool_valid.any(0)

        return pool_keys[:, keep], pool_indices[:, keep], pool_valid[:, keep]

    def append_visible_tail(
        self,
        topk_indices: torch.Tensor,
        token_visible: torch.BoolTensor,
        key_valid: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Append the current incomplete pool as raw token indices.
        So we if we have a pool size of 4:
            - [P P A B C D E F]
            - [A B C D] (selected pool)
            - [E F] (appended tail)
        """
        if (max_tail_width := self.index_kpool - 1) == 0:
            return topk_indices

        batch_size, _, kv_length = token_visible.shape
        device = token_visible.device

        # Example, index_kpool=4:
        #   visible keys: [A, B, C, D, E, F]
        #   full pools:   [A, B, C, D]
        #   tail:                     [E, F]
        #
        # visible_count = 6
        # tail_count    = 6 % 4 = 2
        # tail_start    = first_key + visible_count - tail_count
        # tail_indices  = tail_start + [0, 1, ..., index_kpool - 2]
        first_key = torch.where(
            key_valid.any(-1),
            key_valid.long().argmax(-1),
            torch.full((batch_size,), kv_length, dtype=torch.long, device=device),
        )
        visible_count = token_visible.long().sum(-1)
        tail_count = visible_count.remainder(self.index_kpool)
        tail_offsets = torch.arange(max_tail_width, device=device)

        tail_start = first_key[:, None] + visible_count - tail_count
        tail_indices = tail_start[..., None] + tail_offsets

        # We exclude tails that are just use to fill in positions + those that go beyond the max length
        tail_valid = (tail_offsets[None, None, :] < tail_count[..., None]) & tail_indices.lt(kv_length)

        # Also check for padding based tokens
        kv_idx = tail_indices.clamp(0, kv_length - 1)
        tail_visible = token_visible.gather(dim=-1, index=kv_idx)

        # Get the valid conclusion
        tail_indices = tail_indices.masked_fill(~(tail_valid & tail_visible), -1)

        return torch.cat([topk_indices, tail_indices], dim=-1)


class Glm5NextTextAttention(GlmMoeDsaAttention):
    def __init__(self, config: Glm5NextTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.scaling = self.qk_head_dim ** (-0.5)
        self.q_a_layernorm = (
            Glm5NextTextRMSNorm(config.q_lora_rank, eps=config.rms_norm_eps) if self.q_lora_rank is not None else None
        )
        self.kv_a_layernorm = Glm5NextTextRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.indexer = None if self.skip_topk else Glm5NextTextIndexer(config, layer_idx)
        self.next_skip_topk = (
            not self.skip_topk and config.indexer_types[min(layer_idx + 1, len(config.indexer_types) - 1)] == "shared"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        # LoRA based path is guaranteed based on the config validation
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, self.kv_lora_rank)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        key_states, value_states = self.expand_kv(k_pass, k_rot)

        # Cache update
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if self.indexer is not None:
            topk_indices = self.indexer(
                hidden_states=hidden_states,
                q_resid=q_resid,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
        else:
            if prev_topk_indices is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            topk_indices = prev_topk_indices

        attention_mask = self.build_attention_mask_from_topk(
            topk_indices=topk_indices,
            query_states=query_states,
            kv_length=key_states.shape[2],
        )

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, topk_indices if self.next_skip_topk else None

    def build_attention_mask_from_topk(
        self,
        topk_indices: torch.Tensor,
        query_states: torch.Tensor,
        kv_length: int,
    ) -> torch.Tensor | None:
        """
        Convert topk_indices into the mask expected by the active backend.

        Only supporting SDPA and Eager as we have a 3D dependency which cannot be mapped to FA
        without a custom kernel that can select on a per indices bases per row (query -> topk keys).
        """
        # -1 is invalid as per convention in the indexer
        # NOTE: The indexer already took care of also excluding padding tokens and causality
        topk_valid = topk_indices.ge(0) & topk_indices.lt(kv_length)

        # Clamp only so scatter has a legal index
        safe_indices = topk_indices.clamp(0, kv_length - 1)
        selected_counts = torch.zeros(
            topk_indices.shape[0],  # batch size
            topk_indices.shape[1],  # q_length
            kv_length,  # kv_length
            dtype=torch.int32,
            device=topk_indices.device,
        )
        selected_counts.scatter_add_(-1, safe_indices, topk_valid.to(torch.int32))

        # Final mask 0 == False (not visible), 1 == True (visible)
        mask = selected_counts.ne(0).unsqueeze(1)

        # SDPA
        if self.config._attn_implementation == "sdpa":
            return mask

        # Eager
        min_dtype = torch.finfo(query_states.dtype).min
        # we need 0s where the tokens should be taken into account, and -inf otherwise (mask is already of boolean type)
        mask = torch.where(mask, torch.full((), 0.0, device=query_states.device, dtype=query_states.dtype), min_dtype)
        return mask


class Glm5NextTextDecoderLayer(GlmMoeDsaDecoderLayer):
    def __init__(self, config: Glm5NextTextConfig, layer_idx: int):
        self.block_type = config.layer_types[layer_idx]

        super().__init__(config, layer_idx)
        self.self_attn = (
            Glm5NextTextLinearAttention(config, layer_idx)
            if self.block_type == "linear_attention"
            else Glm5NextTextAttention(config, layer_idx)
        )

        self.attn_hc = Glm5NextTextHyperConnection(config)
        self.ffn_hc = Glm5NextTextHyperConnection(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dtype = hidden_states.dtype

        residual = hidden_states
        post, comb, hidden_states = self.attn_hc(hidden_states)
        # Self attn
        hidden_states = self.input_layernorm(hidden_states)
        topk_indices = None
        if self.block_type == "linear_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            hidden_states, _, topk_indices = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                prev_topk_indices=prev_topk_indices,
                **kwargs,
            )
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

        residual = hidden_states
        post, comb, hidden_states = self.ffn_hc(hidden_states)
        # Feed forward
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

        return hidden_states, topk_indices


@auto_docstring
class Glm5NextPreTrainedModel(PreTrainedModel):
    config: Glm5NextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    # needs index based kernel
    _supports_flash_attn = False
    _supports_sdpa = True
    # needs per layer creation, too expensive
    _supports_flex_attn = False
    _supports_attention_backend = True

    _no_split_modules = ["Glm5NextTextDecoderLayer", "Glm5NextVisionBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    # TODO: this can be fixed but is limited by
    # 1. assuming the cache name
    # 2. linear attention not being considered atm
    _is_stateful = True
    _can_compile_fullgraph = True

    _can_record_outputs = {
        "attentions": Glm5NextTextAttention,
        "hidden_states": Glm5NextTextDecoderLayer,
        "router_logits": OutputRecorder(Glm5NextTextTopkRouter, index=0),
    }
    _keep_in_fp32_modules_strict = ["e_score_correction_bias", "conv1d", "dt_bias", "A_log"]
    _keys_to_ignore_on_load_unexpected = [r"layers\.45\.", r"layers\.\d+\.shared_head\."]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Glm5NextTextForgetGate):
            # Following FLA initialization
            # NOTE: This is incredibly important so keep it this way at all costs
            if module.safe_gate_lower_bound is not None:
                init.zeros_(module.A_log)
            else:
                init.copy_(
                    module.A_log,
                    init.uniform_(module.A_log, a=1.0, b=16.0).log(),
                )

            init.uniform_(
                module.dt_bias,
                a=math.log(1e-3),
                b=math.log(1e-1),
            )
            dt = module.dt_bias.exp().clamp_min(1e-4)

            # (stable) inverse softplus
            init.copy_(
                module.dt_bias,
                dt + torch.log(-torch.expm1(-dt)),
            )
        elif isinstance(module, Glm5NextTextRMSNormGated):
            init.ones_(module.weight)
        elif isinstance(module, Glm5NextTextHyperConnection):
            init.normal_(module.fn, mean=0.0, std=0.02)
            init.zeros_(module.base)
            init.ones_(module.scale)
        elif isinstance(module, Glm5NextTextExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Glm5NextTextTopkRouter):
            init.zeros_(module.e_score_correction_bias)
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Glm5NextTextIndexer):
            init.zeros_(module.index_kpool_compress_ape)
            init.ones_(module.index_kpool_compress_gate)
        elif isinstance(module, Glm5NextVisionRotaryEmbedding):  # noqa: F821
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


# Do not inherit from DSv4 as it messes modular prefixes up for the PreTrainedModel
@auto_docstring
class Glm5NextTextModel(Glm5NextPreTrainedModel):
    config: Glm5NextTextConfig

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Glm5NextTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Glm5NextTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.hc_head = Glm5NextTextHyperHead()

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            attention_mask = create_recurrent_attention_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            # Guarantee the mask to exist for the indexer
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.shape[0],
                    inputs_embeds.shape[1],
                    dtype=torch.bool,
                    device=inputs_embeds.device,
                )
            attention_mask = attention_mask.bool()

            causal_mask_mapping = {
                "deepseek_sparse_attention": attention_mask,
                "linear_attention": attention_mask,
            }

        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

        topk_indices = None
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states, topk_indices = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                # Key change using NoPE
                position_embeddings=None,
                input_ids=input_ids,
                past_key_values=past_key_values,
                prev_topk_indices=topk_indices,
                **kwargs,
            )

        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class Glm5NextVisionMLP(GlmOcrVisionMlp):
    def __init__(self, config, bias: bool = True):
        super().__init__(config, bias=bias)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_state):
        gate = self.gate_proj(hidden_state)
        up = self.up_proj(hidden_state)
        # Key difference using clamping
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextVisionPatchMerger(GlmOcrVisionPatchMerger):
    def __init__(self, dim: int, context_dim: int, hidden_act: str, swiglu_limit: float, bias: bool = False) -> None:
        super().__init__(dim=dim, context_dim=context_dim, hidden_act=hidden_act, bias=bias)
        self.swiglu_limit = swiglu_limit

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.proj(hidden_state)
        hidden_state = self.act1(self.post_projection_norm(hidden_state))
        gate = self.gate_proj(hidden_state)
        up = self.up_proj(hidden_state)
        # Key difference using clamping
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextVisionBlock(GlmOcrVisionBlock):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.mlp = Glm5NextVisionMLP(config, bias=config.attention_bias)


@auto_docstring
class Glm5NextVisionModel(GlmOcrVisionModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.blocks = nn.ModuleList([Glm5NextVisionBlock(config) for _ in range(config.depth)])
        self.merger = Glm5NextVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.projection_intermediate_size,
            hidden_act=config.hidden_act,
            swiglu_limit=config.swiglu_limit,
        )


class Glm5NextModel(Exaone4_5_Model, Glm5NextPreTrainedModel):
    config: Glm5NextConfig

    def __init__(self, config):
        super().__init__(config)
        self.visual = Glm5NextVisionModel._from_config(config.vision_config)
        self.language_model = Glm5NextTextModel._from_config(config.text_config)
        del self.rope_deltas

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        # Same as in `Glm46V`
        # reshape video_grid_thw -> [b, 3] -> [1, h, w] * frames
        t = video_grid_thw[:, 0]
        hw = video_grid_thw[:, 1:]
        # repeat each (h,w) row `t` times
        flattened_hw = torch.repeat_interleave(hw, t, dim=0)
        prefix_ones = video_grid_thw.new_ones(flattened_hw.shape[0], 1)
        flattened_video_grid_thw = torch.cat([prefix_ones, flattened_hw], dim=1)

        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values_videos, grid_thw=flattened_video_grid_thw, **kwargs)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        vision_outputs.pooler_output = torch.split(vision_outputs.pooler_output, split_sizes)
        return vision_outputs

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_mm_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_mm_mask = special_mm_mask.all(-1)
            video_start_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), self.config.video_start_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            video_start_mask = video_start_mask.all(-1)
            video_end_mask = inputs_embeds == self.get_input_embeddings()(
                torch.full((), self.config.video_end_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            video_end_mask = video_end_mask.all(-1)
            in_video_span = video_start_mask.cumsum(-1) > video_end_mask.cumsum(-1)
        else:
            special_mm_mask = input_ids == self.config.image_token_id
            in_video_span = (input_ids == self.config.video_start_token_id).cumsum(-1) > (
                input_ids == self.config.video_end_token_id
            ).cumsum(-1)

        # Core difference to other VLMs as img token == vid token so we differentiate by start/end spans instead
        special_image_mask = special_mm_mask & ~in_video_span
        special_video_mask = special_mm_mask & in_video_span

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)
        if image_features is not None:
            torch_compilable_check(
                n_image_tokens * inputs_embeds.shape[-1] == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).to(inputs_embeds.device)
        if video_features is not None:
            torch_compilable_check(
                n_video_tokens * inputs_embeds.shape[-1] == video_features.numel(),
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, features: {video_features.shape[0]}",
            )
        return special_image_mask, special_video_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, **kwargs).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, **kwargs).pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        # Only change is the output type to Moe
        return MoeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


@auto_docstring
class Glm5NextForConditionalGeneration(Glm46VForConditionalGeneration, Glm5NextPreTrainedModel):
    """
    Main Glm5Next conditional generation class.
    """

    def __init__(self, config):
        super().__init__(config)
        self.router_aux_loss_coef = config.text_config.router_aux_loss_coef
        self.num_experts = config.text_config.num_local_experts
        self.num_experts_per_tok = config.text_config.num_experts_per_tok

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        output_router_logits: bool | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoProcessor, Glm5NextForConditionalGeneration
        >>> import torch

        >>> model = Glm5NextForConditionalGeneration.from_pretrained("zai-org/GLM-5.3-Flash")
        >>> processor = AutoProcessor.from_pretrained("zai-org/GLM-5.3-Flash")

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        ...             {"type": "text", "text": "Describe the image."},
        ...         ],
        ...     }
        ... ]
        >>> inputs = processor.apply_chat_template(
        ...     messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ... )
        >>> inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        >>> generated_ids = model.generate(**inputs, max_new_tokens=64)
        ```
        """

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        raise AttributeError()

    @staticmethod
    def create_masks_for_generate(config, inputs_embeds, attention_mask, past_key_values, **_):
        # We only use the base 2D mask as the indexer is reliant on the padding, not the expanded masks.
        # I.e. 4D masks are built afterwards after subsets have been selected in the indexer.
        # Linear attention can reuse the mask as is, making the layer type difference only
        # necessary for the cache.
        attention_mask = create_recurrent_attention_mask(
            config=config.get_text_config(),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        # Guarantee the mask to exist for the indexer
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[0],
                inputs_embeds.shape[1],
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = attention_mask.bool()

        return {"deepseek_sparse_attention": attention_mask, "linear_attention": attention_mask}


class Glm5NextProcessor(Glm46VProcessor):
    pass


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,
    min_pixels: int = 16,
    max_pixels: int = 8000,
) -> tuple[int, int]:
    """Compute an aligned canvas within the spatiotemporal pixel budget."""

    # Dynamically adjust pixel count
    # TODO: possibly integrate directly into size dict (into the values)
    pixels_per_token = temporal_factor * factor**2
    min_pixels *= pixels_per_token
    max_pixels *= pixels_per_token

    def align(value, factor):
        return math.ceil(value / factor) * factor

    def fit_within_budget(aligned_frames):
        minimum_pixels = aligned_frames * factor**2
        if max_pixels < minimum_pixels:
            raise ValueError(
                f"max_pixels={max_pixels} is too small. "
                f"At least {minimum_pixels} pixels are required for one aligned patch."
            )

        low, high = 1, height
        best_height, best_width = factor, factor
        # Iteratively go over the allowed budget space until resized ratio has been found
        while low <= high:
            content_height = (low + high) // 2
            content_width = max(1, math.floor(width * content_height / height))
            candidate_height = align(content_height, factor)
            candidate_width = align(content_width, factor)

            pixel_budget = aligned_frames * candidate_height * candidate_width
            if pixel_budget <= max_pixels:
                best_height, best_width = candidate_height, candidate_width
                low = content_height + 1
            else:
                high = content_height - 1
        return best_height, best_width

    aligned_frames = max(temporal_factor, round(num_frames / temporal_factor) * temporal_factor)
    aligned_height = align(height, factor)
    aligned_width = align(width, factor)
    aligned_pixel_budget = aligned_frames * aligned_height * aligned_width

    # Readjust budget if too little is found
    if aligned_pixel_budget < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        aligned_height = align(max(1, math.ceil(height * scale)), factor)
        aligned_width = align(max(1, math.ceil(width * scale)), factor)
        aligned_pixel_budget = aligned_frames * aligned_height * aligned_width

    # Cut into budget
    if aligned_pixel_budget > max_pixels:
        aligned_height, aligned_width = fit_within_budget(aligned_frames)

    return aligned_height, aligned_width


class Glm5NextImageProcessorKwargs(GlmgaImageProcessorKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    patch_expand_factor (`int`, *optional*, defaults to 1):
        The patch_expand_factor of the vision encoder to llm encoder.
    min_image_tokens (`int`):
        Minimum number of tokens per image.
    max_image_tokens (`int`):
        Maximum number of tokens per image.
    """

    min_image_tokens: int
    max_image_tokens: int


class Glm5NextImageProcessor(GlmgaImageProcessor):
    size = {"longest_edge": 1}  # TODO: Refactor afterwards to be included within
    min_image_tokens = 16
    max_image_tokens = 8000
    valid_kwargs = Glm5NextImageProcessorKwargs

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Glm5NextImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        images: "torch.Tensor",
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        factor: int,
        temporal_factor: int,
        min_image_tokens: int,
        max_image_tokens: int,
        **kwargs,
    ) -> "torch.Tensor":
        """Resize dynamically based on input image aspect ratio."""

        height, width = images.shape[-2:]
        target_height, target_width = smart_resize(
            height=height,
            width=width,
            num_frames=temporal_factor,
            factor=factor,
            temporal_factor=temporal_factor,
            min_pixels=min_image_tokens,
            max_pixels=max_image_tokens,
        )

        # Dynamic padded to ensure aspect ratio is compatible with `_patchify`
        pixels_per_token = temporal_factor * factor**2
        scale = min(target_height / height, target_width / width)
        if temporal_factor * height * width >= (pixels_per_token * min_image_tokens):
            scale = min(1.0, scale)
        content_height = max(1, min(target_height, math.floor(height * scale)))
        content_width = max(1, min(target_width, math.floor(width * scale)))

        # TODO: Also likely refactorable after min/max pixels has been added to size dict
        if (content_height, content_width) != (height, width):
            images = TorchvisionBackend.resize(
                self, images, SizeDict(height=content_height, width=content_width), resample=resample
            )

        return tvF.pad(images, [0, 0, target_width - content_width, target_height - content_height], fill=0)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,  # TODO: Ignored for now, refactoring after min/max pixels PRs has been merged
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        patch_expand_factor: int,
        min_image_tokens: int,
        max_image_tokens: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images.
        """

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                # New resize requires new kwargs to be passed downstream
                stacked_images = self.resize(
                    images=stacked_images,
                    resample=resample,
                    factor=patch_size * merge_size * patch_expand_factor,
                    temporal_factor=temporal_patch_size,
                    min_image_tokens=min_image_tokens,
                    max_image_tokens=max_image_tokens,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}

        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches, grid_h, grid_w = self.patchify(
                stacked_images,
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )

            processed_images_grouped[shape] = patches
            processed_grids[shape] = [[1, grid_h, grid_w]] * len(stacked_images)

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids = reorder_images(processed_grids, grouped_images_index)

        pixel_values = processed_images[0] if len(processed_images) == 1 else torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        """
        A utility that returns number of image patches for a given image size.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of image patches per image.
        """
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        # Key difference is the dynamically based resize on min/max image tokens
        min_image_tokens = images_kwargs.get("min_image_tokens", self.min_image_tokens)
        max_image_tokens = images_kwargs.get("max_image_tokens", self.max_image_tokens)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            num_frames=self.temporal_patch_size,
            height=height,
            width=width,
            factor=factor,
            min_pixels=min_image_tokens,
            max_pixels=max_image_tokens,
            temporal_factor=self.temporal_patch_size,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


class Glm5NextImageProcessorPil(GlmgaImageProcessorPil):
    size = {"longest_edge": 1}  # TODO: Refactor afterwards to be included within
    min_image_tokens = 16
    max_image_tokens = 8000
    valid_kwargs = Glm5NextImageProcessorKwargs

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Glm5NextImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        image: np.ndarray,
        resample: "PILImageResampling | int | None",
        factor: int,
        temporal_factor: int,
        min_image_tokens: int,
        max_image_tokens: int,
        **kwargs,
    ) -> np.ndarray:
        """Resize dynamically based on input image aspect ratio."""

        height, width = image.shape[-2:]
        target_height, target_width = smart_resize(
            height=height,
            width=width,
            num_frames=temporal_factor,
            factor=factor,
            temporal_factor=temporal_factor,
            min_pixels=min_image_tokens,
            max_pixels=max_image_tokens,
        )

        # Dynamic padded to ensure aspect ratio is compatible with `_patchify`
        pixels_per_token = temporal_factor * factor**2
        scale = min(target_height / height, target_width / width)
        if temporal_factor * height * width >= (pixels_per_token * min_image_tokens):
            scale = min(1.0, scale)
        content_height = max(1, min(target_height, math.floor(height * scale)))
        content_width = max(1, min(target_width, math.floor(width * scale)))

        # TODO: Also likely refactorable after min/max pixels has been added to size dict
        if (content_height, content_width) != (height, width):
            image = PilBackend.resize(
                self, image, SizeDict(height=content_height, width=content_width), resample=resample
            )

        return np.pad(
            image,
            ((0, 0), (0, target_height - content_height), (0, target_width - content_width)),
            mode="constant",
        )

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_expand_factor: int,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        min_image_tokens: int,
        max_image_tokens: int,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess images one by one for PIL backend.
        """
        processed_images = []
        processed_grids = []

        for image in images:
            if do_resize:
                image = self.resize(
                    image,
                    resample=resample,
                    factor=patch_size * merge_size,
                    temporal_factor=temporal_patch_size,
                    min_image_tokens=min_image_tokens,
                    max_image_tokens=max_image_tokens,
                )

            # Rescale and normalize
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)

            patches, grid_h, grid_w = self.patchify(
                image,
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )

            # Remove batch dimension and append: shape is (seq_len, hidden_dim)
            processed_images.append(patches)
            processed_grids.append([1, grid_h, grid_w])

        # Concatenate all images along sequence dimension: (total_seq_len, hidden_dim)
        pixel_values = np.concatenate(processed_images, axis=0)
        image_grid_thw = np.array(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        """
        A utility that returns number of image patches for a given image size.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of image patches per image.
        """
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        # Key difference is the dynamically based resize on min/max image tokens
        min_image_tokens = images_kwargs.get("min_image_tokens", self.min_image_tokens)
        max_image_tokens = images_kwargs.get("max_image_tokens", self.max_image_tokens)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            num_frames=self.temporal_patch_size,
            height=height,
            width=width,
            factor=factor,
            min_pixels=min_image_tokens,
            max_pixels=max_image_tokens,
            temporal_factor=self.temporal_patch_size,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


class Glm5NextVideoProcessorInitKwargs(GlmgaVideoProcessorInitKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to 14):
        The spacial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    patch_expand_factor (`int`, *optional*, defaults to 1):
        The patch_expand_factor of the vision encoder to llm encoder.
    max_frames (`int`, *optional*, defaults to 2048):
        The maximum number of frames that can be sampled.
    max_image_size (`dict`, *optional*, defaults to `28 * 28 * 2 * 30000`):
        The maximum pixels a video can be resized to.
    min_image_tokens (`int`):
        Minimum number of tokens per image.
    max_image_tokens (`int`):
        Maximum number of tokens per image.
    """

    min_image_tokens: int
    max_image_tokens: int


class Glm5NextVideoProcessor(GlmgaVideoProcessor):
    size = {"longest_edge": 1}
    max_image_size = {"longest_edge": 28 * 28 * 2 * 30000}
    min_image_tokens = 16
    max_image_tokens = 240000
    max_duration = 0
    max_frames = 2048
    valid_kwargs = Glm5NextVideoProcessorInitKwargs

    def __init__(self, **kwargs: Unpack[Glm5NextVideoProcessorInitKwargs]):
        super().__init__(**kwargs)

    def sample_frames(
        self,
        metadata: VideoMetadata,
        fps: int | float | None = None,
        **kwargs,
    ):
        if metadata is None or getattr(metadata, "fps", None) is None:
            raise ValueError(
                "Asked to sample frames per second but no video metadata was provided which is required when sampling in Glm5Next. "
                "Please pass in a VideoMetadata object or set do_sample_frames=False"
            )

        total_frames = metadata.total_num_frames
        max_frame_idx = total_frames - 1
        duration = metadata.duration or round(max_frame_idx / metadata.fps) + 1
        # Used later to cap frames, important to base on the original and not capped duration
        max_seconds = int(duration)
        duration = duration if self.max_duration <= 0 else min(duration, self.max_duration)
        target_fps = fps if fps is not None else self.fps

        extract_t = int(duration * target_fps)
        extract_t = min(extract_t, self.max_frames)

        duration_per_frame = 1 / metadata.fps
        timestamps = [i * duration_per_frame for i in range(total_frames)]

        # Key change in the framed indices
        # 1. Use linspace instead floored manual ranges
        # 2. Cap by static max secon value
        if total_frames < extract_t:
            frame_indices = np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()
        else:
            frame_indices = []
            current_second = 0
            inv_fps = 1 / target_fps
            for frame_index in range(total_frames):
                if timestamps[frame_index] >= current_second:
                    current_second += inv_fps
                    frame_indices.append(frame_index)
                    if current_second >= max_seconds:
                        break

        if len(frame_indices) < extract_t:
            if len(frame_indices) == 0:
                start, end = 0, max(total_frames - 1, 0)
            else:
                start, end = frame_indices[0], frame_indices[-1]
            frame_indices = np.linspace(start, end, extract_t, dtype=int).tolist()
        elif len(frame_indices) > extract_t:
            frame_indices = np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()

        seen, uniq = set(), []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                uniq.append(idx)

        if len(uniq) & 1:
            uniq.append(uniq[-1])

        return np.array(uniq)

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: SizeDict | None = None,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_expand_factor: int | None = None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        merge_size: int | None = None,
        min_image_tokens: int | None = None,
        max_image_tokens: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ):
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}

        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            if do_resize:
                # New resize requires new kwargs to be passed downstream
                stacked_videos = self.resize(
                    videos=stacked_videos,
                    resample=resample,
                    factor=patch_size * merge_size * patch_expand_factor,
                    temporal_factor=temporal_patch_size,
                    min_image_tokens=min_image_tokens,
                    max_image_tokens=max_image_tokens,
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)

            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches, grid_t, grid_h, grid_w = self.patchify(
                stacked_videos,
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )

            processed_videos_grouped[shape] = patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * len(stacked_videos)

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)
        data = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

    def resize(
        self,
        videos: "torch.Tensor",
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        factor: int,
        temporal_factor: int,
        min_image_tokens: int,
        max_image_tokens: int,
        **kwargs,
    ) -> "torch.Tensor":
        """Resize dynamically based on input video aspect ratio."""

        height, width = videos.shape[-2:]
        target_height, target_width = smart_resize(
            height=height,
            width=width,
            num_frames=videos.shape[1],
            factor=factor,
            temporal_factor=temporal_factor,
            min_pixels=min_image_tokens,
            max_pixels=max_image_tokens,
        )

        # Dynamic padded to ensure aspect ratio is compatible with `_patchify`
        pixels_per_token = temporal_factor * factor**2
        scale = min(target_height / height, target_width / width)
        if videos.shape[1] * height * width >= (pixels_per_token * min_image_tokens):
            scale = min(1.0, scale)
        content_height = max(1, min(target_height, math.floor(height * scale)))
        content_width = max(1, min(target_width, math.floor(width * scale)))

        # TODO: Also likely refactorable after min/max pixels has been added to size dict
        if (content_height, content_width) != (height, width):
            videos = TorchvisionBackend.resize(
                self, videos, SizeDict(height=content_height, width=content_width), resample=resample
            )

        return tvF.pad(videos, [0, 0, target_width - content_width, target_height - content_height], fill=0)


__all__ = [
    "Glm5NextConfig",
    "Glm5NextTextConfig",
    "Glm5NextVisionConfig",
    "Glm5NextPreTrainedModel",
    "Glm5NextTextModel",
    "Glm5NextVisionModel",
    "Glm5NextModel",
    "Glm5NextForConditionalGeneration",
    "Glm5NextProcessor",
    "Glm5NextImageProcessor",
    "Glm5NextImageProcessorPil",
    "Glm5NextVideoProcessor",
]
