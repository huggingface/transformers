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

from collections import deque
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
)
from ...utils.generic import get_max_seqlen, is_flash_attention_requested
from ...utils.output_capturing import capture_outputs
from ...vision_utils import (
    get_vision_attention_seqlens,
    get_vision_position_ids,
)
from ..kimi_k25.configuration_kimi_k25 import Kimi_K25Config, Kimi_K25VisionConfig
from ..kimi_k25.modeling_kimi_k25 import (
    Kimi_K25CausalLMOutputWithPast,
    Kimi_K25ForConditionalGeneration,
    Kimi_K25Model,
    Kimi_K25ModelOutputWithPast,
    Kimi_K25MultimodalProjection,
    Kimi_K25VisionAttention,
    Kimi_K25VisionEncoderLayer,
    Kimi_K25VisionMLP,
    Kimi_K25VisionModel,
    Kimi_K25VisionPatchEmbed,
    Kimi_K25VisionPositionEmbeddings,
    Kimi_K25VisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
    get_vision_temporal_merge_index,
    repeat_kv,
)
from ..shensi.configuration_shensi import ShensiConfig
from ..shensi.modeling_shensi import (
    ShensiAttention,
    ShensiAttentionResidual,
    ShensiCSACache,
    ShensiCSACompressor,
    ShensiDecoderLayer,
    ShensiExperts,
    ShensiGroupedLinear,
    ShensiHCACache,
    ShensiHCACompressor,
    ShensiHyperConnection,
    ShensiHyperHead,
    ShensiIndexer,
    ShensiIndexerScorer,
    ShensiMLP,
    ShensiPreTrainedModel,
    ShensiRMSNorm,
    ShensiRotaryEmbedding,
    ShensiSparseMoeBlock,
    ShensiTopKRouter,
    ShensiUnweightedRMSNorm,
    compute_loss_func,
    create_sliding_window_causal_mask,
)
from ..vjepa2.modeling_vjepa2 import VJEPA2PoolerCrossAttention


@auto_docstring(checkpoint="louzongzhi/Shensi-VL-Nano")
@strict
class ShensiVlTextConfig(ShensiConfig):
    pass


@auto_docstring(checkpoint="louzongzhi/Shensi-VL-Nano")
@strict
class ShensiVlVisionConfig(Kimi_K25VisionConfig):
    r"""
    pos_emb_height (`int`, *optional*):
        Initial position embedding height.
    pos_emb_width (`int`, *optional*):
        Initial position embedding width.
    pos_emb_time (`int`, *optional*):
        Initial position embedding time dimension.
    merge_kernel_size (`tuple[int] | list[int]`, *optional*):
        Kernel size for patch merging.
    qkv_hidden_size (`int`, *optional*):
        Hidden size of the fused QKV projection in the vision attention.
    attn_res_block_size (`int`, *optional*):
        Layers are grouped into blocks of B, and each block's first layer writes its delta.
    """

    num_attention_heads: int = 12
    num_hidden_layers: int = 27
    hidden_size: int = 1024
    intermediate_size: int = 4096
    qkv_hidden_size: int = 1536
    attn_res_block_size: int = 3

    @property
    def attn_res_block_layer_types(self) -> list[str]:
        return [
            "block_write_layer" if i % self.attn_res_block_size == 0 else "block_read_layer"
            for i in range(self.num_hidden_layers)
        ]


@auto_docstring(checkpoint="louzongzhi/Shensi-VL-Nano")
@strict
class ShensiVlConfig(Kimi_K25Config):
    r"""
    projection_hidden_size (`int`, *optional*):
        The output hidden size for multimodal projector.
    projection_layer_norm_eps (`float`, *optional*):
        Layer norm epsilon for projector.
    num_cross_attention_heads (`int`, *optional*):
        Number of heads in the cross-modal attention modules of the DeepRecur blocks.
    """

    sub_configs = {"text_config": ShensiVlTextConfig, "vision_config": ShensiVlVisionConfig}
    projection_hidden_size: int | None = 1024
    num_cross_attention_heads: int = 16

    def __post_init__(self, **kwargs):
        if isinstance(self.text_config, dict):
            self.text_config = ShensiVlTextConfig(**self.text_config)
        elif self.text_config is None:
            self.text_config = ShensiVlTextConfig()

        if isinstance(self.vision_config, dict):
            self.vision_config = ShensiVlVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = ShensiVlVisionConfig()
        PreTrainedConfig.__post_init__(**kwargs)

    # Flat proxies for the vLLM engine's attention metadata builders, which
    # read these off the top-level config (the engine never sees text_config).
    @property
    def index_topk(self) -> int:
        return self.text_config.index_topk

    @property
    def sliding_window(self) -> int:
        return self.text_config.sliding_window

    @property
    def compress_ratios(self) -> list[int]:
        return self.text_config.compress_ratios


class ShensiVlRMSNorm(ShensiRMSNorm):
    pass


class ShensiVlUnweightedRMSNorm(ShensiUnweightedRMSNorm):
    pass


class ShensiVlRotaryEmbedding(ShensiRotaryEmbedding):
    pass


class ShensiVlHCACache(ShensiHCACache):
    pass


class ShensiVlCSACache(ShensiCSACache):
    pass


class ShensiVlGroupedLinear(ShensiGroupedLinear):
    pass


class ShensiVlHCACompressor(ShensiHCACompressor):
    pass


class ShensiVlIndexerScorer(ShensiIndexerScorer):
    pass


class ShensiVlIndexer(ShensiIndexer):
    pass


class ShensiVlCSACompressor(ShensiCSACompressor):
    pass


class ShensiVlAttention(ShensiAttention):
    pass


class ShensiVlMLP(ShensiMLP):
    pass


class ShensiVlTopKRouter(ShensiTopKRouter):
    pass


class ShensiVlExperts(ShensiExperts):
    pass


class ShensiVlSparseMoeBlock(ShensiSparseMoeBlock):
    pass


class ShensiVlHyperConnection(ShensiHyperConnection):
    pass


class ShensiVlHyperHead(ShensiHyperHead):
    pass


class ShensiVlAttentionResidual(ShensiAttentionResidual):
    def __init__(self, config, has_router: bool = True):
        super().__init__()
        self.norm = ShensiVlUnweightedRMSNorm(1e-5)


class ShensiVlDecoderLayer(ShensiDecoderLayer):
    pass


class ShensiVlVisionPositionEmbeddings(Kimi_K25VisionPositionEmbeddings):
    def __init__(self, config):
        super().__init__()
        self.interpolation_mode = "bilinear"


class ShensiVlVisionPatchEmbed(Kimi_K25VisionPatchEmbed):
    def __init__(self, config):
        super().__init__()
        patch_size = (
            config.patch_size if not isinstance(config.patch_size, int) else (config.patch_size, config.patch_size)
        )
        self.proj = nn.Conv2d(3, config.hidden_size, kernel_size=patch_size, stride=patch_size, bias=False)


class ShensiVlVisionRotaryEmbedding(Kimi_K25VisionRotaryEmbedding):
    def compute_default_rope_parameters(
        config: ShensiVlVisionConfig, device=None, **kwargs
    ) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.qkv_hidden_size // config.num_attention_heads

        # The reference implementation computes RoPE frequencies INDEPENDENTLY
        # for each spatial dimension using the partitioned head_dim (head_dim // ndim),
        # so both x and y dimensions get identical frequency ranges.
        # This is different from splitting the global inv_freq between dimensions.
        spatial_dim = dim // 2

        attention_factor = 1.0  # Unused in this type of RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
        return inv_freq.to(device), attention_factor


def vision_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class ShensiVlVisionAttention(Kimi_K25VisionAttention):
    def __init__(self, config: ShensiVlVisionConfig) -> None:
        super().__init__()
        del self.proj
        del self.q_proj
        del self.k_proj
        del self.v_proj
        self.head_dim = config.qkv_hidden_size // self.num_heads
        self.wqkv = nn.Linear(self.dim, config.qkv_hidden_size * 3, bias=False)
        self.wo = nn.Linear(config.qkv_hidden_size, self.dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        max_seqlen: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]

        query_states, key_states, value_states = self.wqkv(hidden_states).chunk(3, dim=-1)

        query_states = query_states.reshape(1, seq_length, -1, self.head_dim)
        key_states = key_states.reshape(1, seq_length, -1, self.head_dim)
        value_states = value_states.reshape(1, seq_length, -1, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(2, 1)
        key_states = key_states.transpose(2, 1)
        value_states = value_states.transpose(2, 1)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, vision_attention_forward
        )

        if is_flash_attention_requested(self.config):
            # Flash Attention: Use cu_seqlens for variable length attention
            max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs={"max_seqlen": max_seqlen})
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.wo(attn_output)


class ShensiVlVisionMLP(Kimi_K25VisionMLP):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)


class ShensiVlVisionEncoderLayer(Kimi_K25VisionEncoderLayer):
    def __init__(self, config: ShensiVlVisionConfig, layer_idx: int) -> None:
        super().__init__()
        self.norm1 = ShensiVlRMSNorm(config.hidden_size, eps=1e-5)
        self.norm2 = ShensiVlRMSNorm(config.hidden_size, eps=1e-5)
        self.is_block_write_layer = config.attn_res_block_layer_types[layer_idx] == "block_write_layer"
        self.prev_valid_blocks = sum(
            1 for layer_type in config.attn_res_block_layer_types[:layer_idx] if layer_type == "block_write_layer"
        )
        self.block_write_idx = self.prev_valid_blocks
        self.self_attention_attn_res = ShensiVlAttentionResidual(config, self.prev_valid_blocks > 0)
        self.mlp_attn_res = ShensiVlAttentionResidual(config, self.prev_valid_blocks + self.is_block_write_layer > 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        entry = hidden_states if hidden_states is not None else prefix_sum
        delta = hidden_states - prefix_sum if hidden_states is not None else None

        hidden_states, prefix_sum, residual = self.self_attention_attn_res(
            delta, residual, prefix_sum, output_norm_weight=self.norm1.weight, num_blocks=self.prev_valid_blocks
        )
        if self.is_block_write_layer:
            residual = torch.cat(
                [
                    residual[..., : self.block_write_idx, :],
                    entry.to(residual.dtype).unsqueeze(-2),
                    residual[..., self.block_write_idx + 1 :, :],
                ],
                dim=-2,
            )
            prefix_sum = None

        attn_output = self.attn(
            hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs
        )
        hidden_states = hidden_states + attn_output
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states

        hidden_states, prefix_sum, residual = self.mlp_attn_res(
            prefix_sum,
            residual,
            prefix_sum,
            output_norm_weight=self.norm2.weight,
            num_blocks=self.prev_valid_blocks + self.is_block_write_layer,
        )

        mlp_output = self.mlp(hidden_states)
        hidden_states = hidden_states + mlp_output
        prefix_sum = prefix_sum + hidden_states

        return hidden_states, prefix_sum, residual


class ShensiVlPreTrainedModel(ShensiPreTrainedModel):
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        config = getattr(self.config, "text_config", None) or self.config
        std = getattr(config, "initializer_range", 0.02)
        if isinstance(module, ShensiVlTopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, ShensiVlExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, ShensiVlAttention):
            init.zeros_(module.sinks)
        elif isinstance(module, ShensiVlHyperConnection):
            init.normal_(module.pre_fn, mean=0.0, std=std)
            init.zeros_(module.pre_base)
            init.constant_(module.pre_scale, 0.01)
            init.normal_(module.route_fn, mean=0.0, std=std)
            init.zeros_(module.route_base)
            init.ones_(module.route_scale)
            init.normal_(module.post_fn, mean=0.0, std=std)
            init.zeros_(module.post_base)
            init.constant_(module.post_scale, 0.01)
        elif isinstance(module, ShensiVlHyperHead):
            init.normal_(module.hc_fn, mean=0.0, std=std)
            init.zeros_(module.hc_base)
            init.ones_(module.hc_scale)
        elif isinstance(module, (ShensiVlHCACompressor, ShensiVlCSACompressor, ShensiVlIndexer)):
            init.zeros_(module.position_bias)
        elif isinstance(module, ShensiVlRotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)
        elif isinstance(module, ShensiVlAttentionResidual):
            gate_width = module.gate_proj.out_features // 3
            nn.init.zeros_(module.gate_proj.weight)
            with torch.no_grad():
                # decay gate open, erase/write gates closed: preserve the prefix by default.
                module.gate_proj.bias[:gate_width] = 2.0
                module.gate_proj.bias[gate_width : 2 * gate_width] = -2.0
                module.gate_proj.bias[2 * gate_width :] = -2.0
            if module.q_proj is not None:
                nn.init.zeros_(module.q_proj)
            nn.init.normal_(module.k_proj, mean=0.0, std=std)
        elif isinstance(module, ShensiVlVisionPositionEmbeddings):
            buffer_value = module.compute_pos_embed()
            init.copy_(module.time_position_embeddings, buffer_value)
            init.trunc_normal_(module.position_embeddings, mean=0.0)

    def _share_block_routers_and_experts(self):
        for block in self.blocks:
            pool_mlp = None
            moe_positions = []
            for pos, layer in enumerate(block.language_layers):
                if layer.mlp.is_hash:
                    continue
                if pool_mlp is None:
                    pool_mlp = layer.mlp
                else:
                    layer.mlp.gate = pool_mlp.gate
                    layer.mlp.experts = pool_mlp.experts
                moe_positions.append(pos)
            if pool_mlp is not None:
                pool_mlp.sharing_layers = moe_positions

    def post_init(self):
        if hasattr(self, "blocks") and self.blocks:
            self._share_block_routers_and_experts()
            seen_params: dict[int, str] = {}
            registry = getattr(self, "_tied_weights_keys", {}) or {}
            if not isinstance(registry, dict):
                registry = {key: key for key in registry}
            tied_keys = dict(registry)
            for name, param in self.named_parameters(remove_duplicate=False):
                if canonical := seen_params.get(id(param)):
                    tied_keys[name] = canonical
                else:
                    seen_params[id(param)] = name
            self._tied_weights_keys = tied_keys
            self.all_tied_weights_keys = tied_keys
        PreTrainedModel.post_init(self)


class ShensiVlVisionModel(Kimi_K25VisionModel):
    _can_record_outputs = {"hidden_states": ShensiVlVisionEncoderLayer}

    def __init__(self, config: ShensiVlVisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [ShensiVlVisionEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layernorm = ShensiVlRMSNorm(config.hidden_size, eps=1e-05)
        self.num_attn_res_blocks = config.attn_res_block_layer_types.count("block_write_layer")
        self.output_attn_res = ShensiVlAttentionResidual(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        hidden_states = self.patch_embed(pixel_values, grid_thw=grid_thw, **kwargs)
        position_ids = get_vision_position_ids(grid_thw, spatial_merge_size=1, kwargs=kwargs)
        position_ids = position_ids.transpose(0, 1).flip(0)  # (2, positions)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        cu_seqlens, max_seqlen = get_vision_attention_seqlens(
            grid_thw, self.config, merge_temporal=True, kwargs=kwargs
        )

        prefix_sum = hidden_states
        block_residual = hidden_states.new_zeros(
            *hidden_states.shape[:-1], self.num_attn_res_blocks, hidden_states.shape[-1]
        )
        hidden_states = None

        for block in self.layers:
            hidden_states, prefix_sum, block_residual = block(
                hidden_states,
                residual=block_residual,
                prefix_sum=prefix_sum,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states, _, _ = self.output_attn_res(
            hidden_states - prefix_sum if hidden_states is not None else None,
            block_residual,
            prefix_sum,
            output_norm_weight=self.final_layernorm.weight,
            num_blocks=self.num_attn_res_blocks,
        )

        merge_index = get_vision_temporal_merge_index(grid_thw, *self.merge_kernel_size, kwargs=kwargs)
        pooled_hidden_states = hidden_states[merge_index].mean(dim=1)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_hidden_states,
        )


# Adapted from transformers.models.vit.modeling_vit.eager_attention_forward
def cross_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class ShensiVlPoolerCrossAttention(VJEPA2PoolerCrossAttention):
    def __init__(self, q_dim, kv_dim, num_heads, _attn_implementation=None):
        super().__init__()
        del self.config
        self.embed_dim = q_dim
        self.num_heads = num_heads
        self.dropout = 0.0
        self.q_proj = nn.Linear(q_dim, q_dim)
        self.k_proj = nn.Linear(kv_dim, q_dim)
        self.v_proj = nn.Linear(kv_dim, q_dim)
        self._attn_implementation = _attn_implementation

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_seq_length, embed_dim = queries.shape
        kv_seq_length = keys.shape[1]

        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        queries = queries.view(batch_size, q_seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, kv_seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, kv_seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self._attn_implementation, cross_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(batch_size, q_seq_length, embed_dim).contiguous()

        return attn_output, attn_weights


class ShensiVlCrossBlock(GradientCheckpointingLayer):
    def __init__(
        self,
        config: ShensiVlConfig,
        vision_layers: Sequence[nn.Module],
        language_layers: Sequence[nn.Module],
    ):
        super().__init__()
        vision_config, text_config = config.vision_config, config.text_config

        self.vision_layers = tuple(vision_layers)
        self.language_layers = tuple(language_layers)
        self.vision_to_language_attention = ShensiVlPoolerCrossAttention(
            text_config.hidden_size,
            vision_config.hidden_size,
            config.num_cross_attention_heads,
        )
        self.language_to_vision_attention = ShensiVlPoolerCrossAttention(
            vision_config.hidden_size,
            text_config.hidden_size,
            config.num_cross_attention_heads,
        )
        self.gate = nn.Parameter(torch.zeros(text_config.hc_mult + 1))

    def _run_vision_layers(
        self,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        position_embeddings: tuple | None,
        max_seqlen: int | None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if hidden_states is not None:
            for layer in self.vision_layers:
                hidden_states, prefix_sum, residual = layer(
                    hidden_states,
                    residual,
                    prefix_sum,
                    cu_seqlens,
                    position_embeddings,
                    max_seqlen=max_seqlen,
                    **kwargs,
                )
        return hidden_states, prefix_sum, residual

    def _cross_modal_exchange(
        self,
        hidden_states: torch.Tensor | None,
        prefix_sum: torch.Tensor,
        vision_prefix_sum: torch.Tensor | None,
        evidence_mask: torch.Tensor | None,
        guidance_mask: torch.Tensor | None,
        vision_patch_to_row: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if vision_prefix_sum is None:
            return hidden_states, vision_prefix_sum

        # The packed vision sequence is shared across the batch; expand it so query and key
        # batch sizes match. Per-row masks make each sample attend only to its own patches.
        batch_size = prefix_sum.shape[0]

        vision_memory = vision_prefix_sum.unsqueeze(0).expand(batch_size, -1, -1)
        language_queries = prefix_sum + (hidden_states if hidden_states is not None else prefix_sum)
        # Collapse the hc streams before any cross-attention: both projectors live in the
        # language hidden-space, so query, key and value must all be 2-D in sequence-time.
        language_queries = language_queries.mean(dim=2)
        retrieved_evidence = self.vision_to_language_attention(
            language_queries,
            vision_memory,
            vision_memory,
            attention_mask=evidence_mask,
        )[0]
        if hidden_states is not None:
            # Split the fused gate along the stream dimension: `hc_mult` evidence gates
            # followed by a single guidance gate.
            evidence_gate, guidance_gate = self.gate.split([self.gate.size(0) - 1, 1])
            hidden_states = hidden_states + torch.tanh(evidence_gate)[None, None, :, None] * (
                retrieved_evidence.unsqueeze(2)
            )

        retrieved_guidance = self.language_to_vision_attention(
            vision_memory,
            language_queries,
            language_queries,
            attention_mask=guidance_mask,
        )[0]
        # Each patch lives in exactly one sample; keep only its own row's guidance update.
        patch_indices = torch.arange(vision_prefix_sum.shape[0], device=vision_prefix_sum.device)
        retrieved_guidance = retrieved_guidance[vision_patch_to_row, patch_indices]
        vision_prefix_sum = vision_prefix_sum + torch.tanh(guidance_gate) * retrieved_guidance

        return hidden_states, vision_prefix_sum

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor,
        prefix_sum: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        position_embeddings: dict | None = None,
        position_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        vision_hidden_states: torch.Tensor | None = None,
        vision_residual: torch.Tensor | None = None,
        vision_prefix_sum: torch.Tensor | None = None,
        vision_cu_seqlens: torch.Tensor | None = None,
        vision_position_embeddings: tuple | None = None,
        vision_max_seqlen: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        list[torch.Tensor] | None,
    ]:
        evidence_mask = kwargs.pop("evidence_mask", None)
        guidance_mask = kwargs.pop("guidance_mask", None)
        vision_patch_to_row = kwargs.pop("vision_patch_to_row", None)
        run_vision_layers = kwargs.pop("run_vision_layers", True)

        # Block 0's vision layers may already have run during round-0 preparation (the placeholder
        # fill needs refined features before any language layer); the caller says which.
        if run_vision_layers:
            vision_hidden_states, vision_prefix_sum, vision_residual = self._run_vision_layers(
                hidden_states=vision_hidden_states,
                residual=vision_residual,
                prefix_sum=vision_prefix_sum,
                cu_seqlens=vision_cu_seqlens,
                position_embeddings=vision_position_embeddings,
                max_seqlen=vision_max_seqlen,
                **kwargs,
            )

        layer_hidden_states = [] if kwargs.get("output_hidden_states", False) else None
        for layer in self.language_layers:
            hidden_states, prefix_sum, residual = layer(
                hidden_states,
                residual,
                prefix_sum,
                input_ids,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            if layer_hidden_states is not None:
                layer_hidden_states.append(hidden_states)

        hidden_states, vision_prefix_sum = self._cross_modal_exchange(
            hidden_states,
            prefix_sum,
            vision_prefix_sum,
            evidence_mask,
            guidance_mask,
            vision_patch_to_row,
        )

        return (
            hidden_states,
            prefix_sum,
            residual,
            vision_hidden_states,
            vision_prefix_sum,
            vision_residual,
            layer_hidden_states,
        )


class ShensiVlReasoning(nn.Module):
    def __init__(self, config: ShensiVlConfig):
        super().__init__()
        text_config = config.text_config

        self.gate = nn.Parameter(torch.ones(text_config.hc_mult))
        self.proj = nn.Linear(text_config.hidden_size, text_config.hidden_size)
        self.update_gate = nn.Linear(text_config.hidden_size * 2, text_config.hidden_size)

    def forward(
        self,
        retrieved_evidence: torch.Tensor,
        reasoning_state: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        evolving = self.proj(retrieved_evidence)
        update_gate = torch.sigmoid(self.update_gate(torch.cat([evolving, reasoning_state], dim=-1)))
        reasoning_state = update_gate * reasoning_state + (1 - update_gate) * evolving
        if hidden_states is not None:
            hidden_states + torch.tanh(self.gate)[None, None, :, None] * (
                retrieved_evidence + reasoning_state
            ).unsqueeze(2)
        else:
            hidden_states = None
        return reasoning_state, hidden_states


class ShensiVlLoopBlock(GradientCheckpointingLayer):
    def __init__(
        self,
        config: ShensiVlConfig,
        vision_layers: Sequence[nn.Module],
        language_layers: Sequence[nn.Module],
        vision_output_attn_res: nn.Module,
        output_attn_res: nn.Module,
        vision_final_norm: nn.Module,
    ):
        super().__init__()
        vision_config, text_config = config.vision_config, config.text_config

        self.vision_layers = tuple(vision_layers)
        self.language_layers = tuple(language_layers)
        self.vision_to_language_attention = ShensiVlPoolerCrossAttention(
            text_config.hidden_size,
            vision_config.hidden_size,
            config.num_cross_attention_heads,
        )
        object.__setattr__(self, "vision_output_attn_res", vision_output_attn_res)
        object.__setattr__(self, "output_attn_res", output_attn_res)
        object.__setattr__(self, "vision_final_norm", vision_final_norm)
        self.reasoning = ShensiVlReasoning(config)

    def _run_vision_layers(
        self,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        position_embeddings: tuple | None,
        max_seqlen: int | None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if hidden_states is not None:
            for layer in self.vision_layers:
                hidden_states, prefix_sum, residual = layer(
                    hidden_states,
                    residual,
                    prefix_sum,
                    cu_seqlens,
                    position_embeddings,
                    max_seqlen=max_seqlen,
                    **kwargs,
                )
        return hidden_states, prefix_sum, residual

    def _run_damped_language_layers(
        self,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor,
        prefix_sum: torch.Tensor,
        damping: float,
        past_key_values: Cache | None,
        input_ids: torch.LongTensor | None,
        position_embeddings: dict,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        damped_connections = []
        if damping < 1.0:
            for layer in self.language_layers:
                for connection in (layer.attn_hc, layer.ffn_hc):
                    original_write_back = connection.write_back
                    damped_connections.append((connection, original_write_back))
                    connection.write_back = (
                        lambda hidden, output, _original=original_write_back, _damping=damping: _original(
                            hidden, _damping * output
                        )
                    )
        try:
            layer_mask = attention_mask
            if past_key_values is None and attention_mask is not None:
                layer_mask = attention_mask[..., -prefix_sum.shape[1] :]
            layer_hidden_states = [] if kwargs.get("output_hidden_states", False) else None
            for layer in self.language_layers:
                hidden_states, prefix_sum, residual = layer(
                    hidden_states,
                    residual,
                    prefix_sum,
                    input_ids,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    attention_mask=layer_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )
                if layer_hidden_states is not None:
                    layer_hidden_states.append(hidden_states)
        finally:
            for connection, original_write_back in damped_connections:
                connection.write_back = original_write_back
        return hidden_states, prefix_sum, residual, layer_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor,
        prefix_sum: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        position_embeddings: dict | None = None,
        position_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        vision_hidden_states: torch.Tensor | None = None,
        vision_residual: torch.Tensor | None = None,
        vision_prefix_sum: torch.Tensor | None = None,
        vision_cu_seqlens: torch.Tensor | None = None,
        vision_position_embeddings: tuple | None = None,
        vision_max_seqlen: int | None = None,
        num_attn_res_blocks: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        list[torch.Tensor] | None,
    ]:
        evidence_mask = kwargs.pop("evidence_mask", None)
        run_vision_layers = kwargs.pop("run_vision_layers", True)

        if run_vision_layers:
            vision_hidden_states, vision_prefix_sum, vision_residual = self._run_vision_layers(
                hidden_states=vision_hidden_states,
                residual=vision_residual,
                prefix_sum=vision_prefix_sum,
                cu_seqlens=vision_cu_seqlens,
                position_embeddings=vision_position_embeddings,
                max_seqlen=vision_max_seqlen,
                **kwargs,
            )

        vision_final = (
            self.vision_output_attn_res(
                vision_hidden_states,
                vision_residual,
                vision_prefix_sum,
                output_norm_weight=self.vision_final_norm.weight,
                num_blocks=num_attn_res_blocks,
            )[0]
            if vision_hidden_states is not None
            else None
        )

        initial_queries = (prefix_sum + (hidden_states if hidden_states is not None else prefix_sum)).mean(dim=2)
        reasoning_state = torch.zeros_like(initial_queries)

        previous = None
        first_delta = None
        hidden_delta = None
        first_reasoning_delta = None
        reasoning_delta = None
        previous_reasoning_state = None
        delta_history = deque(maxlen=8)
        orbit_history = deque(maxlen=8)
        orbit_counts = deque(maxlen=8)
        while True:
            damping = 0.25
            converged = first_delta is not None and hidden_delta <= 1e-3 * first_delta
            thought_settled = first_reasoning_delta is not None and reasoning_delta <= 1e-1 * first_reasoning_delta
            stream_stalled = delta_history and max(delta_history) >= 0.5 * first_delta
            stop = bool(converged or (thought_settled and stream_stalled) or sum(orbit_counts) >= 3)

            language_queries = (prefix_sum + (hidden_states if hidden_states is not None else prefix_sum)).mean(dim=2)
            if vision_final is not None:
                final_evidence = vision_final.unsqueeze(0).expand(prefix_sum.shape[0], -1, -1)
                retrieved_evidence = self.vision_to_language_attention(
                    language_queries,
                    final_evidence,
                    final_evidence,
                    attention_mask=evidence_mask,
                )[0]
            else:
                retrieved_evidence = language_queries

            reasoning_state, hidden_states = self.reasoning(retrieved_evidence, reasoning_state, hidden_states)

            hidden_states, prefix_sum, residual, layer_hidden_states = self._run_damped_language_layers(
                hidden_states,
                residual,
                prefix_sum,
                damping,
                past_key_values if stop else None,
                input_ids,
                position_embeddings,
                position_ids,
                attention_mask,
                **{**kwargs, "output_attentions": kwargs.get("output_attentions", False) and stop},
            )
            if previous is not None:
                hidden_delta = (hidden_states - previous).abs().max()
                if first_delta is None:
                    first_delta = hidden_delta
                delta_history.append(hidden_delta)
            previous = hidden_states
            if previous_reasoning_state is not None:
                reasoning_delta = (reasoning_state - previous_reasoning_state).abs().max()
                if first_reasoning_delta is None:
                    first_reasoning_delta = reasoning_delta
            previous_reasoning_state = reasoning_state
            if orbit_history:
                orbit_return = min((hidden_states.detach() - past).abs().max() for past in orbit_history)
                orbit_history.append(hidden_states.detach())
                orbit_counts.append(1 if orbit_return <= 1e-1 * first_delta else 0)
            else:
                orbit_history.append(hidden_states.detach())
            if stop:
                break

        hidden_states, _, _ = self.output_attn_res(
            hidden_states,
            residual,
            prefix_sum,
            output_norm_weight=None,
            num_blocks=num_attn_res_blocks,
        )

        return (
            hidden_states,
            prefix_sum,
            residual,
            vision_hidden_states,
            vision_prefix_sum,
            vision_residual,
            layer_hidden_states,
        )


class ShensiVlMultimodalProjection(Kimi_K25MultimodalProjection):
    def __init__(self, config: ShensiVlConfig):
        super().__init__()
        del self.pre_norm
        del self.in_proj
        del self.act
        del self.out_proj
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_size, config.text_config.hidden_size, bias=False)
        self.post_norm = nn.RMSNorm(config.text_config.hidden_size, config.projection_layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.view(batch_size, -1, self.hidden_size)
        hidden_states = self.fc2(self.act(self.fc1(hidden_states)))
        return self.post_norm(hidden_states)


class ShensiVlModelOutputWithPast(Kimi_K25ModelOutputWithPast):
    pass


def find_spans(input_ids: torch.Tensor, token_id: int) -> list[list[tuple[int, int]]]:
    r"""
    Finds contiguous runs of `token_id` in each row of `input_ids`.

    Args:
        input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Input token ids.
        token_id (`int`):
            The media placeholder token to locate.

    Returns:
        `list[list[tuple[int, int]]]`: Per input row, a list of `(start, length)` spans.
    """
    spans = []
    for row in input_ids.tolist():
        row_spans, index = [], 0
        while index < len(row):
            if row[index] == token_id:
                end = index
                while end < len(row) and row[end] == token_id:
                    end += 1
                row_spans.append((index, end - index))
                index = end
            else:
                index += 1
        spans.append(row_spans)
    return spans


def get_clip_counts(
    grid_thw: torch.Tensor,
    kernel_height: int,
    kernel_width: int,
) -> tuple[list[int], list[int]]:
    r"""
    Computes per-clip patch counts from `grid_thw`, before and after temporal patch merging.

    Args:
        grid_thw (`torch.Tensor` of shape `(num_clips, 3)`):
            Per clip, the `(temporal, height, width)` patch grid.
        kernel_height (`int`):
            Spatial merge kernel height.
        kernel_width (`int`):
            Spatial merge kernel width.

    Returns:
        `tuple[list[int], list[int]]`: `(merged_counts, patch_counts)` — patches after
        merging (`h // kernel_height * w // kernel_width`) and before merging (`t * h * w`).
    """
    merged_counts = [(height // kernel_height) * (width // kernel_width) for _, height, width in grid_thw.tolist()]
    patch_counts = [temporal * height * width for temporal, height, width in grid_thw.tolist()]
    return merged_counts, patch_counts


def split_blocks(layers: nn.ModuleList, layer_types: list[str]) -> list[list[nn.Module]]:
    # Each tower chunks its layers by its own block-write layer positions; the stacks
    # only need to agree on the number of blocks, not on their internal sizes.
    write_indices = [i for i, t in enumerate(layer_types) if t == "block_write_layer"]
    return [list(layers[start:end]) for start, end in zip(write_indices, write_indices[1:] + [len(layers)])]


class ShensiVlModel(Kimi_K25Model):
    def __init__(self, config: ShensiVlConfig):
        super().__init__(config)
        vision_config, text_config = config.vision_config, config.text_config
        num_vision_attn_res_blocks = vision_config.attn_res_block_layer_types.count("block_write_layer")
        num_text_attn_res_blocks = text_config.attn_res_block_layer_types.count("block_write_layer")
        assert num_vision_attn_res_blocks == num_text_attn_res_blocks, (
            "DeepRecur requires the vision and language stacks to have the same number of attention-residual blocks."
        )

        self.num_attn_res_blocks = num_vision_attn_res_blocks

        vision_chunks = split_blocks(self.vision_tower.layers, vision_config.attn_res_block_layer_types)
        language_chunks = split_blocks(self.language_model.layers, text_config.attn_res_block_layer_types)
        self.blocks = nn.ModuleList(
            [
                ShensiVlCrossBlock(config, vision_chunks[i], language_chunks[i])
                for i in range(self.num_attn_res_blocks - 1)
            ]
            + [
                ShensiVlLoopBlock(
                    config,
                    vision_chunks[-1],
                    language_chunks[-1],
                    self.vision_tower.output_attn_res,
                    self.language_model.output_attn_res,
                    self.vision_tower.final_layernorm,
                )
            ]
        )

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        embeddings = super()._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        self.language_model._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        return embeddings

    def get_image_features(self) -> None:
        raise AttributeError()

    def get_video_features(self) -> None:
        raise AttributeError()

    def get_cross_mask(
        self,
        input_ids: torch.LongTensor,
        all_patch_counts: list[int],
        all_spans: list[list[tuple[int, int]]],
        sequence_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Builds the cross-modal attention masks over the merged patch set.

        Clip ids increment in packing order (all image clips of row 0..B-1, then all video
        clips), matching the patch order inside the packed vision sequence.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input token ids.
            all_patch_counts (`list[int]`):
                Per clip, the number of packed patches (KV rows).
            all_spans (`list[list[tuple[int, int]]]`):
                Per media modal (image, then video), the `(start, length)` placeholder spans
                found in each input row.
            sequence_length (`int`):
                The input sequence length.
            dtype (`torch.dtype`):
                Output mask dtype.
            device (`torch.device`):
                Output mask device.

        Returns:
            `tuple[torch.Tensor, torch.Tensor, torch.Tensor]`: `(evidence_mask, guidance_mask,
            patch_to_row)`, where `evidence_mask` is causal (`(batch_size, 1, seq_len, total_kv)`),
            `guidance_mask` matches each patch to its own clip (`(batch_size, 1, num_patches,
            seq_len)`), and `patch_to_row` is the batch row owning each packed patch.
        """
        number_of_clips = len(all_patch_counts)
        batch_size = input_ids.shape[0]
        media_spans = [[] for _ in range(batch_size)]  # per row: (start, length, global_clip_id)
        kv_clip_ids = torch.full((batch_size, sequence_length), -1, dtype=torch.long, device=device)
        clip_to_row = torch.full((number_of_clips,), -1, dtype=torch.long, device=device)
        clip_cursor = 0
        for span_group in all_spans:
            for row_index, row_spans in enumerate(span_group):
                for start, length in row_spans:
                    media_spans[row_index].append((start, length, clip_cursor))
                    kv_clip_ids[row_index, start : start + length] = clip_cursor
                    clip_to_row[clip_cursor] = row_index
                    clip_cursor += 1

        min_value = torch.finfo(dtype).min
        total_kv = sum(all_patch_counts)
        evidence_mask = torch.full((batch_size, sequence_length, total_kv), min_value, dtype=dtype, device=device)
        offsets = [0]
        for count in all_patch_counts:
            offsets.append(offsets[-1] + count)
        for row_index, segments in enumerate(media_spans):
            for start, _, clip_id in segments:
                evidence_mask[row_index, start:, offsets[clip_id] : offsets[clip_id] + all_patch_counts[clip_id]] = 0.0
        evidence_mask = evidence_mask.unsqueeze(1)
        full_row_visible = (evidence_mask != min_value).any(dim=-1, keepdim=True).type_as(evidence_mask)
        evidence_mask = evidence_mask * full_row_visible

        patch_clip_ids = torch.arange(number_of_clips, device=device).repeat_interleave(
            torch.tensor(all_patch_counts, device=device)
        )
        patch_to_row = clip_to_row[patch_clip_ids]
        same_clip = patch_clip_ids[None, :, None] == kv_clip_ids[:, None, :]
        guidance_mask = torch.where(same_clip, 0.0, min_value).to(device=device, dtype=dtype).unsqueeze(1)
        return evidence_mask, guidance_mask, patch_to_row

    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | ShensiVlModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config.text_config)

        if inputs_embeds is None:
            multimodal_mask = (input_ids == self.config.image_token_id) | (
                input_ids == getattr(self.config, "video_token_id", -1)
            )
            llm_input_ids = input_ids.clone()
            llm_input_ids[multimodal_mask] = 0
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)
        elif input_ids is None and (pixel_values is not None or pixel_values_videos is not None):
            input_ids = torch.full(inputs_embeds.shape[:2], 0, dtype=torch.long, device=inputs_embeds.device)
            image_embed = self.get_input_embeddings()(
                torch.full((), self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            input_ids[(inputs_embeds == image_embed).all(dim=-1)] = self.config.image_token_id
            video_token_id = getattr(self.config, "video_token_id", None)
            if video_token_id is not None:
                video_embed = self.get_input_embeddings()(
                    torch.full((), video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                input_ids[(inputs_embeds == video_embed).all(dim=-1)] = video_token_id

        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)

        vision_tower = self.vision_tower
        vision_config = self.config.vision_config
        kernel_height, kernel_width = vision_config.merge_kernel_size
        sequence_length = inputs_embeds.shape[1]
        embed_dtype = inputs_embeds.dtype
        device = inputs_embeds.device

        vision_hidden_states = vision_residual = vision_prefix_sum = None
        vision_cu_seqlens = vision_position_embeddings = vision_max_seqlen = None
        vision_evidence_mask = vision_guidance_mask = vision_patch_to_row = None
        all_patch_counts: list[int] = []
        all_spans: list[list[tuple[int, int]]] = []

        if pixel_values is not None and image_grid_thw is not None:
            vision_hidden_states = vision_tower.patch_embed(pixel_values, grid_thw=image_grid_thw)
            _, image_patch_counts = get_clip_counts(image_grid_thw, kernel_height, kernel_width)
            all_patch_counts.extend(image_patch_counts)
            if input_ids is not None:
                all_spans.append(find_spans(input_ids, self.config.image_token_id))

        if pixel_values_videos is not None and video_grid_thw is not None:
            video_hidden = vision_tower.patch_embed(pixel_values_videos, grid_thw=video_grid_thw)
            _, video_patch_counts = get_clip_counts(video_grid_thw, kernel_height, kernel_width)
            all_patch_counts.extend(video_patch_counts)
            video_token_id = getattr(self.config, "video_token_id", None)
            if video_token_id is not None and input_ids is not None:
                all_spans.append(find_spans(input_ids, video_token_id))
            vision_hidden_states = (
                video_hidden
                if vision_hidden_states is None
                else torch.cat([vision_hidden_states, video_hidden], dim=0)
            )

        if vision_hidden_states is not None:
            grids = [grid for grid in (image_grid_thw, video_grid_thw) if grid is not None]
            combined_grid = torch.cat(grids, dim=0)
            vision_position_ids = get_vision_position_ids(combined_grid, spatial_merge_size=1, kwargs=kwargs)
            vision_position_ids = vision_position_ids.transpose(0, 1).flip(0)
            vision_position_embeddings = vision_tower.rotary_emb(vision_hidden_states, vision_position_ids)
            vision_cu_seqlens, vision_max_seqlen = get_vision_attention_seqlens(
                combined_grid, vision_config, merge_temporal=True, kwargs=kwargs
            )
            vision_residual = vision_hidden_states.new_zeros(
                vision_hidden_states.size(0), self.num_attn_res_blocks, vision_hidden_states.size(-1)
            )
            vision_prefix_sum = vision_hidden_states
            vision_hidden_states = None

            for layer in self.blocks[0].vision_layers:
                vision_hidden_states, vision_prefix_sum, vision_residual = layer(
                    vision_hidden_states,
                    vision_residual,
                    vision_prefix_sum,
                    vision_cu_seqlens,
                    vision_position_embeddings,
                    max_seqlen=vision_max_seqlen,
                    **kwargs,
                )

            patch_offset = 0
            if pixel_values is not None and image_grid_thw is not None and input_ids is not None:
                image_patch_count = sum(image_patch_counts)
                pooled_images = vision_tower.temporal_patch_merger(
                    vision_hidden_states[:image_patch_count], image_grid_thw
                )
                projected_images = self.mm_projector(pooled_images).squeeze(1).to(dtype=embed_dtype)
                image_placeholder_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
                inputs_embeds = inputs_embeds.masked_scatter(image_placeholder_mask, projected_images)
                patch_offset = image_patch_count

            if pixel_values_videos is not None and video_grid_thw is not None and input_ids is not None:
                video_patch_count = sum(video_patch_counts)
                pooled_videos = vision_tower.temporal_patch_merger(
                    vision_hidden_states[patch_offset : patch_offset + video_patch_count], video_grid_thw
                )
                projected_videos = self.mm_projector(pooled_videos).squeeze(1).to(dtype=embed_dtype)
                video_token_id = getattr(self.config, "video_token_id", None)
                if video_token_id is not None:
                    video_placeholder_mask = (input_ids == video_token_id).unsqueeze(-1)
                    inputs_embeds = inputs_embeds.masked_scatter(video_placeholder_mask, projected_videos)

            if input_ids is not None:
                vision_evidence_mask, vision_guidance_mask, vision_patch_to_row = self.get_cross_mask(
                    input_ids, all_patch_counts, all_spans, sequence_length, embed_dtype, device
                )

        language_model = self.language_model
        num_streams = self.config.text_config.hc_mult
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, num_streams, -1).contiguous()
        position_embeddings = {
            "main": language_model.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
            "compress": language_model.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
        }
        block_residual = hidden_states.new_zeros(
            *hidden_states.shape[:2], num_streams, self.num_attn_res_blocks, hidden_states.size(-1)
        )
        prefix_sum = hidden_states
        hidden_states = None
        residual = block_residual

        if isinstance(attention_mask, dict):
            causal_mask = next(iter(attention_mask.values()))
        else:
            causal_mask = create_sliding_window_causal_mask(
                config=self.config.text_config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        output_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states)
        all_hidden_states = (inputs_embeds,) if output_hidden_states else ()
        for block_index, block in enumerate(self.blocks):
            block_inputs = {
                "hidden_states": hidden_states,
                "residual": residual,
                "prefix_sum": prefix_sum,
                "input_ids": input_ids,
                "position_embeddings": position_embeddings,
                "position_ids": position_ids,
                "attention_mask": causal_mask,
                "past_key_values": past_key_values,
                "vision_hidden_states": vision_hidden_states,
                "vision_prefix_sum": vision_prefix_sum,
                "vision_residual": vision_residual,
                "vision_cu_seqlens": vision_cu_seqlens,
                "vision_position_embeddings": vision_position_embeddings,
                "vision_max_seqlen": vision_max_seqlen,
                "evidence_mask": vision_evidence_mask,
                "num_attn_res_blocks": self.num_attn_res_blocks,
                "run_vision_layers": (block_index > 0),
            }
            if isinstance(block, ShensiVlCrossBlock):
                block_inputs.update(guidance_mask=vision_guidance_mask, vision_patch_to_row=vision_patch_to_row)
            (
                hidden_states,
                prefix_sum,
                residual,
                vision_hidden_states,
                vision_prefix_sum,
                vision_residual,
                layer_hidden_states,
            ) = block(
                **block_inputs,
                **{
                    **kwargs,
                    "output_hidden_states": output_hidden_states,
                    "output_attentions": kwargs.get("output_attentions", self.config.output_attentions),
                },
            )
            if output_hidden_states:
                all_hidden_states += tuple(
                    language_model.norm(language_model.hc_head(state)) for state in layer_hidden_states
                )

        hidden_states = language_model.norm(language_model.hc_head(hidden_states))
        return ShensiVlModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states or None,
        )


class ShensiVlCausalLMOutputWithPast(Kimi_K25CausalLMOutputWithPast):
    aux_loss: torch.FloatTensor | None = None


class ShensiVlForConditionalGeneration(Kimi_K25ForConditionalGeneration):
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
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_router_logits: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | ShensiVlCausalLMOutputWithPast:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
        )

        outputs: ShensiVlModelOutputWithPast = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            output_coupling_matrix=output_router_logits,
            output_sharing_layers=output_router_logits,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        aux_loss = None
        if output_router_logits and labels is not None:
            aux_loss, erc_loss = compute_loss_func(
                outputs.router_logits,
                self.config.text_config.n_routed_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
                getattr(outputs, "sharing_layers", None),
                getattr(outputs, "coupling_matrix", None),
                self.config.text_config.erc_loss_alpha,
            )
            if aux_loss != 0:
                loss = loss + self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)
            if erc_loss != 0:
                loss = loss + self.config.text_config.erc_loss_coef * erc_loss.to(loss.device)

        return ShensiVlCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        embeddings = super()._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        self.model._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        return embeddings


__all__ = [
    "ShensiVlConfig",
    "ShensiVlTextConfig",
    "ShensiVlVisionConfig",
    "ShensiVlForConditionalGeneration",
    "ShensiVlModel",
    "ShensiVlPreTrainedModel",
    "ShensiVlVisionModel",
]
