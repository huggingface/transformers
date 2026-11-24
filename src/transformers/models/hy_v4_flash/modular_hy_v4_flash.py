# Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch HY4-Flash model."""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec, remap_legacy_layer_types
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MoeModelOutputWithPast,
    MultimodalMoeBaseModelOutputWithPast,
)
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import maybe_autocast
from ..auto import AutoConfig, AutoModel
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention
from ..gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
from ..hy_v4.modeling_hy_v4 import (
    HYV4MLP,
    HYV4DecoderLayer,
    HYV4Experts,
    HYV4MoE,
    HYV4PreTrainedModel,
    HYV4RMSNorm,
)
from ..hy_v4_vl.configuration_hy_v4_vl import HYV4VLTextConfig
from ..hy_v4_vl.modeling_hy_v4_vl import (
    HYV4VLForConditionalGeneration,
    HYV4VLModel,
    HYV4VLTextModel,
    apply_rotary_pos_emb,
)
from ..hy_v4_vl.processing_hy_v4_vl import HYV4VLProcessor
from ..mimo_v2_flash.modeling_mimo_v2_flash import eager_attention_forward


@auto_docstring(checkpoint="tencent/Hy4-flash")
@strict
class HYV4FlashTextConfig(HYV4VLTextConfig):
    r"""
    sliding_window (`int`, *optional*, defaults to 512):
        Number of keys a sliding layer attends to, counting the query's own position.
    moe_hidden_size (`int`, *optional*, defaults to 1536):
        Latent width the routed experts run in.
    index_kpool (`int`, *optional*, defaults to 4):
        Number of consecutive tokens pooled into one compressed index entry.
    engram_layer_ids (`list[int]`, *optional*):
        One-based indices of the layers carrying an Engram memory block. `None` disables Engram.
    engram_max_ngram_size (`int`, *optional*, defaults to 3):
        Longest n-gram hashed into the Engram memory. Orders `2..engram_max_ngram_size` are used.
    engram_n_head_per_ngram (`int`, *optional*, defaults to 8):
        Number of hash heads per n-gram order.
    engram_n_embed_per_ngram (`int`, *optional*, defaults to 768):
        Total embedding width per n-gram order, split evenly across its hash heads.
    engram_layer_head_vocab_sizes (`dict[str, list[list[int]]]`, *optional*):
        Hash-head vocabulary sizes of each Engram layer, keyed by its one-based index: one list per n-gram
        order from 2 to `engram_max_ngram_size`, each holding `engram_n_head_per_ngram` sizes. Every size is
        a distinct prime so two heads never share a modulus.
    engram_pad_id (`int`, *optional*, defaults to 120002):
        Token id used to pad n-grams that reach past the start of a sequence.
    engram_fp8_embedding (`bool`, *optional*, defaults to `True`):
        Whether Engram embeddings undergo per-vector dynamic fp8 (e4m3) round-trip quantization on lookup,
        matching training. Published checkpoints store the tables in bf16 without quantization scales.
    engram_seed (`int`, *optional*, defaults to 0):
        Seed the hash multipliers of every Engram layer are drawn from. The trained memory is only readable
        with the multipliers it was built with, so this must match the checkpoint.
    engram_lookup_table_size (`int`, *optional*, defaults to 120818):
        Number of token ids the frozen normalized-vocabulary map covers: the tokenizer's vocabulary, which
        `vocab_size` pads past. Ids beyond it read as padding.
    n_group (`int`, *optional*, defaults to 1):
        Number of expert groups for routing. Experts are selected globally, so this is 1.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer MLP type, either `"dense"` or `"sparse"`. Defaults to one dense layer followed by
        sparse MoE layers.
    index_topk (`int`, *optional*, defaults to 2048):
        Maximum number of key positions selected by each indexed layer's compressed indexer.
    index_head_dim (`int`, *optional*, defaults to 128):
        Width of each compressed indexer head.
    index_n_heads (`int`, *optional*, defaults to 16):
        Number of scoring heads of the compressed indexer.
    hc_mult (`int`, *optional*, defaults to 4):
        Number of hidden-state channels maintained by iHC.
    hc_magnitude (`float`, *optional*, defaults to 2.0):
        Scale applied to the iHC post-gating branch.
    hc_eps (`float`, *optional*, defaults to 1e-6):
        Numerical epsilon added to iHC sigmoid gates.
    learnable_sink_init (`float`, *optional*, defaults to 0.0):
        Initial value of each learned attention-sink logit.
    swiglu_limit (`float`, *optional*, defaults to 10.0):
        Magnitude of the routed-expert SwiGLU clamp. Values at or below zero disable the clamp.
    """

    model_type = "hy_v4_flash_text"
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}
    default_theta = {"indexed_attention": 10000000.0, "sliding_attention": 10000.0}
    # Each Engram table is ~32 GB in bf16, more than a replicated copy per rank can afford, so the tables
    # are split by rows like the token embeddings.
    base_model_tp_plan = {
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.gate_proj": "colwise",
        "layers.*.self_attn.sinks": "colwise",
        "layers.*.self_attn": "local_params",
        "layers.*.engram_layer.multi_head_embedding.embedding": "embedding_rowwise",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    vocab_size: int = 120832
    hidden_size: int = 3072
    intermediate_size: int = 13824
    moe_intermediate_size: int = 1536
    num_hidden_layers: int = 42
    num_attention_heads: int = 32
    max_position_embeddings: int = 1048576
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    n_routed_experts: int = 512
    num_experts_per_tok: int = 16
    n_shared_experts: int = 1
    routed_scaling_factor: float = 3.998
    moe_hidden_size: int = 1536
    sliding_window: int = 512
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 16
    index_kpool: int = 4
    engram_layer_ids: list[int] | None = None
    engram_max_ngram_size: int = 3
    engram_n_head_per_ngram: int = 8
    engram_n_embed_per_ngram: int = 768
    engram_layer_head_vocab_sizes: dict | None = None
    engram_pad_id: int = 120002
    engram_fp8_embedding: bool = True
    engram_seed: int = 0
    engram_lookup_table_size: int = 120818

    # Every indexed layer runs its own compressed indexer, so HYV4's full/shared split is unused.
    indexer_types = AttributeError()

    def __post_init__(self, **kwargs):
        # BC: published checkpoints name the latent expert width `routed_expert_hidden_size`.
        if "routed_expert_hidden_size" in kwargs:
            self.moe_hidden_size = kwargs.pop("routed_expert_hidden_size")

        # Accept the published checkpoint's compressed-indexer aliases alongside the canonical `index_*` names.
        if "compressed_indexer_compress_ratio" in kwargs:
            self.index_kpool = kwargs.pop("compressed_indexer_compress_ratio")
        kwargs.pop("compressed_indexer_num_attention_heads", None)
        kwargs.pop("compressed_indexer_head_dim", None)
        kwargs.pop("compressed_indexer_group_topk", None)
        kwargs.pop("compressed_indexer_candidate_topk", None)

        # MLA expands the latent to one key/value per query head, so keys are never grouped.
        self.num_key_value_heads = self.num_attention_heads
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # RoPE applies only to the rope slice, so `head_dim` points at it.
        self.head_dim = self.qk_rope_head_dim

        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(1, self.num_hidden_layers) + ["sparse"] * max(
                self.num_hidden_layers - 1, 0
            )
        if self.layer_types is None:
            # Three sliding layers for every indexed one, starting with two indexed layers.
            self.layer_types = [
                "indexed_attention" if layer_idx < 2 or (layer_idx - 1) % 4 == 0 else "sliding_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = remap_legacy_layer_types(list(self.layer_types))

        # Normalize the published checkpoint's SWA aliases into the shared per-layer configuration.
        sliding_config = {
            "num_attention_heads": kwargs.pop("swa_num_attention_heads", 64),
            "qk_nope_head_dim": kwargs.pop("swa_qk_nope_head_dim", self.qk_nope_head_dim),
            "qk_rope_head_dim": kwargs.pop("swa_qk_rope_head_dim", self.qk_rope_head_dim),
            "v_head_dim": kwargs.pop("swa_v_head_dim", self.v_head_dim),
        }
        sliding_config["num_key_value_heads"] = sliding_config["num_attention_heads"]
        sliding_config["qk_head_dim"] = sliding_config["qk_nope_head_dim"] + sliding_config["qk_rope_head_dim"]
        sliding_config["head_dim"] = sliding_config["qk_rope_head_dim"]
        kwargs.pop("swa_qk_head_dim", None)
        kwargs.setdefault(
            "per_layer_config",
            {i: sliding_config for i, layer_type in enumerate(self.layer_types) if layer_type == "sliding_attention"},
        )

        # Skips `HYV4Config.__post_init__`, which derives `indexer_types` for HYV4's full/shared indexer split.
        PreTrainedConfig.__post_init__(self, **kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        # Two things `PreTrainedConfig.convert_rope_params_to_dict` cannot do on its own:
        # 1. `remap_legacy_layer_types` remaps `layer_types`, not the keys inside `rope_parameters`, so the
        #    published checkpoint's `"deepseek_sparse_attention"` entry would otherwise fall back to defaults.
        # 2. Each layer type has its own default theta (`1e7` vs `1e4`) and default `mrope_section`.
        rope_scaling = kwargs.pop("rope_scaling", None)
        default_rope_params = {
            "indexed_attention": {"rope_type": "default"},
            "sliding_attention": {"rope_type": "default"},
        }
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else default_rope_params
        keys = remap_legacy_layer_types(list(self.rope_parameters))
        self.rope_parameters = dict(zip(keys, self.rope_parameters.values()))
        if rope_scaling is not None:
            self.rope_parameters["indexed_attention"].update(rope_scaling)

        rope_theta = kwargs.pop("rope_theta", None)
        for layer_type in ("indexed_attention", "sliding_attention"):
            if self.rope_parameters.get(layer_type) is None:
                self.rope_parameters[layer_type] = {"rope_type": "default"}
            self.rope_parameters[layer_type].setdefault(
                "rope_theta", rope_theta if rope_theta is not None else self.default_theta[layer_type]
            )
            self.rope_parameters[layer_type].setdefault("mrope_section", [12, 10, 10])

        self.standardize_rope_params()
        return kwargs

    def validate_architecture(self):
        PreTrainedConfig.validate_architecture(self)

        if self.index_kpool < 1:
            raise ValueError(f"`index_kpool` must be positive, got {self.index_kpool}.")
        if self.index_topk % self.index_kpool != 0:
            raise ValueError(
                f"`index_topk` ({self.index_topk}) must be divisible by `index_kpool` ({self.index_kpool})."
            )

        for layer_type in set(self.layer_types):
            rope_params = self.rope_parameters[layer_type]
            rope_head_dim = self.per_layer_config[layer_type].qk_rope_head_dim
            mrope_section = rope_params["mrope_section"]
            if len(mrope_section) != 3 or sum(mrope_section) != rope_head_dim // 2:
                raise ValueError(
                    "`mrope_section` must have three entries (time, height, width) summing to half the rotary "
                    f"width, got {mrope_section} for `{layer_type}` with a rotary width of {rope_head_dim}."
                )

        # The sizes are the hash moduli, read by n-gram order and head; a table of the wrong shape would not
        # fail later, it would hash against the wrong moduli.
        if self.engram_layer_ids and not self.engram_pad_id < self.engram_lookup_table_size:
            raise ValueError(
                f"`engram_lookup_table_size` ({self.engram_lookup_table_size}) must cover `engram_pad_id` "
                f"({self.engram_pad_id}): ids past the map read as padding."
            )
        num_orders = self.engram_max_ngram_size - 1
        for layer_id in self.engram_layer_ids or []:
            sizes = (self.engram_layer_head_vocab_sizes or {}).get(str(layer_id))
            if (
                not isinstance(sizes, list)
                or len(sizes) != num_orders
                or any(not isinstance(order, list) or len(order) != self.engram_n_head_per_ngram for order in sizes)
            ):
                raise ValueError(
                    f"`engram_layer_head_vocab_sizes['{layer_id}']` must hold {num_orders} lists (n-gram orders 2 "
                    f"to {self.engram_max_ngram_size}) of {self.engram_n_head_per_ngram} sizes each, got {sizes}."
                )


@auto_docstring(checkpoint="tencent/Hy4-flash")
@strict
class HYV4FlashConfig(PreTrainedConfig):
    model_type = "hy_v4_flash"
    sub_configs_defaults = {
        "vision_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="hy_v4_vl_vision",
            init_kwargs={
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "out_hidden_size": 3072,
            },
        ),
        "text_config": HYV4FlashTextConfig,
    }

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 120120
    video_token_id: int = 120124
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        # BC: Published checkpoints label `vision_config.model_type` as `"hy_v4_flash"`, which would recurse
        # under `AutoConfig` resolution; redirect it to `"hy_v4_vl_vision"` whose tower this model reuses.
        if isinstance(self.vision_config, dict) and self.vision_config.get("model_type") == self.model_type:
            self.vision_config = {**self.vision_config, "model_type": "hy_v4_vl_vision"}
        super().__post_init__(**kwargs)


class HYV4FlashRMSNorm(HYV4RMSNorm):
    pass


class HYV4FlashRotaryEmbedding(Gemma4TextRotaryEmbedding):
    """Per-layer-type RoPE inherited from Gemma4, extended with Qwen3-VL's interleaved 3D M-RoPE."""

    def __init__(self, config: HYV4FlashTextConfig, device=None):
        super().__init__(config, device=device)
        # Key difference: store per-layer-type `mrope_section` for 3D time/height/width interleaving.
        self.mrope_section = {
            layer_type: config.rope_parameters[layer_type]["mrope_section"] for layer_type in self.layer_types
        }

    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids, layer_type: str):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        # Key difference: compute frequencies for 3D (time, height, width) positions and interleave the three planes.
        freqs = position_ids[..., None].float() * inv_freq.to(device=x.device, dtype=torch.float)
        cos = freqs.cos() * attention_scaling
        sin = freqs.sin() * attention_scaling

        cos = self.recomposition_frequencies(cos, layer_type)
        sin = self.recomposition_frequencies(sin, layer_type)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def recomposition_frequencies(self, freq, layer_type: str):
        """Interleave the time, height and width planes into the layout the rotary slice expects."""
        mrope_section = self.mrope_section[layer_type]
        freqs_thw = freq[0]  # the time plane is overwritten in place with the height and width ones
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_thw[..., idx] = freq[dim][..., idx]
        return torch.cat((freqs_thw, freqs_thw), dim=-1)


class HYV4FlashMLP(HYV4MLP):
    pass


class HYV4FlashExperts(HYV4Experts):
    """Routed experts, sized on the MoE latent width rather than on `hidden_size`."""

    def __init__(self, config: HYV4FlashTextConfig):
        super().__init__(config)
        self.hidden_dim = config.moe_hidden_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))


class HYV4FlashMoE(HYV4MoE):
    """MoE block whose routed experts run in a latent space (`ℓ-MoE`).

    `pre_expert_proj` maps the hidden states into the latent width before the experts and
    `post_expert_proj` maps their combined output back. The router and the shared expert keep
    seeing the full-width input, and the sandwich norm closes the block.
    """

    def __init__(self, config: HYV4FlashTextConfig):
        super().__init__(config)
        self.experts = HYV4FlashExperts(config)
        self.shared_experts = HYV4FlashMLP(
            config=config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )
        self.pre_expert_proj = nn.Linear(config.hidden_size, config.moe_hidden_size, bias=False)
        self.post_expert_proj = nn.Linear(config.moe_hidden_size, config.hidden_size, bias=False)
        self.combine_output_norm = HYV4FlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        _, topk_weights, topk_indices = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # Key difference: project routed tokens into and out of the latent expert space (`moe_hidden_size`).
        routed_in = self.pre_expert_proj(hidden_states)
        routed_out = self.experts(routed_in, topk_indices, topk_weights)
        routed_out = self.post_expert_proj(routed_out)

        hidden_states = routed_out + self.shared_experts(hidden_states)
        return self.combine_output_norm(hidden_states).view(*orig_shape)


class HYV4FlashCompressedIndexer(nn.Module):
    """Compressed Lightning Indexer: pick which `index_kpool`-token groups a query may attend to.

    Every `index_kpool` consecutive tokens are pooled into one compressed key, so scoring a query
    against the whole context costs a quarter of a per-token indexer. Unlike `Glm5NextTextIndexer`,
    the pooling transforms cached normalized token keys through separate fp32 value and gate
    projections (`key_value_projection`, `key_gate_projection`) biased by `intra_group_ape` followed
    by `key_norm`, and both queries and pooled group keys receive interleaved 3D M-RoPE on their
    trailing `qk_rope_head_dim` channels (at each query's position and each group's first token
    position respectively).

    A query keeps the `index_topk // index_kpool - 1` best-scoring complete groups plus the group
    it sits in, and attention then runs over the tokens those groups expand to.
    """

    def __init__(self, config: HYV4FlashTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.index_kpool = config.index_kpool
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.group_topk = config.index_topk // config.index_kpool
        self.qk_rope_head_dim = config.qk_rope_head_dim

        self.query_projection = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.head_weight_projection = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.token_key_projection = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.token_key_norm = HYV4FlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # The pooling runs in fp32: it is a softmax over only `index_kpool` slots, so a bf16
        # interior loses the ordering of near-ties and flips which groups a query recalls.
        self.key_value_projection = nn.Linear(self.head_dim, self.head_dim, bias=False, dtype=torch.float32)
        self.key_gate_projection = nn.Linear(self.head_dim, self.head_dim, bias=False, dtype=torch.float32)
        self.key_norm = HYV4FlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.intra_group_ape = nn.Parameter(torch.empty(self.index_kpool, self.head_dim, dtype=torch.float32))
        # The indexer scores with the same rotary phases as the indexed layers, but at the group starts
        # rather than at every token, so it keeps its own table instead of reusing the backbone's.
        self.rotary_emb = HYV4FlashRotaryEmbedding(config)

    def pool_groups(self, token_keys: torch.Tensor) -> torch.Tensor:
        """Pool `[batch, tokens, head_dim]` token keys into `[batch, groups, head_dim]` group keys."""
        batch_size, num_tokens, _ = token_keys.shape
        num_groups = num_tokens // self.index_kpool
        grouped = token_keys[:, : num_groups * self.index_kpool].reshape(
            batch_size, num_groups, self.index_kpool, self.head_dim
        )
        with maybe_autocast(device_type=token_keys.device.type, enabled=False):
            # The pooling weights are declared fp32 and kept there by `_keep_in_fp32_modules_strict`;
            # meet them in their own precision rather than casting them down to the activations'.
            grouped = grouped.to(self.key_value_projection.weight.dtype)
            values = self.key_value_projection(grouped)
            gates = self.key_gate_projection(grouped) + self.intra_group_ape
            pooled = (values * gates.softmax(dim=2)).sum(dim=2)
        return self.key_norm(pooled).to(token_keys.dtype)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        indexer_position_ids: torch.Tensor,
        cache: Cache | None,
        kv_length: int,
    ) -> torch.Tensor:
        """Overlay the compressed indexer's group selection onto the causal attention mask.

        The token keys are cached rather than the pooled groups: a decoding step only carries its own
        token, and the group it belongs to is only complete once `index_kpool` tokens have arrived.
        The indexer only pays off once the context outgrows `index_topk`, so below it the layer
        attends densely and the pooling is skipped entirely.
        """
        ratio = self.index_kpool
        batch_size, seq_length, _ = hidden_states.shape

        token_keys = self.token_key_norm(self.token_key_projection(hidden_states))
        cached_length = kv_length
        if cache is not None:
            token_keys = cache.update_indexer(token_keys, self.layer_idx)
            # Under a static cache `kv_length` is the preallocated buffer size while `get_seq_length`
            # tracks the tokens written so far.
            cached_length = cache.get_seq_length(self.layer_idx)
        if kv_length <= self.index_topk:
            return attention_mask

        group_keys = self.pool_groups(token_keys)
        num_groups = group_keys.shape[1]
        # Groups follow token slots rather than positions, which the vision spans make non-contiguous.
        query_slots = torch.arange(seq_length, device=hidden_states.device) + (cached_length - seq_length)
        query_group_index = (query_slots // ratio).expand(batch_size, -1)
        # A group carries the rotary phase of its first token; positions are `[3, batch, tokens]`.
        group_position_ids = indexer_position_ids[..., : num_groups * ratio : ratio]

        # Flipped RoPE position (trailing `qk_rope_head_dim` channels rotated, as in HYV4Indexer).
        # Queries and group keys live on different position grids so each is rotated with its own cos/sin.
        nope_dim = self.head_dim - self.qk_rope_head_dim
        query = self.query_projection(q_resid).view(batch_size, seq_length, self.n_heads, self.head_dim)
        query_cos, query_sin = self.rotary_emb(query, position_ids, layer_type="indexed_attention")
        q_pass, q_rot = torch.split(query, [nope_dim, self.qk_rope_head_dim], dim=-1)
        q_rot, _ = apply_rotary_pos_emb(q_rot, q_rot, query_cos, query_sin, unsqueeze_dim=2)
        query = torch.cat((q_pass, q_rot), dim=-1)

        group_cos, group_sin = self.rotary_emb(group_keys, group_position_ids, layer_type="indexed_attention")
        k_pass, k_rot = torch.split(group_keys.unsqueeze(2), [nope_dim, self.qk_rope_head_dim], dim=-1)
        k_rot, _ = apply_rotary_pos_emb(k_rot, k_rot, group_cos, group_sin, unsqueeze_dim=2)
        keys = torch.cat((k_pass, k_rot), dim=-1).squeeze(2)

        head_weights = self.head_weight_projection(hidden_states).float() * (self.n_heads**-0.5 * self.head_dim**-0.5)

        # [batch, queries, heads, group] scores, then one weighted sum over the heads
        scores = F.relu(torch.matmul(query.float(), keys.float().unsqueeze(1).transpose(-1, -2)))
        logits = torch.matmul(head_weights.unsqueeze(-2), scores).squeeze(-2)

        # A query only sees groups that closed before it; its own group is recalled unconditionally.
        group_ids = torch.arange(num_groups, device=logits.device)
        logits = logits.masked_fill(group_ids >= query_group_index.unsqueeze(-1), float("-inf"))

        complete_topk = min(self.group_topk - 1, num_groups)
        recalled_groups = torch.cat(
            (logits.topk(complete_topk, dim=-1).indices, query_group_index.unsqueeze(-1)), dim=-1
        )

        # Expand the recalled groups back into the token slots they stand for.
        offsets = torch.arange(ratio, device=recalled_groups.device)
        recalled = (recalled_groups.unsqueeze(-1) * ratio + offsets).flatten(-2).clamp(max=kv_length - 1)
        unrecalled = attention_mask.new_ones((batch_size, seq_length, kv_length), dtype=torch.bool)
        unrecalled.scatter_(-1, recalled, False)
        return attention_mask.masked_fill(unrecalled.unsqueeze(1), torch.finfo(attention_mask.dtype).min)


class HYV4FlashAttention(DeepseekV3Attention):
    """Gated MLA whose per-layer head dimensions come from `config.per_layer_config[layer_idx]`.

    `sliding_attention` layers attend to the last `config.sliding_window` keys with a wider head
    count; `indexed_attention` layers attend to the tokens `compressed_indexer` recalled. Both carry
    learned attention sinks.
    """

    def __init__(self, config: HYV4FlashTextConfig, layer_idx: int):
        model_config = config
        config = config.per_layer_config[layer_idx]
        super().__init__(config, layer_idx)
        self.config = model_config
        self.attention_type = config.layer_types[layer_idx]
        self.q_a_layernorm = HYV4FlashRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.kv_a_layernorm = HYV4FlashRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.scaling = self.qk_head_dim ** (-0.5)
        self.gate_projection_size = self.v_head_dim
        self.gate_proj = nn.Linear(config.hidden_size, self.num_heads * self.gate_projection_size, bias=False)
        self.sinks = nn.Parameter(torch.full((self.num_heads,), config.learnable_sink_init, dtype=torch.float32))
        self.compressed_indexer = (
            HYV4FlashCompressedIndexer(model_config, layer_idx) if self.attention_type == "indexed_attention" else None
        )

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        cache: Cache | None = None,
        indexer_position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        gate_states = self.gate_proj(hidden_states).view(batch_size, seq_length, -1, self.gate_projection_size)

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, self.kv_lora_rank)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        if cache is not None:
            k_pass, k_rot = cache.update(k_pass, k_rot, self.layer_idx, kwargs=kwargs)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states, value_states = self.expand_kv(k_pass, k_rot)

        if self.compressed_indexer is not None:
            position_ids = kwargs["position_ids"]
            attention_mask = self.compressed_indexer(
                hidden_states,
                q_resid,
                attention_mask,
                position_ids,
                indexer_position_ids if indexer_position_ids is not None else position_ids,
                cache,
                key_states.shape[2],
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
            s_aux=self.sinks,
            **kwargs,
        )

        attn_output = attn_output * torch.sigmoid(gate_states)
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class HYV4FlashEngramEmbedding(nn.Module):
    """Hash-head embedding table with a distinct prime vocabulary per head.

    The heads share one flat `nn.Embedding` and are told apart by an offset, which keeps a single
    contiguous table instead of a `ModuleList` of differently sized ones. A negative hash id marks a
    position that owns no n-gram and reads back as zeros.
    """

    def __init__(self, head_vocab_sizes: list[int], embedding_dim: int):
        super().__init__()
        self.head_vocab_sizes = head_vocab_sizes
        self.offsets = nn.Buffer(self.head_offsets(), persistent=False)
        self.embedding = nn.Embedding(sum(head_vocab_sizes), embedding_dim)

    def head_offsets(self) -> torch.Tensor:
        """Row at which each head's slice of the shared table starts."""
        return torch.tensor([0] + self.head_vocab_sizes[:-1]).cumsum(0)

    def forward(self, hash_ids: torch.Tensor) -> torch.Tensor:
        # Explicit device placement mirrors `Qwen4ExpTextNGramEmbedding`: `_no_placement_params` may leave
        # the ~32 GB table on CPU when GPUs cannot fit it, so only the looked-up rows cross to the device.
        execution_device = self.embedding.weight.device if self.embedding.weight.device.type != "meta" else None
        indices = (hash_ids.clamp_min(0) + self.offsets).to(execution_device)
        embeddings = self.embedding(indices).to(hash_ids.device)
        return embeddings.masked_fill(hash_ids.unsqueeze(-1) < 0, 0.0)


class HYV4FlashEngramBranchScales(nn.Module):
    """Per-hyper-connection branch scales; kept as a callable module so accelerate offload hooks materialize them."""

    def __init__(self, hc_mult: int, hidden_size: int):
        super().__init__()
        for i in range(hc_mult):
            self.register_parameter(str(i), nn.Parameter(torch.empty(hidden_size)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.stack([scale * hidden_states for scale in self.parameters()], dim=2)


class HYV4FlashEngramLayer(nn.Module):
    """Engram memory: a hashed n-gram lookup added to the hidden streams before attention.

    Each n-gram order from 2 to `engram_max_ngram_size` is mixed into `engram_n_head_per_ngram` hash
    ids, which index this layer's embedding table. The looked-up vectors are normalized per head,
    concatenated and projected once into a shared value, then gated per hyper-connection channel by
    how well that channel's query matches the memory's key.
    """

    def __init__(self, config: HYV4FlashTextConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_head_per_ngram = config.engram_n_head_per_ngram
        self.fp8_embedding = config.engram_fp8_embedding
        self.pad_id = config.engram_pad_id
        self.seed = config.engram_seed

        # One table for all heads, laid out n-gram-major: the order `hash_ngrams` produces the hash ids in.
        head_vocab_sizes = [size for order in config.engram_layer_head_vocab_sizes[str(layer_id)] for size in order]
        self.config_head_vocab_sizes = head_vocab_sizes
        self.num_hash_heads = len(head_vocab_sizes)
        head_dim = config.engram_n_embed_per_ngram // self.n_head_per_ngram
        engram_size = (self.max_ngram_size - 1) * config.engram_n_embed_per_ngram

        self.head_vocab_sizes = nn.Buffer(torch.tensor(head_vocab_sizes), persistent=False)
        # Engram hashes a normalized vocabulary in which tokens that only differ by case, accent or
        # whitespace share an id. The map is frozen with the checkpoint rather than rebuilt from the
        # tokenizer, so the hashes stay the ones the memory was trained on.
        self.lookup_table = nn.Buffer(torch.arange(config.engram_lookup_table_size, dtype=torch.int32))
        # The multipliers define the hash, so the trained memory is only readable with the values it was built
        # with. The checkpoint keeps the seed they were drawn from; the draw is bounded by the size of the
        # compressed vocabulary, which for the identity map above is its length. `_init_weights` draws them
        # again from the map the checkpoint loads.
        self.hash_multipliers = nn.Buffer(
            self.draw_hash_multipliers(config.engram_lookup_table_size), persistent=False
        )

        self.multi_head_embedding = HYV4FlashEngramEmbedding(head_vocab_sizes, head_dim)
        self.lookup_per_head_norms = nn.ModuleList([nn.RMSNorm(head_dim) for _ in range(self.num_hash_heads)])
        self.value_proj = nn.Linear(engram_size, self.hidden_size)
        self.key_projs = nn.ModuleList([nn.Linear(engram_size, self.hidden_size) for _ in range(self.hc_mult)])
        self.norm_key = nn.ModuleList([nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)])
        self.norm_query = nn.ModuleList([nn.RMSNorm(self.hidden_size) for _ in range(self.hc_mult)])
        self.unified_rmsnorm = nn.RMSNorm(self.hidden_size, elementwise_affine=False)
        self.unified_pre_gamma = nn.Parameter(torch.empty(self.hidden_size))
        self.unified_per_hc_scales = HYV4FlashEngramBranchScales(self.hc_mult, self.hidden_size)

    def draw_hash_multipliers(self, compressed_vocab_size: int) -> torch.Tensor:
        """Draw the odd multipliers an n-gram is mixed with, as the memory was built: NumPy's default generator
        seeded per layer, bounded so that a product with any compressed id stays within int64."""
        half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab_size // 2)
        generator = np.random.default_rng(self.seed + 10007 * self.layer_id)
        drawn = generator.integers(0, half_bound, size=self.max_ngram_size, dtype=np.int64)
        return torch.tensor((drawn * 2 + 1).tolist(), dtype=torch.int64)

    def hash_ngrams(self, ngram_tokens: torch.Tensor) -> torch.Tensor:
        """Mix `[batch, tokens, max_ngram_size]` shifted token ids into one hash id per head."""
        # The map covers the tokenizer's ids; the embedding pads past them and resizing adds more. Those ids
        # have no normalized form and read as padding.
        table_size = self.lookup_table.shape[0]
        compressed = self.lookup_table[ngram_tokens.clamp_max(table_size - 1)]
        compressed = torch.where(ngram_tokens < table_size, compressed, self.lookup_table[self.pad_id])
        mixed = compressed.long() * self.hash_multipliers
        hash_ids = []
        for ngram_size in range(2, self.max_ngram_size + 1):
            mix = mixed[..., 0]
            for offset in range(1, ngram_size):
                mix = torch.bitwise_xor(mix, mixed[..., offset])
            order_start = (ngram_size - 2) * self.n_head_per_ngram
            moduli = self.head_vocab_sizes[order_start : order_start + self.n_head_per_ngram]
            hash_ids.append(mix.unsqueeze(-1) % moduli)
        return torch.cat(hash_ids, dim=-1)

    def forward(
        self, hidden_states: torch.Tensor, ngram_tokens: torch.Tensor, engram_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return the residual to add to `[batch, tokens, hc_mult, hidden_size]` hidden streams."""
        hash_ids = self.hash_ngrams(ngram_tokens)
        embeddings = self.multi_head_embedding(hash_ids).to(hidden_states.dtype)
        if self.fp8_embedding:
            embeddings = _fp8_round_trip(embeddings)
        embeddings = torch.stack(
            [norm(embeddings[:, :, head]) for head, norm in enumerate(self.lookup_per_head_norms)], dim=2
        )
        embeddings = embeddings.flatten(start_dim=-2)

        gates = []
        for hc_index in range(self.hc_mult):
            key = self.norm_key[hc_index](self.key_projs[hc_index](embeddings))
            query = self.norm_query[hc_index](hidden_states[:, :, hc_index])
            gate = (key * query).sum(dim=-1) * self.hidden_size**-0.5
            # A square root on the magnitude keeps the gate responsive once the dot product grows.
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gates.append(gate.sigmoid())
        gates = torch.stack(gates, dim=2).unsqueeze(-1)

        main_signal = self.unified_pre_gamma * self.unified_rmsnorm(self.value_proj(embeddings))
        branch = self.unified_per_hc_scales(main_signal)
        engram_output = gates * main_signal.unsqueeze(2) + branch
        return engram_output.masked_fill(~engram_mask[:, :, None, None], 0.0)


def _fp8_round_trip(x: torch.Tensor) -> torch.Tensor:
    """Quantize to fp8 (e4m3) per vector and back, as Engram embeddings were trained to be read.

    Published checkpoints store the Engram table in bf16 with no `quantization_config` or per-tensor
    `weight_scale`, so `FineGrainedFP8Config` (`modules_to_convert` / `FP8Embedding`) does not apply.
    """
    info = torch.finfo(torch.float8_e4m3fn)
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / info.max
    quantized = (x / scale).clamp(-info.max, info.max).to(torch.float8_e4m3fn).to(x.dtype) * scale
    return x + (quantized - x).detach()


class HYV4FlashDecoderLayer(HYV4DecoderLayer):
    def __init__(self, config: HYV4FlashTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention_type = config.layer_types[layer_idx]
        self.post_self_attn_layernorm = HYV4FlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = HYV4FlashMoE(config) if config.mlp_layer_types[layer_idx] == "sparse" else HYV4FlashMLP(config)
        # Dense layers apply their sandwich norm right after the MLP; MoE layers carry `combine_output_norm`
        # inside the block so `HYV4FlashMLP` stays plain for both dense layers and `shared_experts`.
        self.post_mlp_layernorm = (
            nn.Identity()
            if config.mlp_layer_types[layer_idx] == "sparse"
            else HYV4FlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        )
        self.engram_layer = None
        if config.engram_layer_ids is not None and layer_idx + 1 in config.engram_layer_ids:
            self.engram_layer = HYV4FlashEngramLayer(config, layer_idx + 1)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        indexer_position_ids: torch.Tensor | None = None,
        ngram_tokens: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        dtype = hidden_states.dtype

        # No token ids means no n-gram to look up, the same situation as a visual span, so the memory
        # contributes nothing rather than guessing at hashes.
        if self.engram_layer is not None and ngram_tokens is not None:
            hidden_states = hidden_states + self.engram_layer(hidden_states, ngram_tokens, engram_mask)

        residual = hidden_states
        post, hidden_states = self.attn_hc(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            indexer_position_ids=indexer_position_ids,
            **kwargs,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = (post.float().unsqueeze(-1) * hidden_states.float().unsqueeze(-2) + residual.float()).to(dtype)

        residual = hidden_states
        post, hidden_states = self.ffn_hc(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.post_mlp_layernorm(self.mlp(hidden_states))
        hidden_states = (post.float().unsqueeze(-1) * hidden_states.float().unsqueeze(-2) + residual.float()).to(dtype)

        return hidden_states


@auto_docstring
class HYV4FlashPreTrainedModel(HYV4PreTrainedModel):
    config: HYV4FlashConfig
    input_modalities = ("image", "video", "text")
    _no_split_modules = ["HYV4FlashDecoderLayer"]
    # Each Engram table is ~32 GB in bf16; skip placing it on accelerators under `device_map="auto"` when
    # a GPU cannot fit the layer, mirroring `Qwen4ExpPreTrainedModel`.
    _no_placement_params = ["engram_layer.multi_head_embedding.embedding.weight"]
    # The released checkpoints also carry the speculative-decoding draft layers (`mtp_layers` or `dfly_layers`),
    # which only serving engines use.
    _keys_to_ignore_on_load_unexpected = [r"model\.language_model\.(mtp|dfly)_layers\..*"]
    # Spelled out rather than inherited: the entries are matched as substrings, and HY4's bare `scale`
    # also catches this model's `unified_per_hc_scales`.
    _keep_in_fp32_modules_strict = [
        "e_score_correction_bias",
        "gate.weight",
        "attn_hc.fn",
        "attn_hc.scale",
        "attn_hc.base",
        "ffn_hc.fn",
        "ffn_hc.scale",
        "ffn_hc.base",
        "hc_head.hc_fn",
        "hc_head.hc_scale",
        "hc_head.hc_base",
        "sinks",
        "key_value_projection",
        "key_gate_projection",
        "intra_group_ape",
    ]

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, HYV4FlashCompressedIndexer):
            init.normal_(module.intra_group_ape, mean=0.0, std=self.config.get_text_config().initializer_range)
        elif isinstance(module, HYV4FlashEngramLayer):
            init.ones_(module.unified_pre_gamma)
            for scale in module.unified_per_hc_scales.parameters():
                init.normal_(scale, mean=0.0, std=0.006)
            init.copy_(module.head_vocab_sizes, torch.tensor(module.config_head_vocab_sizes))
            # Without a trained map every token keeps its own id, which is the identity compression. A loaded
            # map is left as is.
            init.copy_(module.lookup_table, torch.arange(module.lookup_table.shape[0], dtype=torch.int32))
            # Loading fills the map before this runs, so the draw sees the map the memory was trained with.
            if not module.lookup_table.is_meta:
                compressed_vocab_size = int(module.lookup_table.max()) + 1
                init.copy_(module.hash_multipliers, module.draw_hash_multipliers(compressed_vocab_size))
        elif isinstance(module, HYV4FlashEngramEmbedding):
            init.copy_(module.offsets, module.head_offsets())
        elif isinstance(module, HYV4FlashRotaryEmbedding):
            for layer_type, rope_init_fn in module.rope_init_fns.items():
                inv_freq, _ = rope_init_fn(module.config.per_layer_config[layer_type], layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), inv_freq)


class HYV4FlashTextModel(HYV4VLTextModel):
    config: HYV4FlashTextConfig

    def __init__(self, config: HYV4FlashTextConfig):
        super().__init__(config)
        self.rotary_emb = HYV4FlashRotaryEmbedding(config=config)

    def build_ngram_tokens(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor | None) -> torch.Tensor:
        """Stack the token at each position with the `max_ngram_size - 1` tokens before it.

        Positions reaching past the start of their sequence are filled with the pad id, so an n-gram
        never spans two documents of a packed batch.
        """
        config = self.config
        pad_id = config.engram_pad_id
        # The leading position plane restarts at zero on every document, which is also where n-grams
        # must stop. Without positions each row is one document, so plain offsets are enough.
        if position_ids is None:
            document_offsets = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        else:
            document_offsets = position_ids[0] if position_ids.ndim == 3 else position_ids

        shifted = [input_ids]
        for offset in range(1, config.engram_max_ngram_size):
            previous = F.pad(input_ids, (offset, 0), value=pad_id)[:, : input_ids.shape[1]]
            shifted.append(torch.where(document_offsets < offset, pad_id, previous))
        return torch.stack(shifted, dim=-1)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        ngram_tokens: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        r"""
        ngram_tokens (`torch.LongTensor` of shape `(batch_size, sequence_length, engram_max_ngram_size)`, *optional*):
            Token ids of the n-gram ending at each position, built by [`HYV4FlashModel.build_engram_inputs`].
            Required when the model carries Engram memories.
        engram_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Positions that own an n-gram. Visual and padding positions are left out, so the Engram
            memories add nothing there.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and cache is None:
            cache = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = cache.get_seq_length() if cache is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # As in Qwen2-VL, an optional leading text axis is used to build the causal mask
        text_position_ids = None
        if position_ids.shape[0] == 4:
            text_position_ids, position_ids = position_ids[0], position_ids[1:]
        elif position_ids.shape[0] == 1:
            # A text-only step carries one plane; the three M-RoPE axes then all read the same positions.
            text_position_ids = position_ids[0]
            position_ids = position_ids.expand(3, -1, -1)

        # Every indexer scores against the whole history, so the positions of the tokens already in the
        # cache are kept with it. Storing them on the first indexed layer lets cache operations such as
        # reset, crop and beam reordering carry the position history along.
        indexer_position_ids = position_ids
        if cache is not None and "indexed_attention" in self.config.layer_types:
            indexer_cache = cache.layers[self.config.layer_types.index("indexed_attention")]
            if hasattr(indexer_cache, "max_cache_len"):
                # Static cache: write positions in-place into a fixed-shape buffer so CUDAGraphs never
                # recompile or overwrite a dynamically concatenated history tensor across decode steps.
                if indexer_cache.indexer_position_ids is None:
                    indexer_cache.indexer_position_ids = torch.zeros(
                        (position_ids.shape[0], position_ids.shape[1], indexer_cache.max_cache_len),
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )
                    if not torch.compiler.is_compiling():
                        torch._dynamo.mark_static_address(indexer_cache.indexer_position_ids)
                write_slots = (
                    torch.arange(position_ids.shape[-1], device=position_ids.device) + indexer_cache.get_seq_length()
                )
                indexer_cache.indexer_position_ids.index_copy_(-1, write_slots, position_ids)
                indexer_position_ids = indexer_cache.indexer_position_ids
            else:
                if indexer_cache.indexer_position_ids is not None:
                    indexer_position_ids = torch.cat([indexer_cache.indexer_position_ids, position_ids], dim=-1)
                indexer_cache.indexer_position_ids = indexer_position_ids

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache": cache,
                "position_ids": text_position_ids,
                "allow_is_causal_skip": False,  # Always force creation to account for the indexer's selection
            }
            causal_mask_mapping = {
                "indexed_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        position_embeddings_mapping = {
            layer_type: self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type=layer_type)
            for layer_type in self.rotary_emb.layer_types
        }
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_type = decoder_layer.attention_type
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[layer_type],
                position_embeddings=position_embeddings_mapping[layer_type],
                position_ids=position_ids,
                cache=cache,
                use_cache=use_cache,
                indexer_position_ids=indexer_position_ids,
                ngram_tokens=ngram_tokens,
                engram_mask=engram_mask,
                **kwargs,
            )

        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, cache=cache)


@auto_docstring
class HYV4FlashModel(HYV4VLModel):
    config: HYV4FlashConfig

    def __init__(self, config: HYV4FlashConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)

    def build_engram_inputs(
        self, input_ids: torch.LongTensor | None, position_ids: torch.LongTensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Hash inputs for the Engram layers, and the mask of positions that own an n-gram.

        Visual spans carry projector output rather than tokens, so no n-gram is defined over them; the
        padding token is neutral for the same reason.
        """
        text_config = self.language_model.config
        if text_config.engram_layer_ids is None or input_ids is None:
            return None, None
        ngram_tokens = self.language_model.build_ngram_tokens(input_ids, position_ids)
        engram_mask = (
            (input_ids != self.config.image_token_id)
            & (input_ids != self.config.video_token_id)
            & (input_ids != text_config.engram_pad_id)
        )
        return ngram_tokens, engram_mask

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        ngram_tokens: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MultimodalMoeBaseModelOutputWithPast:
        r"""
        ngram_tokens (`torch.LongTensor` of shape `(batch_size, sequence_length, engram_max_ngram_size)`, *optional*):
            Compressed token ids of the n-gram ending at each position. Built from `input_ids` when omitted;
            `generate` passes them in because a cached step forwards one token but its n-gram reaches back
            into the history.
        engram_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Positions that own an n-gram, built alongside `ngram_tokens`.
        """
        if (pixel_values is not None or pixel_values_videos is not None) and mm_encoder_outputs is not None:
            raise ValueError(
                "You cannot specify both pixel_values/pixel_values_videos and mm_encoder_outputs at the same time"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **kwargs
            )

        if mm_encoder_outputs.get("video") is None and pixel_values_videos is not None:
            mm_encoder_outputs["video"] = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True, **kwargs
            )

        if mm_encoder_outputs.get("image") is not None:
            image_embeds = torch.cat(mm_encoder_outputs["image"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if mm_encoder_outputs.get("video") is not None:
            video_embeds = torch.cat(mm_encoder_outputs["video"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache=cache,
                mm_token_type_ids=mm_token_type_ids,
            )

        # Key difference: the Engram memories read the token ids, which the text backbone never sees
        # once the visual features have been scattered into the embeddings.
        if ngram_tokens is None:
            ngram_tokens, engram_mask = self.build_engram_inputs(input_ids, position_ids)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            ngram_tokens=ngram_tokens,
            engram_mask=engram_mask,
            **kwargs,
        )

        return MultimodalMoeBaseModelOutputWithPast(
            **outputs, image_outputs=mm_encoder_outputs.get("image"), video_outputs=mm_encoder_outputs.get("video")
        )


@auto_docstring
class HYV4FlashForConditionalGeneration(HYV4VLForConditionalGeneration):
    _keep_in_fp32_modules_strict = HYV4FlashPreTrainedModel._keep_in_fp32_modules_strict + ["lm_head"]

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        # The n-gram ending at the token being generated reaches back into the history, which a cached
        # step no longer forwards, so hash the whole history here and keep the forwarded window.
        ngram_tokens, engram_mask = self.model.build_engram_inputs(input_ids, None)
        if ngram_tokens is not None:
            forwarded = model_inputs["input_ids"].shape[1]
            model_inputs["ngram_tokens"] = ngram_tokens[:, -forwarded:]
            model_inputs["engram_mask"] = engram_mask[:, -forwarded:]
        return model_inputs


@auto_docstring
class HYV4FlashProcessor(HYV4VLProcessor):
    pass


__all__ = [
    "HYV4FlashConfig",
    "HYV4FlashTextConfig",
    "HYV4FlashPreTrainedModel",
    "HYV4FlashModel",
    "HYV4FlashTextModel",
    "HYV4FlashForConditionalGeneration",
    "HYV4FlashProcessor",
]
