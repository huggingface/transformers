# Copyright 2026 IBM and the HuggingFace Inc. team. All rights reserved.
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
"""Granite Switch: a Granite model that carries its LoRA adapters and routes them per token."""

from collections.abc import Callable
from typing import Any

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, slice_logits_to_keep
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..granitemoehybrid.configuration_granitemoehybrid import GraniteMoeHybridConfig
from ..granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridExperts,
    GraniteMoeHybridMLP,
    GraniteMoeHybridMoE,
    GraniteMoeHybridRMSNorm,
    GraniteMoeHybridRotaryEmbedding,
    GraniteMoeHybridTopKRouter,
    apply_rotary_pos_emb,
    eager_attention_forward,
    load_balancing_loss_func,
    rotate_half,
)
from .switch_codes import KerdockDGCodeGenerator, recover_count_from_signal


# The switch model is attention-only. The hybrid parent fills an unset `layer_types` with
# all-`"linear_attention"` (mamba), which would size the per-layer cache wrongly, so the value
# is pinned here instead. It must be a key of the dynamic-cache layer-type mapping.
ATTENTION_LAYER_TYPE = "full_attention"

# Narrowest head the attention kernels accept. Both switch heads carry their signal in a handful
# of components and pad the rest, so this floor -- not the signal -- is what sets the counting
# head's width, and the memory head's when a codeword is narrower than it.
_MIN_ATTENTION_HEAD_DIM = 32


@auto_docstring(checkpoint="ibm-granite/granite-switch-4.1-3b-preview")
@strict
class GraniteSwitchConfig(GraniteMoeHybridConfig):
    r"""
    A Granite Switch checkpoint carries its LoRA adapters baked in and selects one per token from a
    control-token signal, so every field describing the adapter population is frozen into
    `config.json` when the checkpoint is built. This class is the on-disk contract: it is what makes
    a checkpoint self-describing, and its validation is what turns a malformed checkpoint into a
    load-time error instead of silently wrong routing.

    The switch engine is the Kerdock/Delsarte-Goethals coded-memory MultiSwitch. Control tokens are
    handled exclusively through token exchange: the switch reads `input_ids`, decides the active
    adapter, and rewrites each control token to its substitute id (from
    `adapter_substitute_token_ids`) before the decoder embeds the sequence. The decoder never sees a
    control token.

    num_adapters (`int`, *optional*, defaults to 0):
        Number of LoRA adapters embedded in the checkpoint. Counts real adapters only, not the base;
        adapter index 0 always means "base / no adapter".
    adapter_token_ids (`list[int]`, *optional*):
        Token ids that activate an adapter. Length is `num_adapters`, or `num_adapters + 1` with a
        leading base-reset token that returns the stream to base mid-request. Must be unique, since
        token exchange keys its lookup by control-token id. `adapter_token_ids[i]` activates adapter
        `i + 1`; adapter 0 is base and needs no token. Required when `num_adapters > 0`.
    adapter_substitute_token_ids (`list[int]`, *optional*):
        Token ids whose embeddings replace the control-token embeddings before the decoder runs.
        Same length as `adapter_token_ids`. Required when `num_adapters > 0`.
    adapter_names (`list[str]`, *optional*):
        Ordered adapter names, for name-to-index mapping. Length `num_adapters`.
    max_lora_rank (`int`, *optional*, defaults to 8):
        Maximum rank across all adapters. Every adapter's weights are zero-padded to this rank on
        disk so one kernel can serve a rank-heterogeneous adapter set.
    adapter_ranks (`list[int]`, *optional*):
        Per-adapter LoRA ranks. Length `num_adapters`, and the maximum must equal `max_lora_rank`.
        Required when `num_adapters > 0`.
    lora_target_modules (`list[str]`, *optional*):
        Module *group* names to apply LoRA to, drawn from `qkv_proj`, `o_proj`,
        `shared_input_linear` and `shared_output_linear`. Defaults to the groups the architecture
        actually has: a pure sparse-MoE base has no shared MLP, so asking for its groups would build
        zero-width LoRA projections.
    control_token_gain (`float`, *optional*, defaults to 15.0):
        Attention gain that separates control tokens from ordinary tokens in the switch heads.
    switch_head_dim (`int`, *optional*, defaults to 32):
        Dimension of the switch attention Q/K/V vectors.
    ms_code_m (`int`, *optional*, defaults to 6):
        Code parameter of the coded-memory head. A kerdock codebook holds `2 ** (2 * m - 1)` vectors
        of dimension `2 ** m`.
    ms_code_type (`str`, *optional*, defaults to `"kerdock"`):
        Codebook family, `"kerdock"` or `"dg1"`.
    ms_memory_gain (`float`, *optional*, defaults to 28.0):
        Attention gain applied to the memory head's code vectors.
    ms_counting_head_dim (`int`, *optional*, defaults to 32):
        Head dimension of the counting head, whose `1 / (1 + n)` signal recovers the write address.
    dual_stream (`bool`, *optional*, defaults to `False`):
        Whole-checkpoint decoder mode. `False` is ordinary LoRA/aLoRA, one stream. `True` is Shadow
        Residual, where every adapter runs a frozen base stream alongside an adapter stream and
        takes K/V from the base stream. A checkpoint holds either Shadow Residual adapters or
        LoRA/aLoRA adapters, never both, so this is a single flag rather than a per-adapter list.
    cross_stream_rank (`int`, *optional*):
        LoRA rank of the layer-level `cross_stream` injection site. Required when `dual_stream` is
        `True`, and must be `None` otherwise, since the site is not allocated at all.
    shared_intermediate_size (`int`, *optional*):
        Width of the dense shared MLP. Defaults to `intermediate_size` for a dense base, and to 0
        for a pure sparse-MoE base, where 0 is the "no shared MLP" sentinel: the layer is built
        without one rather than with a zero-width one.
    position_embedding_type (`str`, *optional*, defaults to `"rope"`):
        Positional embedding type. Allowed options: `[None, "rope"]`.

    Example:

    ```python
    >>> from transformers import GraniteSwitchConfig, GraniteSwitchModel

    >>> # Initializing a Granite Switch configuration
    >>> configuration = GraniteSwitchConfig()

    >>> # Initializing a model from the configuration
    >>> model = GraniteSwitchModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "granite_switch"

    # The parent aliases `layers_block_type` onto `layer_types` for hybrid checkpoints. Granite
    # Switch is attention-only and ships neither name under an alias, so the indirection is retired
    # rather than inherited.
    attribute_map = AttributeError()

    num_adapters: int = 0
    adapter_token_ids: list[int] | None = None
    adapter_substitute_token_ids: list[int] | None = None
    adapter_names: list[str] | None = None
    max_lora_rank: int = 8
    adapter_ranks: list[int] | None = None
    lora_target_modules: list[str] | None = None

    control_token_gain: float = 15.0
    switch_head_dim: int = 32

    ms_code_m: int = 6
    ms_code_type: str = "kerdock"
    ms_memory_gain: float = 28.0
    ms_counting_head_dim: int = 32

    dual_stream: bool = False
    cross_stream_rank: int | None = None

    # The parent's fixed 1024 default is the wrong width for a dense base and cannot express the
    # "no shared MLP" sentinel (0) that a pure sparse-MoE base needs, so an unset value is resolved
    # in `__post_init__` instead. An explicit value, including 0, is honored.
    shared_intermediate_size: int | None = None
    # Granite 4 dense is the default configuration.
    num_local_experts: int | None = 0
    position_embedding_type: str | None = "rope"
    layer_types: list[str] | None = None
    # Some bases set `head_dim != hidden_size // num_attention_heads`, and the fused `qkv_proj`
    # output width is sized from it. Resolved in `__post_init__` so the value a checkpoint was built
    # against is serialized rather than re-derived by every reader.
    head_dim: int | None = None

    # Fields declared by GraniteMoeHybridConfig that Granite Switch does not use: it is
    # attention-only, so no layer carries mamba state.
    mamba_n_heads = AttributeError()
    mamba_n_groups = AttributeError()
    mamba_d_state = AttributeError()
    mamba_d_head = AttributeError()
    mamba_d_conv = AttributeError()
    mamba_expand = AttributeError()
    mamba_chunk_size = AttributeError()
    mamba_conv_bias = AttributeError()
    mamba_proj_bias = AttributeError()
    time_step_limit = AttributeError()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs):
        """Reject SingleSwitch checkpoints; MultiSwitch is the only engine.

        A checkpoint built for the earlier SingleSwitch engine cannot run as MultiSwitch: it
        carries a one-transition switch whose weights have no counterpart here, and its
        `num_hidden_layers` follows the old convention of counting the switch's cache slots as
        decoder layers. There is no auto-migration; it must be rebuilt.

        This is the one seam that still sees the raw on-disk config, so the rejection lives here.
        Two shapes identify such a checkpoint, with adapters present:

        - an explicit `switch_type` that is not `"multi"`; or
        - no `switch_type` key and no `ms_code_m` key, which marks a checkpoint predating the coded
          engine. `ms_code_m` is serialized by every MultiSwitch checkpoint, so its absence is the
          reliable marker.

        A stale `switch_type` key on a real MultiSwitch checkpoint is stripped before delegating, so
        the removed parameter never reaches the parent.
        """
        if config_dict.get("num_adapters", 0) > 0:
            switch_type = config_dict.get("switch_type")
            legacy_single = switch_type is None and "ms_code_m" not in config_dict
            if (switch_type is not None and switch_type != "multi") or legacy_single:
                raise ValueError(
                    "This checkpoint was built for the SingleSwitch engine, which has been removed. "
                    "MultiSwitch is now the only engine, and a SingleSwitch checkpoint cannot be loaded as "
                    "MultiSwitch because it was sized for a different decoder-layer count. A loadable "
                    "checkpoint must set switch_type to 'multi' or omit it entirely, and must carry "
                    "ms_code_m; rebuild it from its source adapters."
                )
        if "switch_type" in config_dict:
            config_dict = {k: v for k, v in config_dict.items() if k != "switch_type"}
        config = super().from_dict(config_dict, **kwargs)
        return config

    def __post_init__(self, **kwargs):
        # A pre-fusion Shadow Residual checkpoint carries unfused Q/K/V and gate/up weights, which
        # would load into the fused model with silently mismatched keys. `PreTrainedConfig` keeps
        # unknown keys, so the marker survives to here and the rejection has to be explicit.
        if kwargs.get("unfused_qkv"):
            raise ValueError(
                "This checkpoint sets unfused_qkv=True, so it carries the old unfused Shadow Residual "
                "weight layout. Only the fused layout is supported: qkv_proj and shared_mlp.input_linear "
                "must be single pre-fused tensors. Rebuild the checkpoint."
            )

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.layer_types is None:
            # Length must equal `num_hidden_layers`, cache slots included: they are attention too.
            self.layer_types = [ATTENTION_LAYER_TYPE] * self.num_hidden_layers

        if self.shared_intermediate_size is None:
            self.shared_intermediate_size = 0 if self.num_local_experts > 0 else self.intermediate_size

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.lora_target_modules is None:
            lora_target_modules = []
            if self.num_adapters > 0:
                # The switch model is attention-only, so every layer has the attention groups.
                lora_target_modules += ["qkv_proj", "o_proj"]
                # The MLP groups exist only where a shared MLP does to hold them.
                if self.shared_intermediate_size > 0:
                    lora_target_modules += ["shared_input_linear", "shared_output_linear"]
            self.lora_target_modules = lora_target_modules

        # The hybrid parent's `__post_init__` resolves mamba geometry from fields this config
        # retires, so skip it and delegate straight to the base implementation.
        PreTrainedConfig.__post_init__(self, **kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Checks the adapter population is self-consistent."""
        if self.num_adapters < 0:
            raise ValueError(f"num_adapters must be >= 0, got {self.num_adapters}")

        if self.num_adapters == 0:
            return

        # MultiSwitch accepts `num_adapters` control tokens, or `num_adapters + 1` with a leading
        # base-reset token that writes adapter 0 and so enables return-to-base mid-request.
        allowed_lengths = (self.num_adapters, self.num_adapters + 1)

        if self.adapter_token_ids is None:
            raise ValueError(
                "adapter_token_ids is required when num_adapters > 0: every adapter needs a control token "
                "for the switch to route on."
            )
        if len(self.adapter_token_ids) not in allowed_lengths:
            raise ValueError(
                f"adapter_token_ids length ({len(self.adapter_token_ids)}) must be one of {allowed_lengths} "
                f"(num_adapters={self.num_adapters})."
            )
        if len(set(self.adapter_token_ids)) != len(self.adapter_token_ids):
            raise ValueError(f"adapter_token_ids must be unique; got {self.adapter_token_ids}")

        if self.adapter_substitute_token_ids is None:
            raise ValueError(
                "adapter_substitute_token_ids is required when num_adapters > 0. Every adapter needs a "
                "substitute token id whose embedding replaces the control-token embedding before the "
                "decoder runs."
            )
        if len(self.adapter_substitute_token_ids) != len(self.adapter_token_ids):
            raise ValueError(
                "adapter_token_ids and adapter_substitute_token_ids must have the same length; got "
                f"{len(self.adapter_token_ids)} and {len(self.adapter_substitute_token_ids)}."
            )
        if any(token_id < 0 for token_id in self.adapter_substitute_token_ids):
            raise ValueError(
                "adapter_substitute_token_ids must all be >= 0 (real token ids); got "
                f"{self.adapter_substitute_token_ids}"
            )

        if self.adapter_ranks is None:
            raise ValueError("adapter_ranks must be provided when num_adapters > 0")
        if len(self.adapter_ranks) != self.num_adapters:
            raise ValueError(
                f"adapter_ranks length ({len(self.adapter_ranks)}) must equal num_adapters ({self.num_adapters})"
            )
        if max(self.adapter_ranks) != self.max_lora_rank:
            raise ValueError(
                f"max(adapter_ranks)={max(self.adapter_ranks)} must equal max_lora_rank={self.max_lora_rank}. "
                "Adapter weights are zero-padded to max_lora_rank on disk, so the two must agree or the "
                "padding width is wrong."
            )

    def validate_shadow_residual(self):
        """Part of `@strict`-powered validation. Checks the Shadow Residual fields agree."""
        if self.dual_stream and self.cross_stream_rank is None:
            raise ValueError(
                "cross_stream_rank is required when dual_stream is True: the cross_stream injection site "
                "must be allocated."
            )
        if not self.dual_stream and self.cross_stream_rank is not None:
            raise ValueError(
                f"cross_stream_rank must be None when dual_stream is False; got {self.cross_stream_rank}. "
                "The cross_stream site only exists in a Shadow Residual checkpoint."
            )


def build_control_to_substitute_lut(config: GraniteSwitchConfig) -> torch.Tensor | None:
    """Derive the control-token to substitute-token lookup table from `config`.

    Shape `[max(vocab_size, max_control_id + 1)]`, holding `-1` at every non-control id and the
    substitute id at each control slot. Indexing it with `input_ids` therefore yields `-1` wherever
    a token is ordinary, which is what lets `apply_token_exchange` rewrite the sequence without a
    data-dependent branch.

    The `max` is what sizes the table when a checkpoint's control ids sit past `vocab_size`. That
    happens while a checkpoint is being built: the control tokens are appended to the tokenizer
    before the embedding matrix is resized, so for a window the largest control id exceeds the
    recorded vocabulary and sizing at `vocab_size` alone would make the assignment below raise
    `IndexError`. A finished checkpoint has been resized, so the table comes out at exactly
    `vocab_size` and the `max` no longer fires.

    Only the real adapter control tokens are mapped. A base-reset token, when the checkpoint has
    one, is deliberately left alone: it is only ever placed at turn boundaries, which are
    in-distribution positions for the base model, so its embedding needs no substitute.

    Returns `None` when the config carries no mapping, in which case the switch leaves `input_ids`
    untouched. An empty id list counts as no mapping, which is why this tests falsiness rather than
    `is None` — `max(())` raises.
    """
    control_ids = config.adapter_token_ids
    substitute_ids = config.adapter_substitute_token_ids
    if not control_ids or not substitute_ids:
        return None

    lut_size = max(config.vocab_size, max(control_ids) + 1)
    lut = torch.full((lut_size,), -1, dtype=torch.long)
    for control_id, substitute_id in zip(control_ids, substitute_ids):
        lut[control_id] = substitute_id
    return lut


def apply_token_exchange(lut: torch.Tensor | None, input_ids: torch.Tensor) -> torch.Tensor:
    """Rewrite each control token's id to its substitute id through `lut`.

    The switch has already read the original ids to decide the active adapter by the time this
    runs, so rewriting here is what keeps control tokens invisible to the decoder: it embeds the
    substitute instead, and never sees an id it was not trained on.

    `torch.where` runs unconditionally rather than short-circuiting on "are there any control
    tokens", because a `.any()` test is a data-dependent branch that would graph-break under
    `torch.compile`. When `lut` is `None` the input is returned unchanged.

    Ids at or past the end of the table are treated as ordinary tokens instead of indexing out of
    bounds. The table is sized at construction, so `resize_token_embeddings` can grow the vocabulary
    past it; every id it adds is by definition not a control token, since the control ids come from
    the config. Hence the gather is clamped and the out-of-range positions are masked out again,
    which keeps the whole thing branch-free.
    """
    if lut is None:
        return input_ids
    is_in_table = input_ids < lut.shape[0]
    substitute_per_position = lut[torch.where(is_in_table, input_ids, 0)]
    is_control = is_in_table & (substitute_per_position >= 0)
    return torch.where(is_control, substitute_per_position, input_ids)


class GraniteSwitchCountingAttention(nn.Module):
    """Attribute carrier for the counting head's `sdpa_attention_forward` call.

    Holds no parameters and defines no `forward`. The attention interface reads its geometry off
    the module it is handed (`num_key_value_groups`, `is_causal`, `layer_idx`), so the head needs
    an object with those attributes; `MultiSwitch` builds the Q/K/V tensors itself.

    One query head over one KV head, at `counting_head_dim`, which is independent of the
    backbone's `head_dim`. Only component 0 of each vector carries signal; the rest is zero
    padding to reach the minimum width the attention kernels accept.

    `layer_idx` is `0` and unused: the head is handed its full Q/K/V on every call and never writes
    to the model's cache, so it holds no cache slot to be addressed by.
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.num_heads = 1
        self.num_key_value_heads = 1
        self.num_key_value_groups = 1
        self.head_dim = head_dim
        self.scaling = 1.0
        self.attention_dropout = 0.0
        self.is_causal = True
        self.layer_idx = 0


class GraniteSwitchMemoryAttention(nn.Module):
    """Attribute carrier for the memory head's `sdpa_attention_forward` call.

    Same role as `GraniteSwitchCountingAttention`, at `memory_head_dim`, which must be wide enough
    to hold a whole codeword.
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.num_heads = 1
        self.num_key_value_heads = 1
        self.num_key_value_groups = 1
        self.head_dim = head_dim
        self.scaling = 1.0
        self.attention_dropout = 0.0
        self.is_causal = True
        self.layer_idx = 0


class MultiSwitch(nn.Module):
    """Picks one adapter per token from the control tokens seen so far, latest wins.

    Two single-head attention passes do the work. A counting head turns "how many control tokens
    precede this position" into a write address, and a coded-memory head reads back the adapter id
    written at that address. Addresses are Kerdock/Delsarte-Goethals codewords, whose bounded
    coherence is what makes retrieval land on an exact integer rather than a blend of the adapters
    in scope. A request may therefore switch adapters any number of times, and -- on a checkpoint
    composed with a base-reset token -- switch back to no adapter at all.

    Index 0 always means base/no adapter, so valid indices are `0..num_adapters`.
    """

    def __init__(self, config: GraniteSwitchConfig):
        super().__init__()
        self.config = config
        self.num_adapters = config.num_adapters
        self.control_token_gain = config.control_token_gain

        # Two accepted `adapter_token_ids` layouts, distinguished by length alone. With
        # `num_adapters` entries, entry i fires adapter i + 1 and the id is argmax + 1. With
        # `num_adapters + 1`, entry 0 is a base-reset token firing 0 and the id is the argmax
        # unshifted -- which is what lets a request transition back to base mid-sequence.
        # Derived from a length once here, rather than per step, so `forward` stays branch-free.
        if len(config.adapter_token_ids) == self.num_adapters + 1:
            self._expert_id_offset = 0
        else:
            self._expert_id_offset = 1

        # `precompute_codebook` returns `[capacity, N]` unit-norm codewords; N is both the code
        # and the memory dimension. For the m=6 Kerdock default that is 2048 addresses of width 64.
        self.code_gen = KerdockDGCodeGenerator(m=config.ms_code_m, code_type=config.ms_code_type)
        self.capacity = self.code_gen.capacity
        self.memory_dim = self.code_gen.N
        self.memory_gain = config.ms_memory_gain

        # Must be persistent. `from_pretrained` fills only the tensors the checkpoint carries, so a
        # non-persistent buffer comes back all zeros and whatever `__init__` computed is discarded.
        # A zeroed codebook zeroes every memory key and query, collapsing retrieval to a uniform
        # softmax: each position averages the adapter ids in scope instead of taking the most
        # recent, and routes to a blend silently. It also costs only ~512 KB.
        self.codebook = nn.Buffer(self.code_gen.precompute_codebook(dtype=torch.float32), persistent=True)

        # Counting carries its one-hot signal in component 0 and pads the rest, so its width is
        # set by what the attention kernels accept rather than by the signal.
        self.counting_head_dim = max(config.ms_counting_head_dim, _MIN_ATTENTION_HEAD_DIM)

        # Memory must hold a whole codeword. Prefer the backbone's `head_dim` when it is wide
        # enough, so every attention layer in the model shares one head width; otherwise take the
        # code dimension directly. Surplus components are zero-padded.
        if config.head_dim >= self.memory_dim:
            self.memory_head_dim = config.head_dim
        else:
            self.memory_head_dim = max(self.memory_dim, _MIN_ATTENTION_HEAD_DIM)

        self.counting_attn_module = GraniteSwitchCountingAttention(head_dim=self.counting_head_dim)
        self.memory_attn_module = GraniteSwitchMemoryAttention(head_dim=self.memory_head_dim)

        # Persistent for the same reason as `codebook`, and the consequence is worse here: this
        # table uses -1 for "not a control token", so an all-zero table would rewrite every token
        # id in the sequence to 0.
        lut = build_control_to_substitute_lut(config)
        self.control_to_substitute_lut = None if lut is None else nn.Buffer(lut, persistent=True)

    def _apply(self, *args, **kwargs):
        """Hold `codebook` at fp32 across `.to()` / `.half()` / `.bfloat16()`.

        `nn.Module._apply` casts every float buffer, so casting the model would take the codebook
        with it and make save/load non-idempotent: `__init__` rebuilds this buffer as fp32 on every
        load, so a cast model serializes bf16 while a reload of that same model serializes fp32 --
        the same model writing two different byte counts, differing by exactly the codebook.

        Serialization is the only reason, and the forward path must not be cited as another. The
        entries are +/-1/sqrt(N), exact in bf16, and the memory key they feed is assigned into an
        fp32 destination which upcasts whatever arrives.
        """
        out = super()._apply(*args, **kwargs)
        if self.codebook.dtype != torch.float32:
            self.codebook = self.codebook.to(torch.float32)
        return out

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build the heads' own fp32 causal mask.

        Deliberately not the model's `attention_mask`. That one is created in the model dtype,
        and handing a bf16 mask to these fp32 attention calls makes the whole op run in bf16,
        whose resolution corrupts the counting head's `1/(1+n)` recovery: the recovered index
        comes out off by one and routing lags a position. Which keys actually count is already
        encoded in the fp32 key masking, so this only has to be causal and fp32.
        """
        return torch.triu(
            torch.full((seq_len, seq_len), torch.finfo(torch.float32).min, device=device, dtype=torch.float32),
            diagonal=1,
        ).view(1, 1, seq_len, seq_len)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select an adapter per token, then rewrite the control tokens.

        Selection reads the original `input_ids`; the rewrite happens last, so the returned ids
        are what the decoder embeds and the control tokens never reach it.

        `input_ids` must be the sequence from position 0, not a decode step's single new token:
        routing is latest-wins over the control tokens seen so far, so the whole prefix has to be
        in scope. The switch is stateless and keeps no cache of its own, which is affordable
        because every tensor below is a pure function of the ids -- recomputing the prefix costs
        two single-head attentions and nothing else. `GraniteSwitchModel` is what supplies the
        full sequence during generation.

        Returns `(adapter_indices, modified_input_ids)`, both `[batch, seq_len]`, where index 0
        means base.
        """
        bsz, q_len = input_ids.shape
        device = input_ids.device

        # Position 0 anchors the counting signal, taking the place of a dedicated init token.
        is_counting_anchor = (torch.arange(q_len, device=device) == 0).unsqueeze(0).expand(bsz, -1)

        control_token_ids = torch.tensor(self.config.adapter_token_ids, device=device)
        matches = input_ids.unsqueeze(2) == control_token_ids.unsqueeze(0).unsqueeze(0)
        is_control_token = matches.any(dim=2)
        expert_ids = torch.where(
            is_control_token,
            matches.long().argmax(dim=2) + self._expert_id_offset,
            torch.zeros_like(input_ids, dtype=torch.long),
        )

        # Counting head: recover the write address from a 1/(1+n) attention average.
        # Masked-out keys use the fp32 floor rather than a magic sentinel.
        key_states_count = torch.full(
            (bsz, 1, q_len, self.counting_head_dim),
            torch.finfo(torch.float32).min,
            device=device,
            dtype=torch.float32,
        )
        anchor_or_control = is_counting_anchor | is_control_token
        key_states_count[:, 0, :, 0] = torch.where(
            anchor_or_control,
            torch.zeros_like(key_states_count[:, 0, :, 0]),
            key_states_count[:, 0, :, 0],
        )

        value_states_count = torch.zeros((bsz, 1, q_len, self.counting_head_dim), device=device, dtype=torch.float32)
        value_states_count[:, 0, :, 0] = is_counting_anchor.to(torch.float32)

        query_states_count = torch.zeros((bsz, 1, q_len, self.counting_head_dim), device=device, dtype=torch.float32)
        query_states_count[:, 0, :, 0] = 1.0

        # Both heads force SDPA rather than the model's configured backend, because both build
        # fp32 Q/K/V and both need that precision. The memory read is a softmax over coded keys
        # whose low mutual coherence only survives in fp32, and dispatching fp32 tensors through
        # a backend that requires fp16/bf16 silently corrupts the read and routes every token to
        # base -- isolated switch tests still pass while the full model returns all zeros.
        attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]

        # Autocast off for the same reason. Under an enclosing bf16 autocast, SDPA would cast the
        # fp32 Q/K/V and the fp32 mask down to bf16; for the memory head that is fatal, since its
        # keys are `code(n) * memory_gain` and the mask is the fp32 floor, which together overflow
        # the bf16 softmax to NaN and clamp to base.
        autocast_device = device.type if device.type in ("cuda", "cpu") else "cuda"
        with torch.autocast(device_type=autocast_device, enabled=False):
            count_output, _ = attention_interface(
                self.counting_attn_module,
                query_states_count,
                key_states_count,
                value_states_count,
                self._causal_mask(q_len, device),
                dropout=0.0,
                scaling=self.counting_attn_module.scaling,
            )
        counting_signal = count_output[:, :, 0, 0]
        write_addresses = recover_count_from_signal(counting_signal, capacity=self.capacity)

        # Memory head: write the adapter id at each control token's coded address, then have
        # every position query its own address to read back the most recent one.
        code_vectors = self.codebook[write_addresses.reshape(-1)].reshape(bsz, q_len, self.memory_dim)
        write_mask = is_control_token.unsqueeze(2).to(torch.float32)

        key_states_memory = torch.zeros((bsz, 1, q_len, self.memory_head_dim), device=device, dtype=torch.float32)
        key_states_memory[:, 0, :, : self.memory_dim] = code_vectors * self.memory_gain * write_mask

        value_states_memory = torch.zeros((bsz, 1, q_len, self.memory_head_dim), device=device, dtype=torch.float32)
        value_states_memory[:, 0, :, 0] = expert_ids.to(torch.float32) * is_control_token.to(torch.float32)

        query_states_memory = torch.zeros((bsz, 1, q_len, self.memory_head_dim), device=device, dtype=torch.float32)
        query_states_memory[:, 0, :, : self.memory_dim] = code_vectors

        with torch.autocast(device_type=autocast_device, enabled=False):
            memory_output, _ = attention_interface(
                self.memory_attn_module,
                query_states_memory,
                key_states_memory,
                value_states_memory,
                self._causal_mask(q_len, device),
                dropout=0.0,
                scaling=self.memory_attn_module.scaling,
            )

        adapter_indices = torch.clamp(torch.round(memory_output[:, :, 0, 0]).long(), 0, self.num_adapters)
        modified_input_ids = apply_token_exchange(self.control_to_substitute_lut, input_ids)
        return adapter_indices, modified_input_ids


def _add_switched_lora(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    adapter_indices: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
) -> torch.Tensor:
    """
    Add each token's LoRA contribution into `output`, in place, on flattened `[num_tokens, dim]`
    inputs. One matmul pair runs per adapter actually present in the batch rather than one per
    token, which is what keeps a mixed batch to a handful of dense matmuls.

    `lora_B` is pre-scaled by `alpha / rank` when the checkpoint is built, so there is no runtime
    scaling here, and no `alpha` to carry on the module.
    """
    for adapter_idx in adapter_indices[adapter_indices > 0].unique():
        token_indices = torch.where(adapter_indices == adapter_idx)[0]
        # Adapter `i` lives at stacked slice `i - 1`; slot 0 of `adapter_indices` is the base model.
        lora_a = lora_A[adapter_idx - 1, 0]
        lora_b = lora_B[adapter_idx - 1, 0]
        hidden_states_adapter = hidden_states[token_indices]
        lora_output = torch.matmul(torch.matmul(hidden_states_adapter, lora_a.t()), lora_b.t())
        output[token_indices] += lora_output
    return output


class GraniteSwitchLoRALinear(nn.Module):
    r"""
    A linear layer that carries every embedded LoRA adapter and picks one per token.

    The adapter weights are stacked into two parameters indexed by adapter, so a single module
    serves the whole adapter population and the switch's per-token `adapter_indices` chooses which
    slice each token is routed through. Index `0` means "base model, no LoRA"; adapter `i` reads
    stacked slice `i - 1`. Adapters whose rank is below `max_lora_rank` are zero-padded on disk, so
    the padded columns contribute nothing.

    Args:
        in_features (`int`):
            Input dimension.
        out_features (`int`):
            Output dimension.
        num_adapters (`int`):
            Number of embedded LoRA adapters.
        max_lora_rank (`int`):
            Rank every stacked adapter is padded to. This is the on-disk padding contract, not the
            largest rank actually in use.
        bias (`bool`, *optional*, defaults to `True`):
            Whether the base linear layer carries a bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_adapters: int,
        max_lora_rank: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_adapters = num_adapters
        self.max_lora_rank = max_lora_rank

        self.base_layer = nn.Linear(in_features, out_features, bias=bias)

        # The singleton second dimension is part of the on-disk layout, shared with the vLLM
        # backend's stacked LoRA tensors, and is what keeps one checkpoint loadable by both.
        self.lora_A = nn.Parameter(torch.empty(num_adapters, 1, max_lora_rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(num_adapters, 1, out_features, max_lora_rank))

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x: torch.Tensor, adapter_indices: torch.Tensor | None = None) -> torch.Tensor:
        output = self.base_layer(x)

        # Nothing to add when every token is on the base model, which is the whole sequence for an
        # unadapted request and the prefix of an adapted one before its first control token.
        if adapter_indices is None or not torch.any(adapter_indices > 0):
            return output

        is_batched = x.dim() == 3
        if is_batched:
            batch_size, seq_len, _ = x.shape
            x_flat = x.reshape(-1, self.in_features)
            adapter_indices_flat = adapter_indices.reshape(-1)
            output_flat = output.view(-1, self.out_features)
        else:
            x_flat = x
            adapter_indices_flat = adapter_indices
            output_flat = output

        output_flat = _add_switched_lora(x_flat, output_flat, adapter_indices_flat, self.lora_A, self.lora_B)

        if is_batched:
            return output_flat.view(batch_size, seq_len, self.out_features)
        return output_flat


class GraniteSwitchMergedLoRALinear(nn.Module):
    r"""
    The fused-projection counterpart of [`GraniteSwitchLoRALinear`], used for QKV and gate/up.

    The base projection is a single fused `nn.Linear`, but LoRA stays per slice: fusing the adapter
    weights too would force every slice to share one rank. The slices are held in `ParameterList`s
    so the parameter names are `lora_A_slices.0`, `lora_A_slices.1`, ... — the same names the vLLM
    backend writes, which is what makes one checkpoint serve both.

    Args:
        in_features (`int`):
            Input dimension, shared by all slices.
        output_slices (`tuple[int, ...]`):
            Output dimension of each fused slice, e.g. `(q_size, kv_size, kv_size)` for QKV or
            `(intermediate_size, intermediate_size)` for gate/up.
        num_adapters (`int`):
            Number of embedded LoRA adapters.
        max_lora_rank (`int`):
            Rank every stacked adapter is padded to.
        bias (`bool`, *optional*, defaults to `False`):
            Whether the fused base linear layer carries a bias.
    """

    def __init__(
        self,
        in_features: int,
        output_slices: tuple[int, ...],
        num_adapters: int,
        max_lora_rank: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.output_slices = output_slices
        self.total_out_features = sum(output_slices)
        self.num_slices = len(output_slices)
        self.num_adapters = num_adapters
        self.max_lora_rank = max_lora_rank

        self.base_layer = nn.Linear(in_features, self.total_out_features, bias=bias)

        self.lora_A_slices = nn.ParameterList(
            [nn.Parameter(torch.empty(num_adapters, 1, max_lora_rank, in_features)) for _ in output_slices]
        )
        self.lora_B_slices = nn.ParameterList(
            [nn.Parameter(torch.empty(num_adapters, 1, output_size, max_lora_rank)) for output_size in output_slices]
        )

    def forward(self, x: torch.Tensor, adapter_indices: torch.Tensor | None = None) -> torch.Tensor:
        output = self.base_layer(x)

        if adapter_indices is None or not torch.any(adapter_indices > 0):
            return output

        is_batched = x.dim() == 3
        if is_batched:
            batch_size, seq_len, _ = x.shape
            x_flat = x.reshape(-1, self.in_features)
            adapter_indices_flat = adapter_indices.reshape(-1)
            output_flat = output.view(-1, self.total_out_features)
        else:
            x_flat = x
            adapter_indices_flat = adapter_indices
            output_flat = output

        offset = 0
        for slice_idx, output_size in enumerate(self.output_slices):
            # The slice is a view, so writing through it updates the fused output in place.
            _add_switched_lora(
                x_flat,
                output_flat[:, offset : offset + output_size],
                adapter_indices_flat,
                self.lora_A_slices[slice_idx],
                self.lora_B_slices[slice_idx],
            )
            offset += output_size

        if is_batched:
            return output_flat.view(batch_size, seq_len, self.total_out_features)
        return output_flat


class GraniteSwitchMLP(GraniteMoeHybridMLP):
    """
    The dense shared-expert MLP, with switched LoRA on whichever of its two projections
    `lora_target_modules` names. The fused `input_linear` takes the merged variant because gate and
    up are one matmul; `output_linear` is a plain projection and takes the single variant.

    `forward` takes `adapter_indices` explicitly rather than reading it off the submodules, so the
    per-token routing travels with the activations and the module stays stateless.
    """

    def __init__(self, config: GraniteSwitchConfig):
        super().__init__(config)

        # CODEPATH: both branches fire on an adapter-bearing checkpoint whose base has a dense
        # shared MLP; a `num_adapters == 0` skin has an empty `lora_target_modules` and keeps the
        # plain projections the parent built.
        self.has_input_lora = "shared_input_linear" in config.lora_target_modules
        if self.has_input_lora:
            self.input_linear = GraniteSwitchMergedLoRALinear(
                self.input_size,
                output_slices=(self.hidden_size, self.hidden_size),
                num_adapters=config.num_adapters,
                max_lora_rank=config.max_lora_rank,
                bias=False,
            )

        self.has_output_lora = "shared_output_linear" in config.lora_target_modules
        if self.has_output_lora:
            self.output_linear = GraniteSwitchLoRALinear(
                self.hidden_size,
                self.input_size,
                config.num_adapters,
                config.max_lora_rank,
                bias=False,
            )

    def forward(self, hidden_states: torch.Tensor, adapter_indices: torch.Tensor | None = None) -> torch.Tensor:
        if self.has_input_lora:
            hidden_states = self.input_linear(hidden_states, adapter_indices)
        else:
            hidden_states = self.input_linear(hidden_states)
        chunked_hidden_states = hidden_states.chunk(2, dim=-1)
        hidden_states = self.activation(chunked_hidden_states[0]) * chunked_hidden_states[1]
        if self.has_output_lora:
            hidden_states = self.output_linear(hidden_states, adapter_indices)
        else:
            hidden_states = self.output_linear(hidden_states)
        return hidden_states


class GraniteSwitchAttention(nn.Module):
    """
    Granite attention whose projections carry the embedded LoRA adapters, selected per token.

    Q, K and V come out of one fused `qkv_proj` so the on-disk layout matches the vLLM backend's;
    the adapter weights stay per slice inside [`GraniteSwitchMergedLoRALinear`]. `forward` is the
    ordinary single-stream path, and `forward_dual_stream` is the Shadow Residual adapter stream,
    which computes only Q and attends against the base stream's K/V.
    """

    def __init__(self, config: GraniteSwitchConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.scaling = config.attention_multiplier
        self.is_causal = True

        self.q_size = config.num_attention_heads * self.head_dim
        self.kv_size = config.num_key_value_heads * self.head_dim

        # CODEPATH: the LoRA branches fire on any adapter-bearing checkpoint; the plain-`nn.Linear`
        # branches are for a `num_adapters == 0` skin, whose `lora_target_modules` is empty.
        self.has_qkv_lora = "qkv_proj" in config.lora_target_modules
        if self.has_qkv_lora:
            self.qkv_proj = GraniteSwitchMergedLoRALinear(
                self.hidden_size,
                output_slices=(self.q_size, self.kv_size, self.kv_size),
                num_adapters=config.num_adapters,
                max_lora_rank=config.max_lora_rank,
                bias=config.attention_bias,
            )
        else:
            self.qkv_proj = nn.Linear(self.hidden_size, self.q_size + 2 * self.kv_size, bias=config.attention_bias)

        self.has_o_lora = "o_proj" in config.lora_target_modules
        if self.has_o_lora:
            self.o_proj = GraniteSwitchLoRALinear(
                self.q_size,
                self.hidden_size,
                config.num_adapters,
                config.max_lora_rank,
                bias=config.attention_bias,
            )
        else:
            self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        adapter_indices: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, ...]:
        r"""
        adapter_indices (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token adapter selection from the switch. `0` routes a token through the base model.
        return_kv (`bool`, *optional*, defaults to `False`):
            Whether to also return the post-cache `(key_states, value_states)`, so a Shadow
            Residual adapter stream can attend against the base stream's keys and values.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.has_qkv_lora:
            qkv = self.qkv_proj(hidden_states, adapter_indices)
        else:
            qkv = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        # CODEPATH: `None` on a base that opts out of RoPE with `position_embedding_type=None`.
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if self.has_o_lora:
            attn_output = self.o_proj(attn_output, adapter_indices)
        else:
            attn_output = self.o_proj(attn_output)

        if return_kv:
            return attn_output, attn_weights, (key_states, value_states)
        return attn_output, attn_weights

    def forward_dual_stream(
        self,
        hidden_states: torch.Tensor,
        base_kv: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        adapter_indices: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        The Shadow Residual adapter stream: project Q from the adapter stream's hidden states and
        attend against `base_kv`, the base stream's keys and values.

        Args:
            base_kv (`tuple[torch.Tensor, torch.Tensor]`):
                The base stream's post-cache `(key_states, value_states)`, each of shape
                `(batch_size, num_key_value_heads, kv_length, head_dim)`. Being post-cache, they
                already span the full sequence, which is what makes autoregressive decode work.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.has_qkv_lora:
            qkv = self.qkv_proj(hidden_states, adapter_indices)
        else:
            qkv = self.qkv_proj(hidden_states)
        query_states = qkv[..., : self.q_size].view(hidden_shape).transpose(1, 2)

        # Q only: the base stream already applied RoPE to the keys this stream is about to read.
        # CODEPATH: `None` on a base that opts out of RoPE with `position_embedding_type=None`.
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states = apply_rotary_pos_emb_q_only(query_states, cos, sin)

        key_states, value_states = base_kv

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if self.has_o_lora:
            return self.o_proj(attn_output, adapter_indices)
        return self.o_proj(attn_output)


def apply_rotary_pos_emb_q_only(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """
    Rotary position embedding applied to the queries alone, for the Shadow Residual adapter stream:
    the keys it attends against come from the base stream, which has already rotated them.

    Args:
        q (`torch.Tensor`):
            The query tensor, of shape `(batch_size, num_heads, sequence_length, head_dim)`.
        cos (`torch.Tensor`):
            The cosine part of the rotary embedding.
        sin (`torch.Tensor`):
            The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The dimension along which to unsqueeze `cos` and `sin` so they broadcast over `q`'s
            head dimension.

    Returns:
        `torch.Tensor`: The rotated query tensor.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin)


# Inherited unchanged from the hybrid base model. Declared so that the generated single file owns
# them under its own name rather than importing them, which is what the modular converter expects.
class GraniteSwitchRMSNorm(GraniteMoeHybridRMSNorm):
    pass


class GraniteSwitchRotaryEmbedding(GraniteMoeHybridRotaryEmbedding):
    pass


class GraniteSwitchTopKRouter(GraniteMoeHybridTopKRouter):
    pass


class GraniteSwitchExperts(GraniteMoeHybridExperts):
    pass


class GraniteSwitchMoE(GraniteMoeHybridMoE):
    pass


class GraniteSwitchDecoderLayer(GradientCheckpointingLayer):
    """
    A single-stream decoder layer: attention carrying the embedded LoRA adapters, then whichever
    MLP paths the base model has -- a frozen expert bank when `num_local_experts > 0`, a dense
    `shared_mlp` when `shared_intermediate_size > 0`, or both for the Granite 4 hybrid.

    This is the layer for plain LoRA and aLoRA checkpoints. A Shadow Residual checkpoint uses
    [`GraniteSwitchShadowResidualDecoderLayer`], which subclasses this one and holds exactly the
    same parameters plus a `cross_stream` injection site.
    """

    def __init__(self, config: GraniteSwitchConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.residual_multiplier = config.residual_multiplier
        self.self_attn = GraniteSwitchAttention(config, layer_idx)

        # The expert bank is frozen and carries no LoRA, so it is the upstream module unchanged.
        # CODEPATH: present on the sparse and hybrid MoE bases, absent on a dense one.
        self.has_experts = config.num_local_experts > 0
        self.block_sparse_moe = GraniteSwitchMoE(config) if self.has_experts else None

        # The dense shared MLP is absent entirely on a pure sparse-MoE base (`granitemoe`), which
        # upstream encodes as `shared_intermediate_size == 0`. Construction has to be skipped, not
        # merely made zero-width: `nn.Linear(hidden_size, 0)` still registers a `[0, hidden_size]`
        # weight, which would then be demanded of the base checkpoint and added into the result.
        # CODEPATH: present on the dense and hybrid bases, absent on a pure sparse one.
        self.has_shared_mlp = config.shared_intermediate_size > 0
        self.shared_mlp = GraniteSwitchMLP(config) if self.has_shared_mlp else None

        if not self.has_experts and not self.has_shared_mlp:
            raise ValueError(
                "A decoder layer needs at least one MLP path, but this config has neither: "
                "`num_local_experts=0` and `shared_intermediate_size=0`."
            )

        self.input_layernorm = GraniteSwitchRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GraniteSwitchRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the frozen router alone, without touching the expert bank, so that Shadow Residual can
        route once and apply the same partition to both of its streams.

        Returns `(top_k_index, top_k_weights)`: the per-token top-k expert indices and their
        softmaxed gate weights. The router's first return value is the raw logits, which the output
        recorder already collects off the router module itself.
        """
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        _, top_k_weights, top_k_index = self.block_sparse_moe.router(flat)
        return top_k_index, top_k_weights

    def _apply_experts(self, hidden_states: torch.Tensor, routing: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        The expert bank under a routing decision made elsewhere. This is the second half of
        `GraniteSwitchMoE.forward`, duplicated because upstream exposes no seam between routing and
        expert application.
        """
        top_k_index, top_k_weights = routing
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_size)
        expert_output = self.block_sparse_moe.experts(flat, top_k_index, top_k_weights)
        return expert_output.view(batch_size, seq_len, hidden_size)

    def _mlp_block(
        self,
        hidden_states: torch.Tensor,
        adapter_indices: torch.Tensor | None,
        routing: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        The expert bank (where there is one) plus the dense shared MLP (where there is one), for a
        single stream.

        `routing` reuses a partition produced by `_route` on *another* stream instead of routing
        this one. Only the routing is shared: the shared MLP still runs under this stream's own
        `adapter_indices`.
        """
        moe_output = None
        if self.has_experts:
            # CODEPATH: `routing is None` in the single-stream layer, which has only the one stream
            # and lets the MoE module route it.
            if routing is None:
                moe_output = self.block_sparse_moe(hidden_states)
            else:
                moe_output = self._apply_experts(hidden_states, routing)
        if not self.has_shared_mlp:
            return moe_output

        shared_output = self.shared_mlp(hidden_states, adapter_indices)
        if moe_output is None:
            return shared_output
        return moe_output + shared_output

    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        adapter_indices: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        adapter_indices (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token adapter selection from the switch. `0` routes a token through the base model.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            adapter_indices=adapter_indices,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self._mlp_block(hidden_states, adapter_indices)
        return residual + hidden_states * self.residual_multiplier


class GraniteSwitchShadowResidualDecoderLayer(GraniteSwitchDecoderLayer):
    """
    A Shadow Residual decoder layer: a frozen base stream and a LoRA'd adapter stream, run through
    one set of weights.

    It holds exactly the parameters of its single-stream parent plus one `cross_stream` injection
    site. Every module is shared between the two streams -- the base stream simply calls them with
    `adapter_indices=None`, which makes each switched LoRA linear fall back to its base weight --
    and K/V always come from the base stream, so the cache stays base-clean.

    Subclassing rather than writing a parallel module is deliberate: there is no second `__init__`
    that could forget a base-model submodule such as `block_sparse_moe`.
    """

    def __init__(self, config: GraniteSwitchConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        # The one place the base stream feeds the adapter stream. Its base weight is zeroed in
        # `GraniteSwitchPreTrainedModel._init_weights`, so the site contributes only the selected
        # adapter's LoRA delta and a freshly built layer injects exactly nothing.
        self.cross_stream = GraniteSwitchLoRALinear(
            config.hidden_size,
            config.hidden_size,
            config.num_adapters,
            config.cross_stream_rank,
            bias=False,
        )

    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        base_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        adapter_indices: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The adapter stream, which is the one that reaches the LM head. It comes first, and is
            returned first, so that the hidden-state recorder installed for the single-stream layer
            collects the right stream here too.
        base_hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The frozen base stream. Positional, like `hidden_states`, so that gradients reach it
            under gradient checkpointing.
        adapter_indices (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token adapter selection from the switch. `0` routes a token through the base model.
        """
        residual, base_residual = hidden_states, base_hidden_states
        normed = self.input_layernorm(hidden_states)
        base_normed = self.input_layernorm(base_hidden_states)

        # The base stream owns the single cache write. Shadow Residual attends against the base
        # stream by construction, so the keys and values that land in the cache are base-clean.
        base_attn_output, _, base_kv = self.self_attn(
            hidden_states=base_normed,
            attention_mask=attention_mask,
            adapter_indices=None,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            return_kv=True,
            **kwargs,
        )
        # The adapter stream projects its own Q, with LoRA, and reads the keys and values above.
        attn_output = self.self_attn.forward_dual_stream(
            hidden_states=normed,
            base_kv=base_kv,
            attention_mask=attention_mask,
            adapter_indices=adapter_indices,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + attn_output * self.residual_multiplier
        base_hidden_states = base_residual + base_attn_output * self.residual_multiplier

        residual, base_residual = hidden_states, base_hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        base_normed = self.post_attention_layernorm(base_hidden_states)

        # The adapter stream inherits the base stream's expert assignment *and* its gate scalars.
        # Routing on the base stream is what keeps the frozen stream bit-identical to stock.
        # CODEPATH: `routing` stays `None` when there is no expert bank, in which case `_mlp_block`
        # has nothing to route and only the shared MLP runs.
        routing = self._route(base_normed) if self.has_experts else None
        base_mlp_output = self._mlp_block(base_normed, None, routing)
        mlp_output = self._mlp_block(normed, adapter_indices, routing)

        hidden_states = residual + mlp_output * self.residual_multiplier
        base_hidden_states = base_residual + base_mlp_output * self.residual_multiplier

        hidden_states = hidden_states + self.cross_stream(base_hidden_states, adapter_indices)
        return hidden_states, base_hidden_states


@auto_docstring
class GraniteSwitchPreTrainedModel(PreTrainedModel):
    # Deliberately not a subclass of `GraniteMoeHybridPreTrainedModel`, even though every module
    # below comes from that model: the converter inlines a parent's `_init_weights` into the
    # generated file, and the hybrid's covers its Mamba layers, which would drag the whole Mamba
    # mixer and its kernels into a model that is attention-only.
    config: GraniteSwitchConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GraniteSwitchDecoderLayer", "GraniteSwitchShadowResidualDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    # The switched LoRA linears narrow their work to the adapters a batch actually selects, which
    # is a data-dependent shape, so the model cannot be captured as a single static graph.
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _is_stateful = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(GraniteSwitchTopKRouter, index=0),
        "hidden_states": GraniteSwitchDecoderLayer,
        "attentions": GraniteSwitchAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, GraniteSwitchExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, GraniteSwitchTopKRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, GraniteSwitchLoRALinear):
            init.kaiming_uniform_(module.lora_A, a=5**0.5)
            init.zeros_(module.lora_B)
        elif isinstance(module, GraniteSwitchMergedLoRALinear):
            for lora_A in module.lora_A_slices:
                init.kaiming_uniform_(lora_A, a=5**0.5)
            for lora_B in module.lora_B_slices:
                init.zeros_(lora_B)
        elif isinstance(module, MultiSwitch):
            # Both are constructed, not trained, so "initializing" them means re-deriving them from
            # the config -- the same thing `__init__` did. It has to be repeated here because a
            # model built on the meta device has no values yet, and because they are the only
            # tensors in the model that a checkpoint is allowed to omit without the result being
            # wrong (both are a pure function of the config).
            init.copy_(module.codebook, module.code_gen.precompute_codebook(dtype=torch.float32))
            if module.control_to_substitute_lut is not None:
                init.copy_(module.control_to_substitute_lut, build_control_to_substitute_lut(self.config))
        elif isinstance(module, GraniteSwitchShadowResidualDecoderLayer):
            # Children are initialized before their parent, so this overwrites the plain
            # `nn.Linear` init that the `cross_stream` base weight just received -- which is the
            # point, since a non-zero base weight would give the site a non-LoRA contribution. On a
            # checkpoint load the restored tensor is already flagged initialized and this no-ops.
            init.zeros_(module.cross_stream.base_layer.weight)


@auto_docstring
class GraniteSwitchModel(GraniteSwitchPreTrainedModel):
    def __init__(self, config: GraniteSwitchConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embedding_multiplier = config.embedding_multiplier

        # CODEPATH: a `num_adapters == 0` skin has no switch and is then plain Granite, which is
        # what the composer builds to check its own weight transfer.
        if config.num_adapters > 0:
            self.switch = MultiSwitch(config)
        else:
            self.switch = None

        # One kind of layer for the whole checkpoint: Shadow Residual adapters and plain LoRA or
        # aLoRA adapters are never composed together.
        # CODEPATH: the dual-stream class on a Shadow Residual checkpoint, the single-stream one
        # otherwise.
        layer_class = GraniteSwitchShadowResidualDecoderLayer if config.dual_stream else GraniteSwitchDecoderLayer
        self.layers = nn.ModuleList([layer_class(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = GraniteSwitchRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # CODEPATH: `None` on a base that opts out of rotary embeddings entirely with
        # `position_embedding_type=None`.
        if config.position_embedding_type == "rope":
            self.rotary_emb = GraniteSwitchRotaryEmbedding(config=config)
        else:
            self.rotary_emb = None

        self.gradient_checkpointing = False
        self.post_init()

    @auto_docstring
    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        switch_input_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        r"""
        switch_input_ids (`torch.LongTensor` of shape `(batch_size, total_sequence_length)`, *optional*):
            The sequence from position 0, used for routing only. Routing is latest-wins over the
            control tokens seen so far, so a decode step cannot be routed from its single new token
            alone; `prepare_inputs_for_generation` passes the full sequence here while `input_ids`
            stays the step's new token. Defaults to `input_ids`, which is correct whenever that is
            itself the whole sequence.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        else:
            batch_size, seq_length = inputs_embeds.shape[:2]
            device = inputs_embeds.device

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(past_seen_tokens, past_seen_tokens + seq_length, device=device).unsqueeze(0)

        # Mask creation reads the embedding tensor only for batch size, query length and dtype, and
        # nothing is embedded yet -- the switch may rewrite `input_ids` -- so a stub stands in.
        mask_shape_proxy = (
            inputs_embeds
            if inputs_embeds is not None
            else torch.empty(batch_size, seq_length, 1, device=device, dtype=self.dtype)
        )
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=mask_shape_proxy,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # Routing runs over the whole sequence and only its tail is kept, because a decode step's
        # adapter is decided by the control tokens that came before it. The length test rules out
        # an `inputs_embeds` prefill, for which generation keeps an empty `input_ids`.
        routing_input_ids = switch_input_ids if switch_input_ids is not None else input_ids
        if routing_input_ids is not None and routing_input_ids.shape[1] < seq_length:
            routing_input_ids = None

        # The switch also returns `input_ids` with every control token rewritten to its substitute
        # id, so the decoder embeds once and needs no token-exchange awareness of its own.
        # CODEPATH: `None` means base for every token, which is what an unroutable call gets.
        adapter_indices = None
        if self.switch is not None and routing_input_ids is not None:
            adapter_indices, exchanged_input_ids = self.switch(routing_input_ids)
            adapter_indices = adapter_indices[:, -seq_length:]
            if input_ids is not None:
                input_ids = exchanged_input_ids[:, -seq_length:]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds * self.embedding_multiplier

        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        # Shadow Residual runs two streams from the same embeddings; only the adapter stream
        # reaches the LM head.
        # CODEPATH: `None` outside a Shadow Residual checkpoint, which has the one stream.
        base_hidden_states = hidden_states.clone() if self.config.dual_stream else None

        for decoder_layer in self.layers:
            if base_hidden_states is None:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    adapter_indices=adapter_indices,
                    **kwargs,
                )
            else:
                hidden_states, base_hidden_states = decoder_layer(
                    hidden_states,
                    base_hidden_states,
                    attention_mask=causal_mask,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    adapter_indices=adapter_indices,
                    **kwargs,
                )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


@auto_docstring
class GraniteSwitchForCausalLM(GraniteSwitchPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}

    def __init__(self, config: GraniteSwitchConfig):
        super().__init__(config)
        self.model = GraniteSwitchModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.logits_scaling = config.logits_scaling
        self.post_init()

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        switch_input_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MoeCausalLMOutputWithPast:
        r"""
        switch_input_ids (`torch.LongTensor` of shape `(batch_size, total_sequence_length)`, *optional*):
            The sequence from position 0, used for routing only. See
            [`GraniteSwitchModel.forward`]; `prepare_inputs_for_generation` fills it in during
            generation.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GraniteSwitchForCausalLM

        >>> model = GraniteSwitchForCausalLM.from_pretrained("ibm-granite/granite-switch-example")
        >>> tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-switch-example")

        >>> # The adapter is selected by a control token in the prompt, so a plain prompt runs the
        >>> # base model and the same prompt behind `<|adapter|>` runs that adapter.
        >>> inputs = tokenizer("<|adapter|>Summarize the document.", return_tensors="pt")

        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        ```"""
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_router_logits=output_router_logits,
            switch_input_ids=switch_input_ids,
            **kwargs,
        )

        hidden_states = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
        logits = self.lm_head(hidden_states) / self.config.logits_scaling

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                # Make sure to reside on the same device.
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Hand the switch the whole sequence while the decoder gets only the new tokens.

        The base implementation slices `input_ids` down to what the cache has not seen, which is
        exactly right for the decoder and useless for routing: a decode step's single token says
        nothing about which control tokens preceded it. The untrimmed sequence is passed alongside
        it instead, and the model keeps the tail of the routing it computes.

        Generation owns `input_ids`, so this needs no state of its own and a reordered beam or a
        cropped sequence arrives already corrected.
        """
        model_inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        if self.model.switch is not None:
            model_inputs["switch_input_ids"] = input_ids
        return model_inputs


__all__ = [
    "GraniteSwitchConfig",
    "GraniteSwitchForCausalLM",
    "GraniteSwitchModel",
    "GraniteSwitchPreTrainedModel",
]
