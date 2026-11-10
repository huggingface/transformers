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

from typing import Any

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...utils import auto_docstring
from ..granitemoehybrid.configuration_granitemoehybrid import GraniteMoeHybridConfig
from .switch_codes import KerdockDGCodeGenerator, recover_count_from_signal


# Decoder-layer cache slots the switch reserves at the front of the model when adapters are
# present. The coded MultiSwitch engine owns two: a counting slot and a memory slot. Single
# source of truth: `num_hidden_layers` on disk is inflated by this count, and the model
# subtracts it to recover the physical decoder-layer count.
SWITCH_CACHE_LAYERS = 2

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
    fused_add_norm (`bool`, *optional*, defaults to `False`):
        Whether the decoder applies the residual add and the following norm as one fused operation.
        This fixes the floating-point reduction order the checkpoint was built against.
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

    fused_add_norm: bool = False

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

        A checkpoint built for the earlier SingleSwitch engine cannot run as MultiSwitch:
        MultiSwitch reserves `SWITCH_CACHE_LAYERS == 2` cache slots where SingleSwitch reserved 1,
        so a single-sized checkpoint has one decoder layer too few and its weights cannot map. There
        is no auto-migration; it must be rebuilt.

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

        if self.num_hidden_layers <= SWITCH_CACHE_LAYERS:
            raise ValueError(
                f"num_hidden_layers ({self.num_hidden_layers}) must exceed SWITCH_CACHE_LAYERS "
                f"({SWITCH_CACHE_LAYERS}) when adapters are present: the switch reserves that many cache "
                "slots at the front of the model, and the remainder are the physical decoder layers."
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
    """
    if lut is None:
        return input_ids
    substitute_per_position = lut[input_ids]
    is_control = substitute_per_position >= 0
    return torch.where(is_control, substitute_per_position, input_ids)


class GraniteSwitchCountingAttention(nn.Module):
    """Attribute carrier for the counting head's `sdpa_attention_forward` call.

    Holds no parameters and defines no `forward`. The attention interface reads its geometry off
    the module it is handed (`num_key_value_groups`, `is_causal`, `layer_idx`), so the head needs
    an object with those attributes; `MultiSwitch` builds the Q/K/V tensors itself.

    One query head over one KV head, at `counting_head_dim`, which is independent of the
    backbone's `head_dim`. Only component 0 of each vector carries signal; the rest is zero
    padding to reach the minimum width the attention kernels accept.
    """

    def __init__(self, head_dim: int, layer_idx: int):
        super().__init__()
        self.num_heads = 1
        self.num_key_value_heads = 1
        self.num_key_value_groups = 1
        self.head_dim = head_dim
        self.scaling = 1.0
        self.attention_dropout = 0.0
        self.is_causal = True
        self.layer_idx = layer_idx


class GraniteSwitchMemoryAttention(nn.Module):
    """Attribute carrier for the memory head's `sdpa_attention_forward` call.

    Same role as `GraniteSwitchCountingAttention`, at `memory_head_dim`, which must be wide enough
    to hold a whole codeword.
    """

    def __init__(self, head_dim: int, layer_idx: int):
        super().__init__()
        self.num_heads = 1
        self.num_key_value_heads = 1
        self.num_key_value_groups = 1
        self.head_dim = head_dim
        self.scaling = 1.0
        self.attention_dropout = 0.0
        self.is_causal = True
        self.layer_idx = layer_idx


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

    def __init__(self, config: GraniteSwitchConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.num_adapters = config.num_adapters
        self.control_token_gain = config.control_token_gain
        self.layer_idx = layer_idx
        self.counting_layer_idx = layer_idx
        self.memory_layer_idx = layer_idx + 1

        # Two accepted `adapter_token_ids` layouts, distinguished by length alone:
        #   * `num_adapters` entries -- entry i fires adapter i + 1, so the id is argmax + 1.
        #   * `num_adapters + 1` entries -- entry 0 is a base-reset token that fires 0 and the
        #     rest fire 1.., so the id is the argmax unshifted. This is what lets a request
        #     transition back to base, which matters when one sequence holds several control
        #     tokens: ordinary chat emits a control token for the current turn only, so earlier
        #     turns carry none and already route to base.
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

        # Must be persistent. `from_pretrained` materializes modules on the meta device and then
        # fills only the tensors the checkpoint carries, so a non-persistent buffer is absent from
        # the state dict and comes back all zeros -- whatever `__init__` computed is discarded. A
        # zeroed codebook makes every memory key and query zero, which collapses the retrieval
        # logits to a uniform softmax: each position then averages the adapter ids in scope
        # instead of selecting the most recent one, and routes to a blend with no error raised.
        # Keeping it in the state dict also turns an incompatible checkpoint into a loud missing
        # key. The cost is ~512 KB against a multi-GB checkpoint.
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

        self.counting_attn_module = GraniteSwitchCountingAttention(
            head_dim=self.counting_head_dim,
            layer_idx=self.counting_layer_idx,
        )
        self.memory_attn_module = GraniteSwitchMemoryAttention(
            head_dim=self.memory_head_dim,
            layer_idx=self.memory_layer_idx,
        )

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

    @property
    def num_cache_layers(self) -> int:
        return SWITCH_CACHE_LAYERS

    def _causal_mask(self, q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
        """Build the heads' own fp32 causal mask, spanning the key length.

        Deliberately not the model's `attention_mask`. That one is created in the model dtype,
        and handing a bf16 mask to these fp32 attention calls makes the whole op run in bf16,
        whose resolution corrupts the counting head's `1/(1+n)` recovery: the recovered index
        comes out off by one and routing lags a position. Which keys actually count is already
        encoded in the fp32 key masking, so this only has to be causal and fp32.

        The span is `kv_len`, not `q_len`. With a cache the heads attend over the whole history,
        so a `q_len x q_len` mask is the wrong shape -- a decode step of one query over six keys
        gets a (1,1,1,1) bias and SDPA rejects it. The cached prefix is entirely in the past, so
        the causal offset shifts by `kv_len - q_len`; at prefill that is 0 and this reduces to a
        plain causal mask.
        """
        return torch.triu(
            torch.full((q_len, kv_len), torch.finfo(torch.float32).min, device=device, dtype=torch.float32),
            diagonal=1 + (kv_len - q_len),
        ).view(1, 1, q_len, kv_len)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select an adapter per token, then rewrite the control tokens.

        Selection reads the original `input_ids`; the rewrite happens last, so the returned ids
        are what the decoder embeds and the control tokens never reach it.

        Returns `(adapter_indices, modified_input_ids)`, both `[batch, seq_len]`, where index 0
        means base.
        """
        bsz, q_len = input_ids.shape
        device = input_ids.device

        # Position 0 anchors the counting signal, taking the place of a dedicated init token.
        # The switch owns its two cache slots, so their length is the number of positions already
        # seen and is what places the anchor during generation.
        past_len = 0 if past_key_values is None else past_key_values.get_seq_length(self.counting_layer_idx)
        positions = torch.arange(past_len, past_len + q_len, device=device)
        is_counting_anchor = (positions == 0).unsqueeze(0).expand(bsz, -1)

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

        if past_key_values is not None:
            key_states_count, value_states_count = past_key_values.update(
                key_states_count, value_states_count, self.counting_layer_idx
            )

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
                self._causal_mask(q_len, key_states_count.shape[2], device),
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

        if past_key_values is not None:
            key_states_memory, value_states_memory = past_key_values.update(
                key_states_memory, value_states_memory, self.memory_layer_idx
            )

        query_states_memory = torch.zeros((bsz, 1, q_len, self.memory_head_dim), device=device, dtype=torch.float32)
        query_states_memory[:, 0, :, : self.memory_dim] = code_vectors

        with torch.autocast(device_type=autocast_device, enabled=False):
            memory_output, _ = attention_interface(
                self.memory_attn_module,
                query_states_memory,
                key_states_memory,
                value_states_memory,
                # The memory slot is cached separately from the counting slot and was updated
                # just above, so its key length is tracked independently.
                self._causal_mask(q_len, key_states_memory.shape[2], device),
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


__all__ = ["GraniteSwitchConfig"]
