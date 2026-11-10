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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..granitemoehybrid.configuration_granitemoehybrid import GraniteMoeHybridConfig


# Decoder-layer cache slots the switch reserves at the front of the model when adapters are
# present. The coded MultiSwitch engine owns two: a counting slot and a memory slot. Single
# source of truth: `num_hidden_layers` on disk is inflated by this count, and the model
# subtracts it to recover the physical decoder-layer count.
SWITCH_CACHE_LAYERS = 2

# The switch model is attention-only. The hybrid parent fills an unset `layer_types` with
# all-`"linear_attention"` (mamba), which would size the per-layer cache wrongly, so the value
# is pinned here instead. It must be a key of the dynamic-cache layer-type mapping.
ATTENTION_LAYER_TYPE = "full_attention"


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


__all__ = ["GraniteSwitchConfig"]
