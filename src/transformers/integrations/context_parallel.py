# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""
User-facing integration for Context Parallelism (CP).

Registers a new attention implementation, ``"context_parallel_ulysses"``,
under :data:`transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS`. When a
model is wrapped with :func:`apply_context_parallel`, every attention
submodule whose qualified name matches the model's ``_cp_plan`` has the
CP process group stashed on it as ``module._cp_group`` and the model's
``_attn_implementation`` is set to ``"context_parallel_ulysses"``.

The user-facing flow mirrors Tensor Parallelism:

    from transformers import AutoModelForCausalLM
    from transformers.distributed import DistributedConfig

    cfg = DistributedConfig(enable_context_parallel=True, cp_world_size=2)
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b", distributed_config=cfg, dtype="bfloat16",
    )
    # ... or post-load:
    from transformers.integrations.context_parallel import apply_context_parallel
    apply_context_parallel(model, cp_world_size=2)

The model's input must be pre-sharded on the sequence dimension across the
CP group (every rank sees its own ``N_total / cp_world`` chunk). Attention
layers then perform a head-axis all-to-all so the SDPA call sees the full
causal sequence; everything else is point-wise on the local shard.
"""

from __future__ import annotations

import fnmatch
import re

import torch
import torch.distributed as dist

from ..distributed.context_parallel import ulysses_attention
from ..utils import logging


logger = logging.get_logger(__name__)


def _cp_ulysses_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Attention function registered as ``"context_parallel_ulysses"``.

    Reads ``module._cp_group`` (set by :func:`apply_context_parallel`) and
    delegates to :func:`transformers.distributed.context_parallel.ulysses_attention`.

    Sinks and sliding-window are picked up from the attention module's
    own attributes (``module.sinks`` / ``module.sliding_window``) if
    present. This matches how GPT-OSS attention forwards already read
    them in eager / SDPA implementations.

    The input ``query``, ``key``, ``value`` tensors are seq-sharded along
    ``N``. Output is also seq-sharded.
    """
    if dropout != 0.0:
        # CP attention is for training-only large-context flows where the
        # full softmax is preserved bit-exactly. Random dropout would
        # de-synchronise the per-rank attention drops; if you really need
        # it, generate a shared mask on rank 0 and broadcast — out of scope
        # for v1.
        raise NotImplementedError("context_parallel_ulysses does not support attention dropout (would desync ranks).")

    cp_group: dist.ProcessGroup | None = getattr(module, "_cp_group", None)
    sinks = getattr(module, "sinks", None)
    sliding_window = getattr(module, "sliding_window", None)
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)

    if attention_mask is not None:
        raise NotImplementedError(
            "context_parallel_ulysses currently only supports the implicit causal mask "
            "(plus optional sliding window). Pass attention_mask=None and rely on is_causal."
        )

    out = ulysses_attention(
        query,
        key,
        value,
        is_causal=bool(is_causal),
        scale=scaling,
        sinks=sinks,
        sliding_window=sliding_window,
        cp_group=cp_group,
    )
    out = out.transpose(1, 2).contiguous()
    return out, None


def _register_cp_attention_impl() -> None:
    """Idempotently register the CP-Ulysses attention impl."""
    from ..modeling_utils import ALL_ATTENTION_FUNCTIONS

    if "context_parallel_ulysses" not in ALL_ATTENTION_FUNCTIONS.valid_keys():
        ALL_ATTENTION_FUNCTIONS.register(
            "context_parallel_ulysses",
            _cp_ulysses_attention_forward,
        )


def _build_cp_group(cp_world_size: int) -> dist.ProcessGroup | None:
    """Build a single CP process group spanning ``cp_world_size`` consecutive ranks.

    Assumes the default world process group has been initialised. Picks the
    world's first ``cp_world_size`` ranks. For 2-D meshes (e.g. EP × CP),
    callers should construct sub-groups explicitly and pass them to
    :func:`apply_context_parallel` via the ``cp_group`` argument.
    """
    if cp_world_size <= 1:
        return None
    if not dist.is_initialized():
        raise RuntimeError(
            "apply_context_parallel(cp_world_size>1) requires torch.distributed to be initialised (e.g. via torchrun)."
        )
    world_size = dist.get_world_size()
    if world_size % cp_world_size != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by cp_world_size ({cp_world_size}).")
    rank = dist.get_rank()
    cp_group_id = rank // cp_world_size
    ranks = list(range(cp_group_id * cp_world_size, (cp_group_id + 1) * cp_world_size))
    return dist.new_group(ranks=ranks)


def _glob_to_regex(pattern: str) -> re.Pattern:
    """Convert a `_cp_plan` glob (with ``*`` wildcards) to a compiled regex.

    Mirrors how ``_tp_plan`` keys are interpreted: ``"model.layers.*.self_attn"``
    matches any layer index.
    """
    return re.compile(fnmatch.translate(pattern))


def apply_context_parallel(
    model: torch.nn.Module,
    cp_world_size: int,
    cp_group: dist.ProcessGroup | None = None,
    cp_strategy: str = "ulysses",
) -> torch.nn.Module:
    """Enable context parallelism on every attention module declared in ``model._cp_plan``.

    Args:
        model: A `PreTrainedModel` whose class declares ``_cp_plan`` (a dict
            mapping glob patterns over module qualified names to CP strategy
            keys, e.g. ``{"model.layers.*.self_attn": "ulysses"}``).
        cp_world_size: Number of ranks across which to shard the sequence
            dimension. Must divide ``torch.distributed.get_world_size()``.
        cp_group: Optional pre-built CP-axis process group (useful for 2-D
            EP × CP meshes). If omitted, builds a fresh group covering
            ``cp_world_size`` consecutive global ranks.
        cp_strategy: Currently only ``"ulysses"`` is supported. Future work
            may add ``"ring"``.

    Returns:
        The same ``model``, mutated in place.

    Raises:
        ValueError: If ``model.__class__`` does not declare a ``_cp_plan``,
            or if ``cp_strategy`` is unknown.
    """
    _register_cp_attention_impl()

    if cp_strategy != "ulysses":
        raise ValueError(f"Unknown cp_strategy {cp_strategy!r}. Only 'ulysses' is currently supported.")

    plan = getattr(model.__class__, "_cp_plan", None)
    if not plan:
        # Walk MRO to find an inherited _cp_plan (the base PreTrainedModel
        # subclass usually owns it on the *ForCausalLM head).
        for klass in type(model).__mro__:
            plan = getattr(klass, "_cp_plan", None)
            if plan:
                break
    if not plan:
        raise ValueError(
            f"{type(model).__name__} does not declare _cp_plan; add a class-level dict "
            "such as {'model.layers.*.self_attn': 'context_parallel_ulysses'} to enable CP."
        )

    if cp_world_size > 1 and cp_group is None:
        cp_group = _build_cp_group(cp_world_size)

    compiled = [(_glob_to_regex(pat), strat) for pat, strat in plan.items()]
    n_wrapped = 0
    for name, sub in model.named_modules():
        for pat, strat in compiled:
            if pat.match(name):
                sub._cp_group = cp_group
                n_wrapped += 1
                break

    if n_wrapped == 0:
        logger.warning(
            "_cp_plan matched zero modules on %s. CP will be inactive.",
            type(model).__name__,
        )
    else:
        logger.info(
            "Context parallelism enabled: wrapped %d attention modules (cp_world_size=%d).",
            n_wrapped,
            cp_world_size,
        )

    if hasattr(model, "config"):
        model.config._attn_implementation = "context_parallel_ulysses"

    return model


__all__ = ["apply_context_parallel"]
