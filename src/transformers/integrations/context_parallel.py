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

This module mirrors the entry-point split that tensor parallelism uses:

* :func:`initialize_context_parallelism` — runs *before* the model is loaded.
  Initialises ``torch.distributed`` if needed, builds the CP-axis process
  group, and returns it. Analogous to
  :func:`transformers.integrations.tensor_parallel.initialize_tensor_parallelism`.
* :func:`distribute_context_parallel` — runs *after* the model is loaded.
  Validates the model's ``_cp_plan``, registers the CP attention
  implementation in ``ALL_ATTENTION_FUNCTIONS`` if not already done,
  walks the model and stashes the CP process group on every matched
  attention module, and flips ``config._attn_implementation`` to the
  registered CP strategy. Analogous to
  :func:`transformers.integrations.tensor_parallel.distribute_model`.

A one-call convenience wrapper :func:`apply_context_parallel` is kept for
users who want to enable CP outside the :meth:`from_pretrained` flow.
"""

from __future__ import annotations

import fnmatch
import os
import re

import torch
import torch.distributed as dist

from ..distributed.context_parallel import ulysses_attention
from ..utils import logging
from ..utils.import_utils import is_torch_greater_or_equal


logger = logging.get_logger(__name__)


# Maps cp_strategy string → AttentionInterface key registered in
# ALL_ATTENTION_FUNCTIONS. Update here when adding new strategies
# (e.g. "ring" → "context_parallel_ring").
_CP_STRATEGY_TO_ATTN_IMPL = {
    "ulysses": "context_parallel_ulysses",
}


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

    Reads ``module._cp_group`` (set by :func:`distribute_context_parallel`)
    and delegates to :func:`transformers.distributed.context_parallel.ulysses_attention`.

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


def _glob_to_regex(pattern: str) -> re.Pattern:
    """Convert a ``_cp_plan`` glob (with ``*`` wildcards) to a compiled regex.

    Mirrors how ``_tp_plan`` keys are interpreted: ``"model.layers.*.self_attn"``
    matches any layer index.
    """
    return re.compile(fnmatch.translate(pattern))


def initialize_context_parallelism(
    distributed_config=None,
    cp_size: int | None = None,
    device_mesh=None,
) -> tuple[dist.ProcessGroup | None, int]:
    """Set up ``torch.distributed`` and build the CP-axis process group.

    Mirrors :func:`initialize_tensor_parallelism` for the CP axis.
    Called *before* the model is loaded so the CP process group exists
    by the time :func:`distribute_context_parallel` is invoked.

    Args:
        distributed_config (`DistributedConfig`, *optional*):
            The user's distributed config. CP activates iff
            ``enable_context_parallel`` is ``True`` and the requested
            ``cp_world_size > 1``.
        cp_size (`int`, *optional*):
            Explicit CP world size override. Defaults to
            ``distributed_config.cp_world_size``.
        device_mesh (`torch.distributed.device_mesh.DeviceMesh`, *optional*):
            Pre-built 1-D or N-D mesh. If N-D, the ``"cp"`` axis is used.
            When omitted, a fresh sub-group is built from the default
            world process group covering ``cp_size`` consecutive ranks.

    Returns:
        Tuple of ``(cp_group, cp_size)``. ``cp_group`` is ``None`` when
        CP is disabled or the world is size-1.
    """
    if distributed_config is None or not getattr(distributed_config, "enable_context_parallel", False):
        return None, 1

    if cp_size is None:
        cp_size = getattr(distributed_config, "cp_world_size", 1)
    if cp_size <= 1:
        return None, 1

    if device_mesh is not None:
        if device_mesh.ndim > 1:
            if "cp" not in device_mesh.mesh_dim_names:
                raise ValueError(
                    "When using `enable_context_parallel` with an n-d `device_mesh`, it must contain a "
                    "'cp' dimension. Please provide a valid `device_mesh`."
                )
            cp_dim = device_mesh["cp"]
        else:
            cp_dim = device_mesh
        return cp_dim.get_group(), cp_dim.size()

    if not is_torch_greater_or_equal("2.0"):
        raise OSError("Context parallel is only supported for `torch>=2.0`.")

    if not torch.distributed.is_initialized():
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            device_type = torch._C._get_accelerator().type
            backend_map = {
                "cuda": "nccl",
                "cpu": "gloo",
                "xpu": "xccl",
                "hpu": "hccl",
            }
            torch.distributed.init_process_group(
                backend=backend_map.get(device_type, "nccl"),
                rank=rank,
                world_size=world_size,
            )
            if device_type != "cpu":
                local_rank = int(os.environ["LOCAL_RANK"])
                getattr(torch, device_type).set_device(local_rank)
        except Exception as e:
            raise OSError(
                "We tried to initialize torch.distributed for you, but it failed. Make "
                "sure you init torch distributed in your script to use `enable_context_parallel`."
            ) from e

    world = torch.distributed.get_world_size()
    if world % cp_size != 0:
        raise ValueError(f"world_size ({world}) must be divisible by cp_size ({cp_size}).")

    rank = torch.distributed.get_rank()
    cp_group_id = rank // cp_size
    ranks = list(range(cp_group_id * cp_size, (cp_group_id + 1) * cp_size))
    cp_group = torch.distributed.new_group(ranks=ranks)
    return cp_group, cp_size


def distribute_context_parallel(
    model: torch.nn.Module,
    cp_group: dist.ProcessGroup | None,
    distributed_config=None,
) -> torch.nn.Module:
    """Apply the model's ``_cp_plan`` after the model has been loaded.

    Mirrors :func:`distribute_model` for the CP axis. Walks
    ``model.named_modules()`` and for every module whose qualified name
    matches one of the globs in ``_cp_plan``, stashes ``cp_group`` on it
    as ``module._cp_group``. Validates that each plan entry maps to a
    strategy registered in ``ALL_ATTENTION_FUNCTIONS``. Finally sets
    ``model.config._attn_implementation`` to the registered CP strategy
    so subsequent forward calls go through the CP attention function.

    Args:
        model: The loaded ``PreTrainedModel`` (or other model whose
            class declares ``_cp_plan``).
        cp_group: The CP-axis process group from
            :func:`initialize_context_parallelism`. May be ``None`` for
            ``cp_size == 1`` (no-op).
        distributed_config (`DistributedConfig`, *optional*):
            Records ``cp_strategy`` for forward-compat. Defaults to
            ``"ulysses"``.

    Returns:
        The same ``model``, mutated in place.

    Raises:
        ValueError: If the model class does not declare ``_cp_plan``,
            if any strategy in the plan is not registered in
            ``ALL_ATTENTION_FUNCTIONS``, or if ``cp_strategy`` is unknown.
    """
    if cp_group is None:
        return model

    cp_strategy = getattr(distributed_config, "cp_strategy", "ulysses") if distributed_config else "ulysses"
    if cp_strategy not in _CP_STRATEGY_TO_ATTN_IMPL:
        raise ValueError(f"Unknown cp_strategy {cp_strategy!r}. Supported: {sorted(_CP_STRATEGY_TO_ATTN_IMPL)}.")

    _register_cp_attention_impl()
    from ..modeling_utils import ALL_ATTENTION_FUNCTIONS

    # Walk MRO to find _cp_plan inherited from a base class if needed.
    plan = None
    for klass in type(model).__mro__:
        plan = getattr(klass, "_cp_plan", None)
        if plan:
            break
    if not plan:
        raise ValueError(
            f"{type(model).__name__} does not declare _cp_plan; add a class-level dict "
            "such as {'model.layers.*.self_attn': 'context_parallel_ulysses'} to enable CP."
        )

    valid_keys = set(ALL_ATTENTION_FUNCTIONS.valid_keys())
    for strategy in plan.values():
        if strategy not in valid_keys:
            raise ValueError(
                f"Unknown CP strategy {strategy!r} in _cp_plan on {type(model).__name__}. "
                f"Registered attention impls: {sorted(valid_keys)}"
            )

    compiled = [(_glob_to_regex(pat), strat) for pat, strat in plan.items()]
    n_wrapped = 0
    for name, sub in model.named_modules():
        for pat, _ in compiled:
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
            "Context parallelism enabled: wrapped %d attention modules (cp_size=%d).",
            n_wrapped,
            dist.get_world_size(cp_group) if cp_group is not None else 1,
        )

    model._cp_group = cp_group
    model._cp_size = dist.get_world_size(cp_group) if cp_group is not None else 1
    if hasattr(model, "config"):
        model.config._attn_implementation = _CP_STRATEGY_TO_ATTN_IMPL[cp_strategy]
    return model


def apply_context_parallel(
    model: torch.nn.Module,
    cp_world_size: int,
    cp_group: dist.ProcessGroup | None = None,
    cp_strategy: str = "ulysses",
) -> torch.nn.Module:
    """One-call convenience wrapper around init + distribute.

    Equivalent to::

        from transformers.distributed import DistributedConfig
        cfg = DistributedConfig(
            enable_context_parallel=True,
            cp_world_size=cp_world_size,
            cp_strategy=cp_strategy,
        )
        cp_group, _ = initialize_context_parallelism(cfg)
        distribute_context_parallel(model, cp_group, cfg)

    Use this when enabling CP outside the
    :meth:`~transformers.PreTrainedModel.from_pretrained` flow (e.g. on
    a model you instantiated directly).
    """
    if cp_group is None:
        # Build a synthetic config so initialize_context_parallelism does the work.
        from ..distributed.configuration_utils import DistributedConfig

        cfg = DistributedConfig(
            enable_context_parallel=True,
            cp_world_size=cp_world_size,
            cp_strategy=cp_strategy,
        )
        cp_group, _ = initialize_context_parallelism(cfg)
    else:
        cfg = None
    return distribute_context_parallel(model, cp_group, cfg)


__all__ = [
    "apply_context_parallel",
    "distribute_context_parallel",
    "initialize_context_parallelism",
]
