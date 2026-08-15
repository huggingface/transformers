# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
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

"""Shared export utilities used by all exporter backends.

Organised into five sections (search for the `# ── Name ──` banners):

- **Patch and fix registries** — backend-keyed `_PATCHES` / `_FX_NODE_FIXES` /
  `_FX_PROGRAM_FIXES` populated via `@register_patch(backend, *paths)` /
  `@register_fx_node_fix` / `@register_fx_program_fix`, applied via
  `apply_patches` / `apply_fx_node_fixes` / `apply_fx_program_fixes`.
- **Recursive structure traversal** — internal helpers (`_map_leaf_tensors`,
  `_iter_leaf_tensors`) that drive every other tensor utility.
- **Public tensor utilities** — `get_leaf_tensors`, `duplicate_leaf_tensors`,
  `cast_leaf_tensors`, and `prepare_for_export` (sets attention/experts impl,
  patches non-exportable patterns, strips output flags).
- **Export input preparers** — `@register_export_input_preparer(marker)`
  registry that precomputes the per-encoder kwargs (`cu_seqlens`, `position_ids`,
  audio chunks, …) the model would otherwise need data-dependent ops for.
- **Decomposition** — `decompose_prefill_decode` (split a generative forward
  into prefill + decode) and `decompose_multimodal` + `is_multimodal` (split a
  multimodal forward into one entry per submodule), backed by `_capture_forward`.
"""

from __future__ import annotations

import contextlib
import copy
import enum
import functools
import importlib
import inspect
from collections.abc import Mapping, MutableMapping
from typing import Any

from ..utils import logging
from ..utils.generic import get_max_seqlen
from ..utils.import_utils import is_torch_available


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

    from ..configuration_utils import PreTrainedConfig
    from ..modeling_outputs import BaseModelOutput
    from ..modeling_utils import PreTrainedModel
    from ..vision_utils import (
        get_vision_attention_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_merged_shape,
        get_vision_nearest_position_ids,
        get_vision_position_ids,
        get_vision_window_index,
    )


# ── Patch and fix registries ────────────────────────────────────────────────
# Single contract across exporters: `_PATCHES[backend]` lists `(obj, attribute, factory)` triples
# to install reversibly, and `_FX_NODE_FIXES[backend]` lists `(gm, node) -> bool` fixers to
# apply in place. Each exporter populates its slot at module load (via `@register_patch` /
# `@register_fx_node_fix` decorators, or direct list-append for cases that can't be expressed
# as dotted paths). The export pipeline drives them via the backend-keyed helpers below.

_PATCHES: dict[str, list[tuple[Any, str, callable]]] = {}
_FX_PROGRAM_FIXES: dict[str, list[callable]] = {}
_FX_NODE_FIXES: dict[str, list[callable]] = {}


@contextlib.contextmanager
def patch_attribute(obj: Any, attribute: str, factory: Any):
    """Swap `obj.<attribute>` with `factory(original)` for the duration of the block."""
    original = getattr(obj, attribute)
    setattr(obj, attribute, factory(original))
    try:
        yield
    finally:
        setattr(obj, attribute, original)


@contextlib.contextmanager
def patch_attributes(patches: list[tuple[Any, str, callable]]):
    """Install `(obj, attribute, factory)` patches for the duration of the block.

    Plural form of `patch_attribute` — each `factory(original)` returns the replacement
    callable. Originals are restored on exit, even if the body raises.
    """
    with contextlib.ExitStack() as stack:
        for obj, attribute, factory in patches:
            stack.enter_context(patch_attribute(obj, attribute, factory))
        yield


@contextlib.contextmanager
def apply_patches(backend: str):
    """Install `_PATCHES[backend]` for the duration of the block."""
    with patch_attributes(_PATCHES.get(backend, [])):
        yield


def register_fx_node_fix(backend: str):
    """Append the decorated `(gm, node) -> bool` fix to `_FX_NODE_FIXES[backend]`."""

    def decorator(fn):
        _FX_NODE_FIXES.setdefault(backend, []).append(fn)
        return fn

    return decorator


def register_fx_program_fix(backend: str):
    """Append the decorated `(exported_program) -> None` fix to `_FX_PROGRAM_FIXES[backend]`.

    Use this for fixes that need program-level context (range_constraints, graph_signature,
    state_dict) — the per-node `_FX_NODE_FIXES` shape only sees one node at a time.
    """

    def decorator(fn):
        _FX_PROGRAM_FIXES.setdefault(backend, []).append(fn)
        return fn

    return decorator


def apply_fx_program_fixes(backend: str, exported_program) -> None:
    """Apply `_FX_PROGRAM_FIXES[backend]` to `exported_program` (in place)."""
    for fix in _FX_PROGRAM_FIXES.get(backend, []):
        fix(exported_program)


def register_patch(backend: str, *paths: str):
    """Append the decorated `factory(original)` to `_PATCHES[backend]`, once per `path`.

    Each `path` is a dotted Python path like `"torch.where"`, `"torch.Tensor.unsqueeze"`,
    or `"transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeTop2Router._cast_classifier"`.
    The rightmost segment is the attribute to swap; the rest is the object that owns it.
    Paths are resolved at decoration time — submodules are imported as needed, falling
    back to `getattr` for class attributes. A path that fails to resolve (e.g. the backend
    isn't installed) is silently skipped so the module still imports.

    Passing multiple paths registers the SAME factory against each — useful for swapping
    the same method or torch op across several call sites (e.g. ``torch.unsqueeze`` +
    ``torch.Tensor.unsqueeze``, or one vision-attention forward across N model classes).
    """

    def decorator(fn):
        for path in paths:
            obj_path, _, attribute = path.rpartition(".")
            obj = _resolve_dotted_path(obj_path)
            if obj is None:
                continue
            _PATCHES.setdefault(backend, []).append((obj, attribute, fn))
        return fn

    return decorator


def _resolve_dotted_path(path: str):
    """Resolve a dotted Python path to the actual object — importing submodules where
    possible, falling back to `getattr` for class attributes (e.g. `torch.Tensor`).
    Returns `None` if the path can't be resolved (e.g. the backend isn't installed)."""
    parts = path.split(".")
    try:
        obj = importlib.import_module(parts[0])
        for part in parts[1:]:
            try:
                obj = importlib.import_module(f"{obj.__name__}.{part}")
            except (ImportError, AttributeError):
                obj = getattr(obj, part)
        return obj
    except (ImportError, AttributeError):
        return None


def apply_fx_node_fixes(backend: str, graph_module) -> None:
    """Walk every call_function node and apply the first matching `_FX_NODE_FIXES[backend]`
    fix, then DCE.

    Each fix has signature `(gm, node) -> bool`. Returning `True` means the fix consumed
    the node — no further fixes run against it. Fixes are expected to be disjoint by
    `node.target`; if multiple could apply, list order decides.

    After the walk, `Graph.eliminate_dead_code` runs on every sub-GraphModule and
    `gm.recompile()` is called once. PyTorch DCE occasionally raises `SystemError` /
    `KeyError` from `erase_node._update_args_kwargs` on orphaned symbolic-size nodes —
    we swallow both; any survivors are handled by the downstream backend optimizer.
    """
    fixes = _FX_NODE_FIXES.get(backend, [])
    for gm in graph_module.modules():
        if not isinstance(gm, torch.fx.GraphModule):
            continue
        for node in list(gm.graph.nodes):
            if node.op != "call_function":
                continue
            for fix in fixes:
                if fix(gm, node):
                    break
        try:
            gm.graph.eliminate_dead_code()
            gm.recompile()
        except (SystemError, KeyError):
            pass


# ── Recursive structure traversal ──────────────────────────────────────────
# All tensor utilities share this traversal. _map_leaf_tensors applies a function
# to every tensor leaf; _iter_leaf_tensors yields (path, tensor) pairs.

# Types that should not be recursed into when extracting leaf tensors. Sym* types
# carry PyTorch shape_env internals that cause infinite recursion; Enums are scalars
# with no tensor fields.
_LEAF_SKIP_TYPES: tuple[type, ...] = (type,)
if is_torch_available():
    _LEAF_SKIP_TYPES += (enum.Enum, torch.SymInt, torch.SymFloat, torch.SymBool)


def _map_leaf_tensors(obj: Any, fn: callable) -> Any:
    """Apply `fn` to every tensor in a nested structure, preserving container types.

    Mutates dicts and `__dict__`-bearing objects in place (preserving identity — callers
    rely on this so downstream pops/mutations propagate back to the original mapping);
    rebuilds lists/tuples/sets/frozensets (immutable or order-sensitive containers).
    Skips non-traversable leaf types (enum, SymInt, etc.).
    """
    if isinstance(obj, _LEAF_SKIP_TYPES):
        return obj
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(_map_leaf_tensors(item, fn) for item in obj)
    if isinstance(obj, dict):
        for k in list(obj):
            obj[k] = _map_leaf_tensors(obj[k], fn)
        return obj
    if hasattr(obj, "__dict__"):
        for attr, attr_val in vars(obj).items():
            setattr(obj, attr, _map_leaf_tensors(attr_val, fn))
    return obj


def _iter_leaf_tensors(obj: Any, prefix: str = ""):
    """Yield `(dotted_path, tensor)` for every tensor in a nested structure."""
    if isinstance(obj, _LEAF_SKIP_TYPES):
        return
    if isinstance(obj, torch.Tensor):
        yield prefix or "output", obj
    elif isinstance(obj, (list, tuple, set)):
        for index, item in enumerate(obj):
            path = f"{prefix}.{index}" if prefix else str(index)
            yield from _iter_leaf_tensors(item, path)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            yield from _iter_leaf_tensors(value, path)
    elif hasattr(obj, "__dict__"):
        yield from _iter_leaf_tensors(vars(obj), prefix)


# ── Public tensor utilities ────────────────────────────────────────────────
# Extract or cast tensors from nested model outputs.


def get_leaf_tensors(obj: Any) -> dict[str, torch.Tensor]:
    """Recursively retrieve all leaf tensors from a potentially nested structure.

    Args:
        obj (`Any`):
            A tensor, dataclass, dict, list, tuple, or any nesting thereof.

    Returns:
        `dict[str, torch.Tensor]`: Flat mapping from dotted path strings to tensors.
    """
    return dict(_iter_leaf_tensors(obj))


def duplicate_leaf_tensors(obj: Any) -> Any:
    """Clone tensors that appear more than once in an output structure.

    When a model returns the same tensor under two output names (e.g. `last_hidden_state`
    and `hidden_states[0]`), the ONNX optimizer deduplicates the two output nodes and
    renames one, breaking the expected name mapping. Cloning duplicates gives each output
    leaf a distinct identity so the optimizer has nothing to merge.
    """
    seen = set()

    def _dedup(tensor: torch.Tensor) -> torch.Tensor:
        if id(tensor) in seen:
            return tensor.clone()
        seen.add(id(tensor))
        return tensor

    return _map_leaf_tensors(obj, _dedup)


def cast_leaf_tensors(obj: Any, dtype: torch.dtype, device: torch.device) -> Any:
    """Recursively cast all floating-point tensors to the given dtype and device."""

    def _cast(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype=dtype if tensor.is_floating_point() else None, device=device)

    return _map_leaf_tensors(obj, _cast)


def module_device(model: PreTrainedModel | torch.nn.Module) -> torch.device | None:
    """`.device` for any `nn.Module`. `PreTrainedModel` exposes it directly via `ModuleUtilsMixin`;
    for plain submodules (e.g. a `Linear` or `MultiModalProjector` from a decomposed multimodal model)
    we fall back to the first parameter. Returns `None` if the module has no parameters at all."""
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def module_dtype(model: PreTrainedModel | torch.nn.Module) -> torch.dtype | None:
    """`.dtype` for any `nn.Module`. Same fallback story as `module_device`."""
    if hasattr(model, "dtype"):
        return model.dtype
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return None


# Output flags that should be set on `model.config`, not passed as forward() kwargs.
_OUTPUT_FLAGS = ("use_cache", "output_attentions", "output_hidden_states", "return_dict", "return_loss")


def prepare_for_export(
    model: PreTrainedModel | torch.nn.Module, inputs: MutableMapping[str, Any]
) -> tuple[PreTrainedModel | torch.nn.Module, MutableMapping[str, Any], dict[str, Any]]:
    """Configure model and inputs for export. Mutates both `model` and `inputs` in place,
    returning `(model, inputs, output_flags)` where `output_flags` holds the values popped
    from `inputs` for `use_cache`, `return_dict`, etc. (to be applied reversibly onto
    `model.config` by `patch_model_config` during the trace).

    - Strips label inputs (`labels`, `future_values`) — loss computation is unsupported.
    - Pops output flags (`use_cache`, `return_dict`, …) from `inputs` so they don't appear
      as traced kwargs; the values are returned for the trace block to apply onto
      `model.config`.
    - Pre-computes data-dependent vision/audio kwargs registered via
      `@register_export_input_preparer` and writes them into `inputs`.
    - Casts input tensors to match the model's `dtype` / `device`.
    """
    # Strip label inputs — loss computation is not supported during export.
    for label_key in ("labels", "future_values"):
        value = inputs.pop(label_key, None)
        if value is not None:
            raise ValueError(
                f"Found '{label_key}' in inputs. Loss computation is not supported during export. "
                f"Please remove '{label_key}' from your inputs before calling export()."
            )
    if hasattr(model, "config") and getattr(model.config, "return_loss", False):
        raise ValueError(
            "Found 'model.config.return_loss=True'. Loss computation is not supported during export. "
            "Please set 'model.config.return_loss=False' before calling export()."
        )
    if inputs.get("return_loss", False):
        raise ValueError(
            "Found 'return_loss=True' in inputs. Loss computation is not supported during export. "
            "Please remove 'return_loss' from your inputs or set it to False."
        )

    # Pop output flags from `inputs` and return them so the caller can decide how to
    # honour them during the trace (we don't want them as traced kwargs).
    output_flags = {flag: inputs.pop(flag) for flag in _OUTPUT_FLAGS if flag in inputs}

    # Pre-compute data-dependent vision/audio tensors that use loops, .tolist(),
    # repeat_interleave, or itertools.groupby — untraceable by dynamo.
    # TODO: use the collator API once it covers these cases.
    with torch.no_grad():
        if (config := getattr(model, "config", None)) is not None:
            inputs.update(precompute_export_inputs(config, inputs))

    # Move input tensors onto the model's device (e.g. a cache built on CPU before a backend moved the
    # model). Dtypes are left as-is on purpose: inputs already carry the caller's/model's dtype, and cache
    # entries keep the dtype the model allocated them at — notably SSM/recurrent states the mixer holds in
    # fp32 for scan stability even in a bf16 model — which a blanket downcast would corrupt, making an
    # exported decode step diverge from eager.
    device = module_device(model)
    if device is not None:
        inputs = cast_leaf_tensors(inputs, dtype=None, device=device)

    return model, inputs, output_flags


# ── Export input preparers ────────────────────────────────────────────────────
# Registry of `model_type -> (model, inputs) -> None` callables that precompute the
# data-dependent tensors (cu_seqlens, position_ids, padded audio chunks, …) the model
# would otherwise compute in its forward via `.tolist()` / `nonzero()` / etc. Inject
# the results into `inputs` so the forward skips the untraceable branch.


def _find_config_attr(config: Any, name: str) -> Any | None:
    """First non-`None` `name` on `config` or any of its (recursive) `sub_configs` (`vision_config` /
    `audio_config` / `text_config` / …).

    This is how the preparers below read every parameter they need, which is what lets the precompute run
    from a saved config with no model instance: a plain field, or a `@property` where the vision module
    derives the value (`num_grid_per_side`, muse_glimmer's `window_size`). A model whose module hardcodes a
    value a preparer needs should expose it on its config the same way."""
    value = getattr(config, name, None)
    if value is not None:
        return value
    for sub_key in getattr(config, "sub_configs", {}):
        sub = getattr(config, sub_key, None)
        if sub is not None and (value := _find_config_attr(sub, name)) is not None:
            return value
    return None


def _resolve_modeling_module(config: Any):
    """The model's `modeling_*` module, from its config's module (`configuration_x` → `modeling_x`) — the
    model-free counterpart of `sys.modules[type(model).__module__]`, used to reach a model's own precompute
    helpers (`get_vision_frame_index`, `chunk_and_pad_features`, …)."""
    return importlib.import_module(type(config).__module__.replace(".configuration_", ".modeling_"))


# Marker kwarg tuples -> preparer. A preparer runs when every marker in its key is present in the inputs
# (`@register_export_input_preparer(*markers)`), so a model gets exactly the precompute its encoder needs.
_EXPORT_INPUT_PREPARERS: dict[tuple[str, ...], callable] = {}


def register_export_input_preparer(*markers: str):
    """Register `fn(config, inputs) -> None`. Dispatched when every `marker` is a key in
    `inputs` with a non-`None` value — no model_type list to maintain. The preparer reads what it needs
    from `config` (via `_precompute_attr` / `_resolve_modeling_module`), never a live model. Use multiple
    markers to narrow the match when a single kwarg is too ambiguous (e.g.
    `("input_features", "feature_lens")` for omni audio encoders)."""

    def decorator(fn):
        _EXPORT_INPUT_PREPARERS[markers] = fn
        return fn

    return decorator


@register_export_input_preparer("grid_thw")
def _prepare_grid_thw_vision_inputs(config: Any, inputs: dict[str, Any]) -> None:
    """Precompute helpers driven by `grid_thw`: `cu_seqlens`, `max_seqlen`, `position_ids`, plus optional
    `window_index`/`cu_window_seqlens`/`max_window_seqlen` (XNet-style window attn) and
    `bilinear_indices`/`bilinear_weights` (interpolation-based merging).

    Optional helpers are gated by a config attribute (`window_size`+`patch_size` for window attention,
    `num_grid_per_side` for interpolation — see `_find_config_attr`) or, for
    model-specific ones, by the encoder's modeling module defining the helper (`get_vision_frame_index` /
    `get_vision_temporal_merge_index` for kimi_k25) — so a model that doesn't use a feature won't get its
    kwarg injected.
    """
    grid_thw = inputs["grid_thw"]
    spatial_merge_size = _find_config_attr(config, "spatial_merge_size")
    if spatial_merge_size is None:
        # Video-Llama-3 carries per-image merge sizes as an input tensor rather than on its config.
        spatial_merge_size = inputs.get("merge_sizes", 1)
    # An encoder that resamples its position grid before merging (kimi_k25, muse_glimmer, paddleocr_vl)
    # builds these tensors at patch resolution — the same value its module passes.
    resample_merge_size = 1 if _find_config_attr(config, "resample_before_merge") is True else spatial_merge_size

    # Whether packed attention spans a whole clip (kimi_k25) or one segment per frame.
    module = _resolve_modeling_module(config)
    merge_temporal = _find_config_attr(config, "merge_temporal_attention") is True
    inputs["cu_seqlens"], inputs["max_seqlen"] = get_vision_attention_seqlens(
        grid_thw, config, merge_temporal=merge_temporal, kwargs=inputs
    )
    # 3-axis (t, h, w) rotary encoders expose an ``axis_dim`` on their rotary_emb (minimax_m3_vl); default
    # 2-axis (h, w) covers qwen2_5_vl / qwen3_vl / glm4v / paddleocr_vl.
    include_temporal = _find_config_attr(config, "include_temporal_position_ids") is True
    inputs["position_ids"] = get_vision_position_ids(grid_thw, resample_merge_size, include_temporal=include_temporal)

    window_size = _find_config_attr(config, "window_size")
    patch_size = _find_config_attr(config, "patch_size")
    if window_size is not None and patch_size is not None:
        inputs["window_index"], inputs["cu_window_seqlens"] = get_vision_window_index(
            grid_thw, spatial_merge_size, window_size, patch_size
        )
        inputs["max_window_seqlen"] = get_max_seqlen(
            inputs["cu_window_seqlens"], config, kwargs=inputs, kwarg_name="max_window_seqlen"
        )

    num_grid_per_side = _find_config_attr(config, "num_grid_per_side")
    if num_grid_per_side is not None:
        # How the vision embedding resamples its learned grid (kimi_k25 bicubic, qwen3_vl / paddleocr_vl
        # bilinear with aligned corners, muse_glimmer grid_sample zeros padding) — each declared on the
        # vision config; the defaults here are what a config that says nothing means.
        mode = _find_config_attr(config, "interpolation_mode") or "bilinear"
        padding = _find_config_attr(config, "interpolation_padding") or "border"
        align_corners = _find_config_attr(config, "interpolation_align_corners") is True
        inputs["interp_indices"], inputs["interp_weights"] = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side,
            mode=mode,
            align_corners=align_corners,
            spatial_merge_size=resample_merge_size,
            padding=padding,
        )

    # Per-frame additive position table (kimi_k25): gathered by frame index instead of a per-clip loop.
    if hasattr(module, "get_vision_frame_index"):
        inputs["frame_index"] = module.get_vision_frame_index(grid_thw)

    # Temporal-pooling spatial merger (kimi_k25): one gather index replaces its per-clip merge loop.
    if hasattr(module, "get_vision_temporal_merge_index"):
        merge_kernel_size = _find_config_attr(config, "merge_kernel_size")
        kernel_height, kernel_width = (
            merge_kernel_size if not isinstance(merge_kernel_size, int) else (merge_kernel_size, merge_kernel_size)
        )
        inputs["temporal_merge_index"] = module.get_vision_temporal_merge_index(grid_thw, kernel_height, kernel_width)

    # Pixel-shuffle spatial merger (muse_glimmer): one gather index replaces its per-image merge loop.
    if hasattr(module, "get_vision_pixel_shuffle_index"):
        merge_size = _find_config_attr(config, "merge_size")
        inputs["pixel_shuffle_index"] = module.get_vision_pixel_shuffle_index(grid_thw, merge_size)


@register_export_input_preparer("target_sizes")
def _prepare_navit_vision_inputs(config: Any, inputs: dict[str, Any]) -> None:
    """NaViT-style packed encoders carry per-image `(h, w)` as `target_sizes` instead of `grid_thw`.
    Synthesise `grid_thw = [1, h, w]` and run the nearest-position-id / window-index /
    merged-shape / maximum-sequence-length helpers outside the traced graph."""
    target_sizes = inputs["target_sizes"]
    num_patches_per_side = _find_config_attr(config, "num_patches_per_side")
    if num_patches_per_side is not None:
        inputs["position_ids"] = get_vision_nearest_position_ids(target_sizes, num_patches_per_side)

    window_kernel_size = _find_config_attr(config, "window_kernel_size")
    if window_kernel_size is not None:
        grid_thw = torch.nn.functional.pad(target_sizes, (1, 0), value=1)
        inputs["window_index"], inputs["cu_window_seqlens"] = get_vision_window_index(
            grid_thw, spatial_merge_size=1, window_size=window_kernel_size[0], patch_size=1
        )
        inputs["merged_shape"] = get_vision_merged_shape(target_sizes, window_kernel_size)
        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(target_sizes[:, 0] * target_sizes[:, 1], dim=0, dtype=torch.int32), (1, 0)
        )
        inputs["max_seqlen"] = get_max_seqlen(cu_seqlens, config, kwargs=inputs)


@register_export_input_preparer("input_features", "feature_lens")
def _prepare_omni_audio_inputs(config: Any, inputs: dict[str, Any]) -> None:
    """Replace `input_features`/`feature_lens` with precomputed `padded_feature`, `chunk_lengths`,
    `cu_seqlens`, `max_seqlen`, `valid_indices` (+ `pool_indices` on Qwen2.5-Omni-style encoders) so the
    encoder's `.split(.tolist(), dim=0)` and related data-dependent ops happen outside the
    traced graph.

    The helpers (`chunk_and_pad_features`, `get_audio_cu_seqlens`, …) all live in the model's
    own ``modeling_*.py`` module, resolved from `config`. ``n_window_infer`` selects the Qwen3-Omni-style
    four-arg ``get_audio_cu_seqlens`` over the Qwen2.5-Omni-style single-arg form.
    """
    feature_lens = inputs["feature_lens"]
    input_features = inputs["input_features"]
    module = _resolve_modeling_module(config)
    n_window = _find_config_attr(config, "n_window")
    n_window_infer = _find_config_attr(config, "n_window_infer")

    chunk_and_pad_features = getattr(module, "chunk_and_pad_features")
    get_audio_cu_seqlens = getattr(module, "get_audio_cu_seqlens")
    get_valid_indices = getattr(module, "get_valid_indices")

    padded_feature, chunk_lengths = chunk_and_pad_features(input_features, feature_lens, n_window)
    inputs["padded_feature"] = padded_feature
    inputs["chunk_lengths"] = chunk_lengths
    if n_window_infer is not None:
        inputs["cu_seqlens"] = get_audio_cu_seqlens(chunk_lengths, feature_lens, n_window_infer, n_window)
        inputs["valid_indices"] = get_valid_indices(chunk_lengths, n_window)
    else:
        inputs["cu_seqlens"] = get_audio_cu_seqlens(chunk_lengths)
        inputs["valid_indices"] = get_valid_indices(chunk_lengths)
        inputs["pool_indices"] = getattr(module, "get_pool_indices")(feature_lens)
    inputs["max_seqlen"] = get_max_seqlen(inputs["cu_seqlens"], config, kwargs=inputs)


@register_export_input_preparer("input_features", "input_features_mask")
def _prepare_qwen3_asr_audio_inputs(config: Any, inputs: dict[str, Any]) -> None:
    """Precompute `cu_seqlens` and `max_seqlen` for Qwen3-ASR so the encoder pops them from
    ``kwargs``. Mirrors the few lines that build ``feature_lens``/``chunk_lengths`` in
    ``Qwen3ASREncoder.forward``.
    """
    from ..models.qwen3_asr.modeling_qwen3_asr import get_audio_cu_seqlens

    n_window = _find_config_attr(config, "n_window")
    n_window_infer = _find_config_attr(config, "n_window_infer")
    if n_window is None or n_window_infer is None:
        return

    input_features_mask = inputs["input_features_mask"]
    batch_size, padded_feature_length = input_features_mask.shape
    num_chunks = padded_feature_length // (n_window * 2)
    feature_lens = input_features_mask.sum(-1).to(torch.long)
    chunk_lengths = input_features_mask.view(batch_size, num_chunks, -1).sum(dim=-1).reshape(-1).to(torch.long)
    inputs["cu_seqlens"] = get_audio_cu_seqlens(chunk_lengths, feature_lens, n_window_infer, n_window)
    inputs["max_seqlen"] = get_max_seqlen(inputs["cu_seqlens"], config, kwargs=inputs)


def precompute_export_inputs(config: PreTrainedConfig, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Return `inputs` plus the tensors a model would otherwise compute data-dependently while tracing.

    Driven entirely by the config — no model and no weights — so the same call serves the export path and
    the runtime, which only ever has the saved config. `inputs` is not modified; the precomputed tensors
    come back in a new dict.

    Two layers:
    - Outer LLM M-RoPE positions, via [`~modeling_rope_utils.get_mrope_index`] — the same layout the
      model's own `get_rope_index` runs, read off `config.mrope_layout`.
    - Per-encoder preparer dispatched by marker kwargs present in `inputs` (e.g. `grid_thw`,
      `target_sizes`, `(input_features, feature_lens)`) — see `register_export_input_preparer`.
      A preparer fires only when every one of its markers is present in `inputs`.
    """
    from ..modeling_multimodal_utils import get_mrope_index

    inputs = dict(inputs)

    # Outer-model M-RoPE positions, for a config that declares a layout. The layout reads the token ids to
    # place modality spans, so this must not run on encoder-only components (an exported
    # `get_image_features`) that carry no `input_ids`, nor on a text config — which carries `mrope_section`
    # but not the layout, since laying out the spans is the multimodal model's job.
    declares_mrope_layout = getattr(config, "mrope_layout", None) is not None
    if inputs.get("position_ids") is None and inputs.get("input_ids") is not None and declares_mrope_layout:
        input_ids = inputs["input_ids"]
        attn_mask = inputs.get("attention_mask")
        is_prefill = attn_mask is None or input_ids.shape[1] == attn_mask.shape[1]
        if is_prefill:
            rope_inputs = {
                key: inputs[key]
                for key in ("attention_mask", "image_grid_thw", "video_grid_thw", "second_per_grid_ts")
                if inputs.get(key) is not None
            }
            inputs["position_ids"] = get_mrope_index(
                config, input_ids, inputs.get("mm_token_type_ids"), **rope_inputs
            )[0]

    # Encoder-level: dispatch by marker kwargs (preparer fires when every marker is in `inputs`
    # with a non-`None` value).
    for markers, preparer in _EXPORT_INPUT_PREPARERS.items():
        if all(inputs.get(m) is not None for m in markers):
            preparer(config, inputs)
    return inputs


# ── Decomposition ─────────────────────────────────────────────────────────────
# Split a model into independently exportable components. `decompose_prefill_decode`
# captures the prefill and decode forward kwargs from a real `model.generate()` call;
# `decompose_multimodal` runs a single forward and captures per-submodule kwargs (one
# entry per encoder / projector / language model). Both rely on `_capture_forward` to
# wrap a target submodule and record every call's kwargs.


@contextlib.contextmanager
def _capture_forward(module: torch.nn.Module):
    """Capture forward call kwargs into a list (one dict per call).

    Positional args are normalised to kwargs via `inspect.signature` so the
    captured dicts can be passed directly as `kwargs=inputs` to `torch.export`.
    """

    calls: list[dict] = []
    original = module.forward
    sig = inspect.signature(original)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        captured = {}
        bound = sig.bind(*args, **kwargs)
        for name, value in bound.arguments.items():
            param = sig.parameters[name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                captured.update(copy.deepcopy(value))
            elif param.kind != inspect.Parameter.VAR_POSITIONAL:
                captured[name] = copy.deepcopy(value)
        calls.append(captured)
        return original(*args, **kwargs)

    module.forward = wrapper
    try:
        yield calls
    finally:
        module.forward = original


def _merge_decode_calls(decode_calls: list[dict]) -> dict:
    """Merge consecutive single-token decode captures into one multi-token decode input.

    Each `model.generate` decode step feeds a single new token, so `torch.export` (with `Dim.AUTO`)
    sees a query-sequence axis of length 1 and specializes it to a constant — the exported decode can
    then only ever run one token. Concatenating `N` consecutive decode steps along that axis yields a
    genuine `N`-token decode input: the traced graph is identical (a KV-cache forward), but the sequence
    axis now has hint `N > 1` so it stays dynamic. The exported decode then handles both a single token
    (ordinary decoding) and many (continuation-from-past for multi-turn, or a plain prefill when the cache is empty).

    The cache (`past_key_values`) is taken from the FIRST step (the state right after prefill, before
    the chunk). The per-token tensors are concatenated along their sequence axis; `attention_mask` is
    handled below (its layout depends on the cache).
    """
    first = decode_calls[0]
    merged = copy.copy(first)

    # Concatenation assumes single-token decode steps. When `use_cache` is off the steps re-run the whole
    # growing sequence (query length > 1) — each is already a valid multi-token forward, so take the last.
    def query_length(call: dict) -> int | None:
        for key in ("input_ids", "inputs_embeds"):
            value = call.get(key)
            if value is not None:
                return value.shape[1]
        return None

    if any(query_length(call) != 1 for call in decode_calls):
        return copy.copy(decode_calls[-1])

    def concat_along(key: str, dim: int) -> None:
        values = [call[key] for call in decode_calls if call.get(key) is not None]
        if len(values) == len(decode_calls):
            merged[key] = torch.cat(values, dim=dim)

    concat_along("input_ids", 1)
    concat_along("inputs_embeds", 1)
    concat_along("cache_position", 0)
    # `position_ids` is `[batch, seq]` or `[n_axes, batch, seq]` (m-rope) — the sequence axis is
    # last in both, so a negative dim concatenates it correctly either way.
    concat_along("position_ids", -1)

    # `attention_mask` is either a 2D padding mask `[batch, kv]` (a growing `DynamicCache`: the model
    # rebuilds the causal mask from `position_ids` / `cache_position` internally, so the last step's
    # mask — spanning the most positions — is all it needs) or a 4D causal mask `[batch, heads, query,
    # kv]` (a static cache passes the mask in explicitly). For the 4D case each single-token step is one
    # causal query row against the fixed-size cache, so concatenating along the query axis rebuilds the
    # correct `N`-token causal mask; taking just the last step would freeze the query axis at 1 and the
    # exported decode could never run more than one token. Hybrid-attention models pass a dict
    # `{attention_type: 4D mask}` instead of a single tensor — merge each entry the same way.
    masks = [call.get("attention_mask") for call in decode_calls]
    if all(mask is not None for mask in masks):
        merged["attention_mask"] = _merge_step_masks(masks)

    return merged


def _merge_step_masks(masks: list[Any]) -> Any:
    """Merge one attention mask per decode step into a single multi-token mask.

    A 4D causal mask `[batch, heads, query, kv]` is concatenated along the query axis (each step is one
    causal row against the fixed cache); a 2D padding mask keeps the last step (it already spans the most
    positions). A dict `{attention_type: mask}` (hybrid-attention models) is merged entry by entry, and a
    `None` mask (an attention type the model leaves unmasked) is preserved as `None`.
    """
    last_mask = masks[-1]
    if last_mask is None:
        return None
    if isinstance(last_mask, dict):
        return {key: _merge_step_masks([mask[key] for mask in masks]) for key in last_mask}
    if last_mask.dim() == 4 and all(mask.shape[3] == last_mask.shape[3] for mask in masks):
        return torch.cat(masks, dim=2)
    return last_mask


def materialize_cache_layers(cache: Any, batch_size: int, config: Any, dtype: Any, device: Any) -> None:
    """Give every lazily-uninitialized cache layer real tensors — `torch.export` can't trace lazy
    allocation, so both the traced (prefill) cache and the cache the runtime builds must be materialized,
    and identically (dynamo bakes the cache pytree into the graph's input spec). Static layers allocate
    their full buffers (the same `lazy_initialization` path `Cache.early_initialization` takes); growing
    layers get rank-4 zero-length `[batch, kv_heads, 0, head_dim]` tensors the graph can `cat` onto — NOT
    the 1-D empty tensor their own lazy init makes, which would bake a rank-1 guard into the graph."""
    if hasattr(cache, "self_attention_cache"):  # EncoderDecoderCache — materialize both sub-caches
        materialize_cache_layers(cache.self_attention_cache, batch_size, config, dtype, device)
        materialize_cache_layers(cache.cross_attention_cache, batch_size, config, dtype, device)
        return
    text_config = config.get_text_config()
    num_kv_heads = getattr(text_config, "num_key_value_heads", None) or text_config.num_attention_heads
    head_dim = getattr(text_config, "head_dim", None) or text_config.hidden_size // text_config.num_attention_heads
    for layer in cache.layers:
        if getattr(layer, "is_initialized", True):
            continue
        empty_kv = torch.zeros(batch_size, num_kv_heads, 0, head_dim, dtype=dtype, device=device)
        if layer.get_max_length() == -1:
            layer.dtype, layer.device = dtype, device
            layer.keys, layer.values = empty_kv, empty_kv.clone()
            layer.is_initialized = True
        else:
            layer.lazy_initialization(empty_kv, empty_kv)


def decompose_prefill_decode(
    model: PreTrainedModel,
    inputs: dict[str, Any],
    generation_config: Any = None,
    multi_token_decode: bool = False,
) -> dict[str, tuple[torch.nn.Module, dict]]:
    """Run `model.generate()` and capture prefill and decode inputs.

    Reuses the full generation machinery so every architecture (decoder-only, SSM,
    encoder-decoder, multi-modal, …) gets correct inputs without reimplementing the loop.

    `generation_config` is forwarded to `generate()` (defaulting to the model's own), so the captured
    inputs use whatever cache `generate()` would build. Pass one with `cache_implementation="static"`
    and `max_cache_len=N` to capture a **statically sized** cache in the decode inputs — the basis for
    a static-cache export. `max_cache_len` sizes the cache independently of the capture, so the
    exported decode takes a fixed `[..., N, ...]` cache rather than a growing one.

    When `multi_token_decode`, the `decode` component is captured as a **multi-token** decode — two
    consecutive decode steps merged (see `_merge_decode_calls`) so its query-sequence axis stays
    symbolic (a single-token decode would specialize that axis to 1). It then handles both one token
    (ordinary decoding) and many (continuation-from-past, or a plain prefill when the cache is empty). Otherwise `decode` is the
    classic single-token decode.

    Returns:
        `dict[str, tuple[torch.nn.Module, dict]]`:
        `{"prefill": (model, prefill_inputs), "decode": (model, decode_inputs)}`.
    """
    # 1 prefill forward + 1 decode (or 2 decode steps merged, when `multi_token_decode`) forward to capture.
    # Set the capture window on the config itself, not as generate() kwargs — passing a
    # `generation_config` alongside generation kwargs is deprecated. Base it on the model's own config
    # when none is given (preserving its defaults), and deep-copy into a distinct `capture_config` so
    # the caller's `generation_config` is never mutated.
    # Encoder-decoder decoding is single-token from an (almost) empty self-attention cache: the first
    # decode step runs at cache length 1, which 0/1 specialization would freeze into the graph — capture
    # from the SECOND decode step (cache length 2, symbolic) instead.
    first_decode = 2 if getattr(model.config, "is_encoder_decoder", False) else 1
    num_new_tokens = first_decode + (2 if multi_token_decode else 1)
    capture_config = copy.deepcopy(generation_config if generation_config is not None else model.generation_config)
    capture_config.max_new_tokens = num_new_tokens
    capture_config.min_new_tokens = num_new_tokens
    try:
        with _capture_forward(model) as calls:
            model.generate(**copy.deepcopy(inputs), generation_config=capture_config)
    except Exception as e:
        raise RuntimeError(
            f"decompose_prefill_decode failed for {type(model).__name__}. "
            f"Inputs passed: {list(inputs.keys())}. "
            f"Make sure the inputs are compatible with model.generate()."
        ) from e

    if len(calls) < num_new_tokens:
        raise RuntimeError(
            f"decompose_prefill_decode expected at least {num_new_tokens} calls to "
            f"{type(model).__name__}.forward() during generate(max_new_tokens={num_new_tokens}), but "
            f"captured {len(calls)}. This likely means generate() bypasses the top-level forward() "
            "(e.g. delegates to an inner model), so prefill/decode decomposition is not supported "
            "for this architecture."
        )

    # Remove `logits_to_keep` from the captured calls — it's a generation-time hint for the model's
    # internal top-k pruning, not a forward input. The export graph should not depend on it.
    for call in calls:
        call.pop("logits_to_keep", None)

    # A single-token decode specializes its query-sequence axis to 1 (never dynamic). When
    # `multi_token_decode`, merge the two decode steps into one multi-token decode so that axis stays
    # symbolic (continuation-from-past, or a plain prefill when the cache is empty, and it still covers seq == 1).
    prefill_inputs = calls[0]
    decode_inputs = (
        _merge_decode_calls(calls[first_decode:num_new_tokens]) if multi_token_decode else calls[first_decode]
    )
    return {
        "prefill": (copy.copy(model), prefill_inputs),
        "decode": (copy.copy(model), decode_inputs),
    }


# Projector attribute names — no canonical accessor on `PreTrainedModel`, kept as a heuristic.
# Encoders and language model are resolved via `get_encoder(modality)` / `get_decoder()`.
_MULTIMODAL_PROJECTOR_NAMES = ("multi_modal_projector", "connector", "embed_vision", "embed_audio")


def _find_multimodal_submodules(model: PreTrainedModel) -> dict[str, torch.nn.Module]:
    """Return `{attr_name: module}` for multi-modal submodules found on `model`.

    Uses the canonical `PreTrainedModel.get_encoder("image"/"audio")` and `get_decoder()`
    accessors for the encoders and the decoder. Projectors are looked
    up by name on `model` and its `base_model` (e.g. `LlavaModel` under `LlavaForConditionalGeneration`).

    Only returns results when at least one modal encoder AND a decoder are found —
    otherwise the model is not multi-modal and should be exported as a single unit.
    """
    found: dict[str, torch.nn.Module] = {}

    has_encoder = False
    for modality in ("image", "audio"):
        encoder = model.get_encoder(modality=modality)
        # `get_encoder` returns `self` as the "no match" fallback, and some models keep
        # `self.audio_tower = None` / `self.vision_tower = None` when the corresponding
        # sub-config is absent — `hasattr` is True but `getattr` is None.
        if encoder is not None and encoder is not model:
            found[f"{modality}_encoder"] = encoder
            has_encoder = True

    decoder = model.get_decoder()
    if decoder is not None and decoder is not model:
        found["text_decoder"] = decoder

    for root in {model, model.base_model}:
        for name in _MULTIMODAL_PROJECTOR_NAMES:
            if name not in found and getattr(root, name, None) is not None:
                found[name] = getattr(root, name)

    if not has_encoder or "text_decoder" not in found:
        return {}

    return found


def is_multimodal(model: PreTrainedModel | torch.nn.Module) -> bool:
    """Returns `True` if the model is multi-modal with modal encoders and a language model.

    A non-`PreTrainedModel` (e.g. a bare `nn.Module`) has no canonical `get_encoder`/`get_decoder`
    accessors and is trivially not multi-modal, so it short-circuits to `False`.
    """
    return isinstance(model, PreTrainedModel) and bool(_find_multimodal_submodules(model))


if is_torch_available():

    class _ModelComponent(torch.nn.Module):
        """Base for the standalone export/runtime components a multi-modal model decomposes into. Wraps a
        model (the full VLM, its base, or the text decoder) so a single method can be exported on its own;
        missing attributes fall through to it, so the export precompute introspects the component (`config`,
        submodules, `get_rope_index`, device) exactly as it would the real model."""

        def __init__(self, model: PreTrainedModel):
            super().__init__()
            self.model = model

        def __getattr__(self, name):
            # nn.Module owns params/buffers/submodules (incl. `model`); anything else delegates to the
            # wrapped model. `super().__getattr__("model")` (not `self.model`) avoids re-entering this hook.
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(super().__getattr__("model"), name)

    class ModalityEncoder(_ModelComponent):
        """Wraps one modality's `get_<modality>_features` method.

        `forward` runs `model.<getter>(**kwargs)` and normalises the result to a single
        `[num_tokens, hidden]` tensor — concatenating per-item `pooler_output` lists, else the bare
        `pooler_output` / `last_hidden_state` / tensor — remapping the precompute marker `grid_thw` back to
        the getter's native grid kwarg.
        """

        def __init__(self, model: PreTrainedModel, getter: str, grid_kwarg: str | None = None):
            super().__init__(model)
            self._getter = getter
            self._grid_kwarg = grid_kwarg

        def forward(self, **kwargs):
            if self._grid_kwarg is not None and "grid_thw" in kwargs:
                kwargs[self._grid_kwarg] = kwargs.pop("grid_thw")
            outputs = getattr(self.model, self._getter)(**kwargs)
            features = getattr(outputs, "pooler_output", None)
            if features is None:
                features = getattr(outputs, "last_hidden_state", outputs)
            return torch.cat(features) if isinstance(features, (tuple, list)) else features

    class TokenEmbedder(_ModelComponent):
        """`input_ids -> inputs_embeds`, zeroing the placeholder ids (out of the text vocab) first, the way
        a VLM `forward` does before scattering in encoder features. Wraps the text decoder (never the outer
        VLM), so the export precompute's `get_rope_index` branch stays off on the `input_ids` it carries.
        """

        def __init__(self, decoder: PreTrainedModel, placeholder_ids: list[int]):
            super().__init__(decoder)
            self._placeholder_ids = placeholder_ids

        def forward(self, input_ids):
            placeholder = torch.zeros_like(input_ids, dtype=torch.bool)
            for token_id in self._placeholder_ids:
                placeholder = placeholder | (input_ids == token_id)
            return self.model.get_input_embeddings()(input_ids.masked_fill(placeholder, 0))


# One row per input modality: (component name, `get_*_features` method, the input kwarg that signals the
# modality is present, the getter's native grid kwarg — or `None` for audio, the placeholder-id config
# field). Video/audio slot in exactly like image; a modality is exported only when its getter exists and
# its input is passed.
_MODALITY_SPECS = (
    ("image_encoder", "get_image_features", "pixel_values", "image_grid_thw", "image_token_id"),
    ("video_encoder", "get_video_features", "pixel_values_videos", "video_grid_thw", "video_token_id"),
    ("audio_encoder", "get_audio_features", "input_features", None, "audio_token_id"),
)


@contextlib.contextmanager
def _capture_calls(obj: Any, attribute: str):
    """Capture the kwargs of each `obj.<attribute>(...)` call during the block (positional args normalised
    to kwargs), restoring the attribute afterwards. Generalises `_capture_forward` to any method — used to
    record exactly what the model passes each `get_*_features`, so we don't hardcode per-model input keys."""
    calls: list[dict] = []
    original = getattr(obj, attribute)
    was_instance_attr = attribute in vars(obj)
    sig = inspect.signature(original)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        captured = {}
        for name, value in sig.bind(*args, **kwargs).arguments.items():
            kind = sig.parameters[name].kind
            if kind == inspect.Parameter.VAR_KEYWORD:
                captured.update(copy.deepcopy(value))
            elif kind != inspect.Parameter.VAR_POSITIONAL:
                captured[name] = copy.deepcopy(value)
        calls.append(captured)
        return original(*args, **kwargs)

    setattr(obj, attribute, wrapper)
    try:
        yield calls
    finally:
        if was_instance_attr:
            setattr(obj, attribute, original)
        else:
            delattr(obj, attribute)


def decompose_multimodal(model: PreTrainedModel, inputs: dict[str, Any]) -> dict[str, tuple[torch.nn.Module, dict]]:
    """Split a multi-modal model into independently exportable `name: (module, inputs)` pairs.

    Exports the model's own composition methods rather than raw submodules, so each component is
    self-contained and the set can be reassembled into a generation runtime:
    - `embed_tokens` — `input_ids -> inputs_embeds` (`get_input_embeddings`, placeholder ids zeroed),
    - `<modality>_encoder` — the modality features (`get_<modality>_features`, i.e. encoder **and**
      projection), one per input modality present (image / video / audio),
    - `text_decoder` — captured from the forward (`get_decoder()`; it produces the logits, the head included).

    The token-merge step (`masked_scatter`) stays outside the graphs — the caller assembles
    `inputs_embeds` from the encoder outputs before running the decoder.

    Raises:
        `ValueError`: if no known multi-modal submodules are found on the model.
    """
    submodules = _find_multimodal_submodules(model)
    if not submodules:
        raise ValueError(
            f"decompose_multimodal found no multi-modal submodules on {type(model).__name__}. "
            f"Expected an image/audio encoder + language model, found neither."
        )

    # Each active modality's `get_*_features` is invoked on the base model during `forward` (the outer
    # `ForConditionalGeneration` getter just delegates), so capture — and later export — from there.
    base = model.base_model
    active_modalities = []
    for name, getter, input_key, grid_key, _token_field in _MODALITY_SPECS:
        owner = base if hasattr(base, getter) else (model if hasattr(model, getter) else None)
        if owner is not None and inputs.get(input_key) is not None:
            active_modalities.append((name, getter, owner, grid_key))

    # the `text_decoder` takes activations, not user inputs, so capture its kwargs; capture each
    # modality getter's call kwargs — all in one real forward.
    lm_targets = {name: submodules[name] for name in ("text_decoder",) if name in submodules}
    try:
        with contextlib.ExitStack() as stack, torch.no_grad():
            captured_lm = {name: stack.enter_context(_capture_forward(module)) for name, module in lm_targets.items()}
            captured_features = {
                name: stack.enter_context(_capture_calls(owner, getter))
                for name, getter, owner, _ in active_modalities
            }
            model(**copy.deepcopy(inputs))
    except Exception as e:
        raise RuntimeError(
            f"decompose_multimodal failed for {type(model).__name__}. Inputs passed: {list(inputs.keys())}."
        ) from e

    components = {name: (module, captured_lm[name][-1]) for name, module in lm_targets.items() if captured_lm[name]}

    # embed_tokens: `input_ids -> inputs_embeds`, zeroing the placeholder ids (out of the text vocab)
    # first, the way a VLM `forward` does before scattering in encoder features.
    placeholder_ids = [
        getattr(model.config, spec[-1], None)
        for spec in _MODALITY_SPECS
        if getattr(model.config, spec[-1], None) is not None
    ]
    components["embed_tokens"] = (
        TokenEmbedder(model.get_decoder(), placeholder_ids),
        {"input_ids": inputs["input_ids"]},
    )

    # One feature graph per modality, from the captured getter call — a `ModalityEncoder` wrapping the
    # owner, whose `forward` runs `get_<modality>_features` and delegates introspection to the model.
    for name, getter, owner, grid_key in active_modalities:
        calls = captured_features[name]
        if not calls:
            continue
        feature_inputs = {
            ("grid_thw" if key == grid_key else key): value for key, value in calls[-1].items() if value is not None
        }
        components[name] = (ModalityEncoder(owner, getter, grid_key), feature_inputs)
    return components


def decompose_for_generation(
    model: PreTrainedModel, inputs: dict[str, Any], generation_config: Any = None, multi_token_decode: bool = False
) -> dict[str, tuple[torch.nn.Module, dict]]:
    """Decompose a generative model into independently exportable `(model, forward_inputs)` pairs.

    Runs `decompose_prefill_decode` to capture prefill and decode forward kwargs from a real
    `model.generate(**inputs, max_new_tokens=2)`. If the prefill is multi-modal (per `is_multimodal`),
    further splits it into one entry per submodule (vision/audio encoder, projector, language model,
    `text_decoder`) via `decompose_multimodal`.

    Args:
        model: Generative model. Must support `model.generate(**inputs)`.
        inputs: **Generate** kwargs — what you'd pass to `model.generate(**inputs)`.
        generation_config: Optional `GenerationConfig` forwarded to `generate()` during capture. Pass
            one with `cache_implementation="static"` + `max_cache_len=N` to export against a statically
            sized cache (see `decompose_prefill_decode`).
        multi_token_decode: When `True`, capture the `decode` component as a multi-token decode (dynamic
            query sequence axis: multiple tokens at once — continuation-from-past, or a plain prefill when the cache is empty); a
            single-token decode can't stay dynamic (see `decompose_prefill_decode`).

    Returns:
        `{component_name: (submodel, forward_inputs)}`. Keys are `"prefill"` / `"decode"` for
        plain generative models and `"embed_tokens"` / `"image_encoder"` / `"audio_encoder"` /
        `"text_decoder"` / `"decode"` for multi-modal generative models. For multi-modal
        models the `decode` component takes `inputs_embeds` (not `input_ids`) so the caller can scatter the
        encoder features into the embeddings before running it.
    """
    if getattr(model.config, "is_encoder_decoder", False):
        # `generate` runs the encoder once outside the decoder loop (`get_encoder()(...)`), so it never
        # appears in the captured forwards — capture its call during the same generate to export it as its
        # own component (the runtime serves it back through `get_encoder()`).
        with _capture_calls(model.get_encoder(), "forward") as encoder_calls:
            stages = decompose_prefill_decode(
                model, inputs, generation_config=generation_config, multi_token_decode=multi_token_decode
            )
        encoder_inputs = {k: v for k, v in encoder_calls[0].items() if isinstance(v, torch.Tensor)}
        stages = {"encoder": (model.get_encoder(), encoder_inputs), **stages}
        # Each encoder returns its own `ModelOutput` subclass, and dynamo bakes the pytree type into the
        # decoder graphs' input spec. The decoder only reads `last_hidden_state`, so normalize to the base
        # class — every model's graphs then take the same `encoder_outputs` the runtime reconstructs.
        for _model, stage_inputs in stages.values():
            if (encoder_outputs := stage_inputs.get("encoder_outputs")) is not None:
                stage_inputs["encoder_outputs"] = BaseModelOutput(last_hidden_state=encoder_outputs.last_hidden_state)
    else:
        stages = decompose_prefill_decode(
            model, inputs, generation_config=generation_config, multi_token_decode=multi_token_decode
        )
    prefill_model, prefill_inputs = stages["prefill"]

    if not is_multimodal(prefill_model):
        # The captured prefill cache is the pre-forward, lazily-uninitialized one; materialize it so the
        # exported prefill takes the same cache pytree the runtime feeds (decode, captured post-prefill,
        # already does). Text path only — the multi-modal prefill is discarded after the submodule split,
        # and materializing it would leak cache inputs into the `text_decoder` component's capture.
        if (cache := prefill_inputs.get("past_key_values")) is not None:
            batch_size = next(t for t in prefill_inputs.values() if isinstance(t, torch.Tensor)).shape[0]
            materialize_cache_layers(cache, batch_size, model.config, module_dtype(model), module_device(model))
        return stages

    components = decompose_multimodal(prefill_model, prefill_inputs)

    # Feed the decode graph `inputs_embeds` (not `input_ids`) so the runtime can scatter the encoder embeds
    # into the embeddings before the text stack; the full forward accepts `inputs_embeds` and — with no
    # modality inputs — skips the encoders. This is what lets the components reassemble into a loop.
    decode_model, decode_inputs = stages["decode"]
    decode_inputs = copy.copy(decode_inputs)
    for _name, _getter, _input_key, grid_key, _token_field in _MODALITY_SPECS:
        decode_inputs.pop(_input_key, None)
        if grid_key is not None:
            decode_inputs.pop(grid_key, None)
    # `mm_token_type_ids` only drives the model's internal M-RoPE (`get_rope_index`); once `position_ids`
    # is captured, the decode forward never reads it. Drop it from the decode graph's inputs so the runtime
    # (which supplies `position_ids` via `modeling_rope_utils.get_mrope_index`) needn't thread a per-step token-type tensor.
    if decode_inputs.get("position_ids") is not None:
        decode_inputs.pop("mm_token_type_ids", None)
    if decode_inputs.get("input_ids") is not None:
        embedding = components["embed_tokens"][0]
        with torch.no_grad():
            decode_inputs["inputs_embeds"] = embedding(decode_inputs.pop("input_ids"))
    components["decode"] = (decode_model, decode_inputs)
    return components
