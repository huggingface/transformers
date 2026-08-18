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

    from ..cache_utils import StaticLayer
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

    # Drop kwargs that are `None`: `torch.export` still records them as placeholders carrying no value, so
    # the graph declares an "input" there and dynamo then demands the key back on every call — a hole the
    # runtime has to fill with `None` for no benefit. Only when the parameter's default is `None` too, or
    # omitting it would switch the traced path (a `use_cache=True` default handed `None` would flip to True).
    forward = getattr(model, "forward", None)
    if forward is not None:
        parameters = inspect.signature(forward).parameters
        for name in [name for name, value in inputs.items() if value is None]:
            parameter = parameters.get(name)
            if parameter is not None and parameter.default is None:
                inputs.pop(name)

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

    if hasattr(module, "get_vision_temporal_slice_index"):
        # ernie4_5_vl_moe's merger interleaves even/odd frames through a `range(0, temporal_size, 2)` loop
        # over the grid's values — untraceable, and the indices depend on nothing but the grid.
        inputs["temporal_slice_index"] = module.get_vision_temporal_slice_index(grid_thw, spatial_merge_size)


@register_export_input_preparer("target_sizes")
def _prepare_navit_vision_inputs(config: Any, inputs: dict[str, Any]) -> None:
    """NaViT-style packed encoders carry per-image `(h, w)` as `target_sizes` instead of `grid_thw`.
    Synthesise `grid_thw = [1, h, w]` and run the nearest-position-id / window-index /
    merged-shape / maximum-sequence-length helpers outside the traced graph."""
    target_sizes = inputs["target_sizes"]
    num_patches_per_side = _find_config_attr(config, "num_patches_per_side")
    if num_patches_per_side is None:
        # The tower derives the grid side rather than declaring it (minicpmv4_6's embeddings hold
        # `image_size // patch_size`), and the precompute only ever sees the config — so derive it the same
        # way. Reached only via the `target_sizes` marker, so an anyres model never lands here.
        image_size = _find_config_attr(config, "image_size")
        patch_size = _find_config_attr(config, "patch_size")
        if image_size is not None and patch_size is not None:
            num_patches_per_side = image_size // patch_size
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
    # `token_type_ids` is per-token too, and left at one step it specializes the merged graph's query axis
    # back to 1 (`Guard failed: token_type_ids.size()[1] == 1`) — defeating the whole point of the merge.
    concat_along("token_type_ids", -1)

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


# Model types whose latent attention caches the *compressed* latent as a single head; the others sharing
# those config fields (axk, deepseek_v32, glm_moe_dsa) cache decompressed keys, one per KV head. Listing a
# model here only selects between two readings of `kv_lora_rank` / `qk_rope_head_dim` — a config without
# those fields is not caching a latent at all, whatever its model type, and derives its geometry the
# ordinary way.
_COMPRESSED_LATENT_ATTENTIONS = {
    "axk1",
    "deepseek_v2",
    "deepseek_v3",
    "glm4_moe_lite",
    "kimi_k25",
    "minicpm3",
    "youtu",
}


def _cache_kv_geometry(config: Any, layer_idx: int | None = None) -> tuple[int, int, int] | None:
    """`(num_kv_heads, key_head_dim, value_head_dim)` of the KV cache a model writes, from its config.

    `None` for a model with no attention at all (mamba, rwkv, …): its cache holds only recurrent states,
    which the graph carries and the write-back fills in, so there is no key/value geometry to derive.

    A heterogeneous config (gemma4, …) declares geometry fields like `head_dim` as *per-layer*, and
    reading them off the global config raises rather than silently returning a value that may be wrong
    for some layers. Pass `layer_idx` to read that layer's own config — the geometry is per layer, so
    the callers that allocate or check a specific layer resolve it that way.
    """
    text_config = config.get_text_config()
    if layer_idx is not None and getattr(text_config, "is_heterogeneous", False):
        text_config = text_config.per_layer_config[layer_idx]
    if getattr(text_config, "num_attention_heads", None) is None:
        return None
    # Latent attention (deepseek_v2/v3, kimi_k25, minicpm3, axk, …) caches something other than one entry
    # per KV head, and in two styles their configs cannot tell apart — identical `kv_lora_rank` /
    # `qk_*_head_dim` fields, but one caches the *compressed* latent and the other the decompressed keys,
    # so the model type is the only discriminator.
    kv_lora_rank = getattr(text_config, "kv_lora_rank", None)
    if kv_lora_rank is not None and getattr(text_config, "model_type", None) in _COMPRESSED_LATENT_ATTENTIONS:
        # one head holding the compressed latent as keys and the shared rope part as values — so the key
        # and value head dims differ, unlike standard attention
        num_kv_heads = 1
        key_dim = kv_lora_rank
        value_dim = getattr(text_config, "qk_rope_head_dim", None) or key_dim
    else:
        num_kv_heads = getattr(text_config, "num_key_value_heads", None) or text_config.num_attention_heads
        default_dim = (
            getattr(text_config, "head_dim", None) or text_config.hidden_size // text_config.num_attention_heads
        )
        # decompressed latent attention keys are `qk_nope + qk_rope` wide against `v_head_dim` values
        qk_nope = getattr(text_config, "qk_nope_head_dim", None)
        qk_rope = getattr(text_config, "qk_rope_head_dim", None)
        key_dim = qk_nope + qk_rope if qk_nope and qk_rope else default_dim
        value_dim = getattr(text_config, "v_head_dim", None) or default_dim
    return num_kv_heads, key_dim, value_dim


def _cache_halves(cache: Any, reference: Any = None) -> list[tuple[Any, Any]]:
    """The `(cache, reference)` pairs to walk. An `EncoderDecoderCache` keeps its layers in the two caches
    it pairs rather than on itself, and each half has to be matched with the same half of the reference."""
    if hasattr(cache, "self_attention_cache"):
        return [
            (getattr(cache, half), getattr(reference, half, None))
            for half in ("self_attention_cache", "cross_attention_cache")
        ]
    return [(cache, reference)]


def check_cache_geometry(config: Any, cache: Any) -> None:
    """Raise if the geometry `_cache_kv_geometry` derives disagrees with what the model really cached.

    Called on a post-prefill cache, whose layers hold real tensors. The derivation cannot be read off
    the config values alone for latent attention — a compressed and a decompressed model carry
    identical `kv_lora_rank` / `qk_*_head_dim` fields — so this turns the resulting mismatch into a
    message naming the fix, instead of an `index_copy_()` shape error deep in a later forward.

    Raises only when the derivation matches *no* layer. A model may cache different geometries across
    layers (deepseek_v32's sparse-indexer layers next to its latent ones), and the exporter fills only the
    layers that reach it uninitialized — so a layer disagreeing is normal, and none agreeing is the
    failure: whatever the exporter would have materialized fits nothing the model actually writes.
    """
    cached, derived_any = [], None
    for cache_half, _ in _cache_halves(cache):
        for layer_idx, layer in enumerate(cache_half.layers):
            if getattr(layer, "keys", None) is None:
                continue
            derived = _cache_kv_geometry(config, layer_idx)
            if derived is None:
                continue
            actual = (layer.keys.shape[1], layer.keys.shape[3], layer.values.shape[3])
            if actual == derived:
                return
            cached.append(actual)
            derived_any = derived
    if cached:
        model_type = getattr(config.get_text_config(), "model_type", type(config).__name__)
        raise ValueError(
            f"`{model_type}` caches (heads, key_dim, value_dim)={sorted(set(cached))} but the exporter "
            f"derives {derived_any} for every layer, so the runtime would build a cache the exported "
            "graph rejects. If this model caches a compressed latent (one head), add its model type to "
            "`_COMPRESSED_LATENT_ATTENTIONS`; otherwise `_cache_kv_geometry` needs to learn its layout."
        )


def kv_geometry_of(cache: Any) -> dict[int, tuple[int, int, int]]:
    """`{layer index: (num_kv_heads, key_head_dim, value_head_dim)}` of a cache the model itself filled."""
    return {
        index: (layer.keys.shape[1], layer.keys.shape[3], layer.values.shape[3])
        for index, layer in enumerate(getattr(cache, "layers", []) or [])
        if getattr(layer, "keys", None) is not None and layer.keys.dim() == 4
    }


def materialize_cache_layers(
    cache: Any,
    batch_size: int,
    config: Any,
    dtype: Any,
    device: Any,
    kv_geometry: dict[int, tuple[int, int, int]] | None = None,
) -> None:
    """Give every lazily-uninitialized cache layer real tensors — `torch.export` can't trace lazy
    allocation, so both the traced (prefill) cache and the cache the runtime builds must be materialized,
    and identically (dynamo bakes the cache pytree into the graph's input spec). Static layers allocate
    their full buffers (the same `lazy_initialization` path `Cache.early_initialization` takes); growing
    layers get rank-4 zero-length `[batch, kv_heads, 0, head_dim]` tensors the graph can `cat` onto — NOT
    the 1-D empty tensor their own lazy init makes, which would bake a rank-1 guard into the graph.

    `kv_geometry` is the per-layer geometry read off the graph (`ModelRunner.kv_geometry`) or off a cache the
    model filled (`kv_geometry_of`). It wins where present, because a config cannot always give it —
    mimo_v2_flash caches 2 KV heads on its sliding layers and 4 on its full ones — and the config derivation
    covers the layers it does not reach.
    """
    kv_geometry = kv_geometry or {}
    for cache_half, _ in _cache_halves(cache):
        _materialize_layers(cache_half, batch_size, config, dtype, device, kv_geometry)


def _materialize_layers(cache, batch_size, config, dtype, device, kv_geometry) -> None:
    """`materialize_cache_layers` for one flat cache — see there."""
    for layer_idx, layer in enumerate(cache.layers):
        # `is_initialized` is not enough on its own: a layer whose own lazy init already ran holds a *rank-1*
        # empty, and the graph was traced against the rank-4 `[batch, kv_heads, 0, head_dim]` form this
        # helper builds. Feeding the rank-1 one indexes an axis that isn't there, inside the graph.
        keys = getattr(layer, "keys", None)
        rank_1_empty = keys is not None and keys.dim() < 4 and keys.numel() == 0
        if getattr(layer, "is_initialized", True) and not rank_1_empty:
            continue
        geometry = kv_geometry.get(layer_idx) or _cache_kv_geometry(config, layer_idx)
        if geometry is None:
            return
        num_kv_heads, key_dim, value_dim = geometry
        # A static layer also *records* its head count, and that record is compared as part of the graph's
        # input spec — so it has to come from the same place the buffers do, not from the config's single
        # value (mimo_v2_flash caches 2 heads on sliding layers and 4 on full ones).
        if hasattr(layer, "num_heads"):
            layer.num_heads = num_kv_heads
        empty_keys = torch.zeros(batch_size, num_kv_heads, 0, key_dim, dtype=dtype, device=device)
        empty_values = torch.zeros(batch_size, num_kv_heads, 0, value_dim, dtype=dtype, device=device)
        # Growing vs fixed-size is the layer's *kind*, not what `get_max_length` reports: a
        # `DynamicSlidingWindowLayer` grows and crops, yet reports its window as a max length — read that way
        # it goes through its own `lazy_initialization` and ends up with the rank-1 empties the graph cannot
        # index, which surfaces as `IndexError: tuple index out of range` inside the graph.
        if not isinstance(layer, StaticLayer):
            layer.dtype, layer.device = dtype, device
            layer.keys, layer.values = empty_keys, empty_values
            layer.is_initialized = True
            # Whatever else the layer already holds goes to the same device: a sliding layer builds
            # `_sliding_window_tensor` in `__init__` and relies on its own `lazy_initialization` to move it,
            # which this branch skips (those rank-1 empties would bake a rank-1 guard). Left behind, it is a
            # cpu leaf among cuda ones and the graph's input spec mismatches on device. Dtypes stay as they
            # are — that tensor is `long`, not the cache dtype.
            for attribute, value in vars(layer).items():
                if isinstance(value, torch.Tensor) and attribute not in ("keys", "values"):
                    setattr(layer, attribute, value.to(device))
        else:
            layer.lazy_initialization(empty_keys, empty_values)
        # A sparse-indexer layer (deepseek_v32, axk2) caches a *third* tensor beside keys and values, and
        # it is a graph input like the others — leave it lazy and every later cache leaf shifts by one.
        # Rank-3 zero-length for the same reason the others are rank-4: its own lazy init makes a 1-D
        # empty, which would bake a rank-1 guard into the graph.
        if hasattr(layer, "is_indexer_initialized") and not layer.is_indexer_initialized:
            index_head_dim = config.get_text_config().index_head_dim
            empty_indexer_keys = torch.zeros(batch_size, 0, index_head_dim, dtype=dtype, device=device)
            if layer.get_max_length() == -1:
                layer.indexer_dtype, layer.indexer_device = dtype, device
                layer.indexer_keys = empty_indexer_keys
                layer.is_indexer_initialized = True
            else:
                # allocates the full `[batch, max_cache_len, index_head_dim]` buffer from the hint's shape,
                # and puts the layer's own `indexer_cumulative_length` counter on the right device
                layer.lazy_initialization_indexer(empty_indexer_keys)


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
        # A cache-shape error here means the geometry `_cache_kv_geometry` derived for the materialized
        # cache disagrees with what the model writes into it — say so, rather than leaving an
        # `index_copy_()` slice error from deep inside the forward.
        if "slice shapes" in str(e) or "Sizes of tensors must match" in str(e):
            raise RuntimeError(
                f"decompose_prefill_decode failed for {type(model).__name__}: the exporter materialized the "
                f"cache as (heads, key_dim, value_dim)={_cache_kv_geometry(model.config, 0)}, which is not what "
                "this model caches. If its attention caches a compressed latent (one head), add its model "
                "type to `_COMPRESSED_LATENT_ATTENTIONS`; otherwise `_cache_kv_geometry` needs to learn its "
                "layout."
            ) from e
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
    # `generate` built this cache itself, so its layers carry the model's real geometry — the one thing
    # that can tell us whether the geometry the exporter derives (and materializes for the prefill capture
    # and the runtime) is right for this architecture.
    if (captured_cache := decode_inputs.get("past_key_values")) is not None:
        check_cache_geometry(model.config, captured_cache)

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
            # `precompute_export_inputs` derives its tensors from the config alone, so it offers whatever
            # the config implies — a windowed vision config yields `window_index` even for a getter that
            # never takes one (minicpmv4_6). Keep only what this getter actually declares.
            getter = getattr(self.model, self._getter)
            parameters = inspect.signature(getter).parameters
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
                kwargs = {name: value for name, value in kwargs.items() if name in parameters}
            outputs = getter(**kwargs)
            # Most getters put the features in `pooler_output` or `last_hidden_state`. Some declare both
            # and fill neither (granite4_vision returns its features as `hidden_states` +
            # `deepstack_features`), so fall through to the whole output rather than the `None` those
            # fields hold — a default on `getattr` only covers a *missing* attribute, not a null one.
            features = getattr(outputs, "pooler_output", None)
            if features is None:
                features = getattr(outputs, "last_hidden_state", None)
            if features is None:
                features = outputs
            return torch.cat(features) if isinstance(features, (tuple, list)) else features

    class PatchVisionEncoder(_ModelComponent):
        """An anyres vision tower + projector, cut *before* `pack_image_features`.

        The packing decides how many tokens each image contributes from that image's own size, so tracing it
        bakes one `(image count, sizes)` pair into the graph. Everything up to the projector is plain batched
        compute over a flat `(total_patches, channels, height, width)` tensor, so the component stops there
        and the runtime packs the result — the same split optimum-intel's `OVModelForVisualCausalLM` uses.
        `image_newline` rides along as a second output: the packing needs that weight and the runtime holds
        no module to read it off.
        """

        def projector_specs(self) -> list[tuple[int, Any, Any]] | None:
            """`(llm_layer, vision_layer, projector)` per projector this tower feeds, or `None` for the
            single-projector case. A deepstack tower (granite4_vision) runs one projector per
            `deepstack_layer_map` entry and one per `spatial_target_layers` group, each injected into the
            decoder at its own layer — both loop counts come from the config, so they unroll legitimately."""
            config = self.model.config
            layer_map = getattr(config, "deepstack_layer_map", None)
            if not layer_map:
                return None
            specs = [
                (llm_layer, vision_layer, self.model.layerwise_projectors[index])
                for index, (vision_layer, llm_layer) in enumerate(layer_map)
            ]
            specs += [
                (llm_layer, config.spatial_vision_layer, self.model.spatial_projectors[index])
                for index, llm_layer in enumerate(config.spatial_target_layers)
            ]
            return specs

        def forward(self, pixel_values, vision_feature_layer=None, vision_feature_select_strategy=None):
            outputs = self.model.vision_tower(pixel_values, output_hidden_states=True, return_dict=True)

            def project(layer, projector):
                if isinstance(layer, int):
                    selected = outputs.hidden_states[layer]
                else:
                    selected = torch.cat([outputs.hidden_states[index] for index in layer], dim=-1)
                if vision_feature_select_strategy == "default":
                    selected = selected[:, 1:]
                return projector(selected)

            specs = self.projector_specs()
            if specs is None:
                features = {"image_features": project(vision_feature_layer, self.model.multi_modal_projector)}
            else:
                # Keyed by the decoder layer each one is injected at, so the runtime rebuilds the
                # `deepstack_features` map without needing the config's ordering again.
                features = {f"image_features.{llm}": project(layer, proj) for llm, layer, proj in specs}
            features["image_newline"] = self.model.image_newline
            return features

    class TokenEmbedder(_ModelComponent):
        """`input_ids -> inputs_embeds`, zeroing the placeholder ids (out of the text vocab) first, the way
        a VLM `forward` does before scattering in encoder features. Wraps the text decoder (never the outer
        VLM), so the export precompute's `get_rope_index` branch stays off on the `input_ids` it carries.

        A decoder with per-layer embeddings (gemma3n, gemma4) reads a *second* per-token embedding straight
        from `input_ids`, and recovers them by an exact reverse lookup when handed `inputs_embeds` alone —
        data-dependent, and it fails outright once features are scattered in. So this returns that tensor
        too, under the `per_layer_inputs` kwarg the decoder's `forward` already takes to skip the lookup.
        Its placeholder rows survive into the decoder untouched (nothing scatters over them), so they use
        the pad id the eager forward substitutes rather than the zero standing in for the text embedding.
        """

        def __init__(self, decoder: PreTrainedModel, placeholder_ids: list[int]):
            super().__init__(decoder)
            self._placeholder_ids = placeholder_ids

        def _placeholder_mask(self, input_ids):
            placeholder = torch.zeros_like(input_ids, dtype=torch.bool)
            for token_id in self._placeholder_ids:
                placeholder = placeholder | (input_ids == token_id)
            return placeholder

        def forward(self, input_ids):
            placeholder = self._placeholder_mask(input_ids)
            inputs_embeds = self.model.get_input_embeddings()(input_ids.masked_fill(placeholder, 0))
            if not hasattr(self.model, "get_per_layer_inputs"):
                return inputs_embeds
            pad_token_id = self.model.config.get_text_config().pad_token_id or 0
            per_layer_ids = input_ids.masked_fill(placeholder, pad_token_id)
            per_layer_inputs = self.model.get_per_layer_inputs(per_layer_ids, None)
            return {"inputs_embeds": inputs_embeds, "per_layer_inputs": per_layer_inputs}


# One row per input modality: (component name, `get_*_features` method, the input kwarg that signals the
# modality is present, the getter's native grid kwarg — or `None` for audio, the placeholder-id config
# field). Video/audio slot in exactly like image; a modality is exported only when its getter exists and
# its input is passed.
# The input kwarg is a tuple: a model may name the same modality differently (video_llava splits images
# and videos, so its images arrive as `pixel_values_images`). The first name present is the one used.
_MODALITY_SPECS = (
    (
        "image_encoder",
        "get_image_features",
        ("pixel_values", "pixel_values_images"),
        "image_grid_thw",
        "image_token_id",
    ),
    ("video_encoder", "get_video_features", ("pixel_values_videos",), "video_grid_thw", "video_token_id"),
    # `audio_input_ids` (inkling) is the same slot as `input_features` — the tensor whose presence means
    # this call carries audio — just named for a getter that takes discrete codes rather than a spectrogram.
    ("audio_encoder", "get_audio_features", ("input_features", "audio_input_ids"), None, "audio_token_id"),
)


_MODALITY_GETTERS = {name: getter for name, getter, *_ in _MODALITY_SPECS}


def anyres_patch_counts(config: Any, image_sizes) -> list[int]:
    """Tiles each image snaps to, plus the base patch — the per-image split sizes `get_image_features`
    derives from `image_sizes`. Config-only, so the export and the runtime agree without a model."""
    from ..image_processing_utils import select_best_resolution

    pinpoints = _find_config_attr(config, "image_grid_pinpoints")
    tile = _find_config_attr(config, "image_size")
    counts = []
    for size in image_sizes:
        height, width = select_best_resolution(size.tolist() if hasattr(size, "tolist") else list(size), pinpoints)
        counts.append(-(-height // tile) * -(-width // tile) + 1)
    return counts


def flatten_anyres_patches(config: Any, pixel_values, image_sizes):
    """The flat `(total_patches, …)` tensor the tower takes, dropping each image's padding rows. The getter
    does this from `image_sizes`; doing it here keeps the sizes out of the graph entirely."""
    if pixel_values.dim() != 5:
        return pixel_values
    counts = anyres_patch_counts(config, image_sizes)
    return torch.cat([pix[:count] for pix, count in zip(pixel_values, counts)], dim=0)


def packs_anyres_features(owner: Any, config: Any) -> bool:
    """Whether this image getter ends in the anyres `pack_image_features`, so the component must stop at the
    projector and the runtime packs instead. Covers a deepstack tower (granite4_vision) too — it just has one
    projector per injected decoder layer rather than one overall."""
    projectors = ("multi_modal_projector", "layerwise_projectors")
    return (
        _find_config_attr(config, "image_grid_pinpoints") is not None
        and any(hasattr(owner, name) for name in projectors)
        and all(hasattr(owner, attr) for attr in ("vision_tower", "image_newline"))
    )


def _modality_owner(model, getter):
    """Whichever of the model or its base actually defines a modality getter."""
    base = model.base_model
    return base if hasattr(base, getter) else (model if hasattr(model, getter) else None)


def _present_input_key(inputs, input_keys):
    """The modality's input kwarg that this call actually carries, or `None` when the modality is absent."""
    return next((key for key in input_keys if inputs.get(key) is not None), None)


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


def _embeds_input_ids(decoder: Any) -> bool:
    """Whether `decoder` turns `input_ids` into embeddings with a single module.

    Models that embed nothing raise rather than return. A multi-codebook decoder (musicgen) answers with a
    `ModuleList` — one embedding per codebook — which is a container with no `forward`, so there is no one
    `input_ids -> inputs_embeds` graph to export for it either."""
    try:
        embeddings = decoder.get_input_embeddings() if decoder is not None else None
    except NotImplementedError:
        return False
    return isinstance(embeddings, torch.nn.Module) and not isinstance(embeddings, torch.nn.ModuleList)


def decompose_multimodal(
    model: PreTrainedModel,
    inputs: dict[str, Any],
    recorded_features: dict[str, list] | None = None,
    prompt_ids: torch.Tensor | None = None,
) -> dict[str, tuple[torch.nn.Module, dict]]:
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
    # `inputs` says which modalities this forward carries. A model that consumed them earlier — an
    # encoder-decoder feeds its images through the encoder, so by prefill they are gone — has none left to
    # find, and the caller instead hands over what it recorded the getters doing during the same generate.
    recorded_features = recorded_features or {}
    active_modalities = []
    for name, getter, input_keys, grid_key, _token_field in _MODALITY_SPECS:
        owner = base if hasattr(base, getter) else (model if hasattr(model, getter) else None)
        if owner is None:
            continue
        if _present_input_key(inputs, input_keys) is not None or recorded_features.get(name):
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
    # first, the way a VLM `forward` does before scattering in encoder features. Only a model whose
    # prompt *is* text gets one: an encoder-decoder (t5gemma, seamless_m4t, …) prompts with the encoded
    # modality and reaches its decoder through cross-attention, with no placeholder rows to scatter into,
    # so there is nothing for this component to do and the runtime drives those graphs directly. A
    # dual-encoder (owlvit, groupvit, …) has a text tower rather than a decoder that embeds ids, and says
    # so by refusing `get_input_embeddings` — take it at its word instead of failing the whole export.
    # `prompt_ids` covers the same gap as `recorded_features`: an encoder-decoder's prefill kwargs carry
    # `decoder_input_ids`, so the prompt this component embeds has to come from the generate inputs.
    token_ids = inputs.get("input_ids") if inputs.get("input_ids") is not None else prompt_ids
    if token_ids is not None and _embeds_input_ids(model.get_decoder()):
        placeholder_ids = [
            getattr(model.config, spec[-1], None)
            for spec in _MODALITY_SPECS
            if getattr(model.config, spec[-1], None) is not None
        ]
        components["embed_tokens"] = (
            TokenEmbedder(model.get_decoder(), placeholder_ids),
            {"input_ids": token_ids},
        )

    # One feature graph per modality, from the captured getter call — a `ModalityEncoder` wrapping the
    # owner, whose `forward` runs `get_<modality>_features` and delegates introspection to the model.
    for name, getter, owner, grid_key in active_modalities:
        calls = captured_features.get(name) or recorded_features.get(name) or []
        if not calls:
            continue
        feature_inputs = {
            ("grid_thw" if key == grid_key else key): value for key, value in calls[-1].items() if value is not None
        }
        # Hand the graph the tensors the precompute derives from the config, so it takes them as inputs
        # instead of deriving them itself — the point of the precompute, and the only way a getter that
        # reads its grid or image sizes *as data* can be traced at all.
        if name == "image_encoder" and packs_anyres_features(owner, model.config):
            tower_inputs = {
                "pixel_values": flatten_anyres_patches(
                    model.config, feature_inputs["pixel_values"], feature_inputs["image_sizes"]
                )
            }
            tower_inputs.update(
                {
                    key: feature_inputs[key]
                    for key in ("vision_feature_layer", "vision_feature_select_strategy")
                    if key in feature_inputs
                }
            )
            components[name] = (PatchVisionEncoder(owner), tower_inputs)
            continue
        feature_inputs = precompute_export_inputs(model.config, feature_inputs)
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
    recorded_features: dict[str, list] = {}
    if getattr(model.config, "is_encoder_decoder", False):
        # `generate` runs the encoder once outside the decoder loop (`get_encoder()(...)`), so it never
        # appears in the captured forwards — capture its call during the same generate to export it as its
        # own component (the runtime serves it back through `get_encoder()`).
        # Record the modality getters over the same generate: this model runs its vision tower once, into
        # the encoder, so the prefill kwargs the split sees below no longer carry the images.
        modality_owners = {
            name: owner
            for name, getter, *_ in _MODALITY_SPECS
            if (owner := _modality_owner(model, getter)) is not None
        }
        with contextlib.ExitStack() as stack:
            encoder_calls = stack.enter_context(_capture_calls(model.get_encoder(), "forward"))
            live = {
                name: stack.enter_context(_capture_calls(owner, _MODALITY_GETTERS[name]))
                for name, owner in modality_owners.items()
            }
            stages = decompose_prefill_decode(
                model, inputs, generation_config=generation_config, multi_token_decode=multi_token_decode
            )
        recorded_features = {name: list(calls) for name, calls in live.items() if calls}
        encoder_inputs = {k: v for k, v in encoder_calls[0].items() if isinstance(v, torch.Tensor)}
        stages = {"encoder": (model.get_encoder(), encoder_inputs), **stages}
        # Each encoder returns its own `ModelOutput` subclass, and dynamo bakes the pytree type into the
        # decoder graphs' input spec. The decoder only reads `last_hidden_state`, so normalize to the base
        # class — every model's graphs then take the same `encoder_outputs` the runtime reconstructs.
        # A decoder that reads more than that (cohere_asr wants the encoder's own `attention_mask`) cannot
        # be served this way: `ModelOutput` flattens by its *dict*, so a field the encoder attached after
        # construction never reaches the traced region, and a `None` one cannot be put in the dict either
        # (the base class would reject the key on unflatten). Those need their encoder-output class kept
        # end-to-end and rebuilt by every backend's runtime — see the `cohere_asr` skip.
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
            # The decode capture ran after prefill, so its cache carries the geometry the model really
            # writes — better than any derivation from the config.
            # The decode capture ran after prefill, so its cache carries the geometry the model really
            # writes, per layer — read the shapes off it for the prefill cache the graph will be traced with.
            materialize_cache_layers(
                cache,
                batch_size,
                model.config,
                module_dtype(model),
                module_device(model),
                kv_geometry=kv_geometry_of(stages["decode"][1].get("past_key_values")),
            )
        return stages

    components = decompose_multimodal(prefill_model, prefill_inputs, recorded_features, inputs.get("input_ids"))
    # The multi-modal split rebuilds the component set from the prefill, so carry over the stages that
    # belong to the model as a whole — an encoder-decoder's `encoder` runs once outside the decode loop and
    # is captured above, and dropping it leaves the runtime with a decode graph asking for `encoder_outputs`
    # nothing produces.
    if "encoder" in stages:
        components["encoder"] = stages["encoder"]

    # Feed the decode graph `inputs_embeds` (not `input_ids`) so the runtime can scatter the encoder embeds
    # into the embeddings before the text stack; the full forward accepts `inputs_embeds` and — with no
    # modality inputs — skips the encoders. This is what lets the components reassemble into a loop.
    decode_model, decode_inputs = stages["decode"]
    decode_inputs = copy.copy(decode_inputs)
    for _name, _getter, _input_keys, grid_key, _token_field in _MODALITY_SPECS:
        for _input_key in _input_keys:
            decode_inputs.pop(_input_key, None)
        if grid_key is not None:
            decode_inputs.pop(grid_key, None)
    # `mm_token_type_ids` only drives the model's internal M-RoPE (`get_rope_index`); once `position_ids`
    # is captured, the decode forward never reads it. Drop it from the decode graph's inputs so the runtime
    # (which supplies `position_ids` via `modeling_rope_utils.get_mrope_index`) needn't thread a per-step token-type tensor.
    if decode_inputs.get("position_ids") is not None:
        decode_inputs.pop("mm_token_type_ids", None)
    if decode_inputs.get("input_ids") is not None and "embed_tokens" in components:
        embedding = components["embed_tokens"][0]
        with torch.no_grad():
            embedded = embedding(decode_inputs.pop("input_ids"))
        # A decoder with per-layer embeddings returns those alongside `inputs_embeds`; both are per-token
        # inputs of the decode graph, so they go in as they come out.
        decode_inputs.update(embedded if isinstance(embedded, dict) else {"inputs_embeds": embedded})
    components["decode"] = (decode_model, decode_inputs)
    return components
