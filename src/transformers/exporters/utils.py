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
import enum
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
        # `!= ""` rather than falsiness, and `str(key)` below: a dict keyed by *integers* (granite4_vision
        # keys its deepstack features by layer index) hands down a path of `0` for the first entry, which is
        # falsy — so a truthiness test renamed that leaf "output", a name the graph never declared, and its
        # real one went missing from the feed ("Required inputs (['deepstack_features.0']) are missing").
        # Only bites a container flattened at the top level: nested under a name the path is already a
        # non-empty string.
        yield prefix if prefix != "" else "output", obj
    elif isinstance(obj, (list, tuple, set)):
        for index, item in enumerate(obj):
            path = f"{prefix}.{index}" if prefix else str(index)
            yield from _iter_leaf_tensors(item, path)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_leaf_tensors(value, path)
    elif hasattr(obj, "__dict__"):
        yield from _iter_leaf_tensors(vars(obj), prefix)


# ── Export metadata ───────────────────────────────────────────────────────
# What an exported artifact *is*, written into the artifact so a runner never has to work it out.


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


def duplicate_leaf_tensors(obj: Any, seen: set[int] | None = None) -> Any:
    """Clone tensors that appear more than once in an output structure.

    When a model returns the same tensor under two output names (e.g. `last_hidden_state`
    and `hidden_states[0]`), the ONNX optimizer deduplicates the two output nodes and
    renames one, breaking the expected name mapping. Cloning duplicates gives each output
    leaf a distinct identity so the optimizer has nothing to merge.

    `seen` pre-seeds the identities that already count as taken — pass the *input* tensors so an output the
    model hands straight back is cloned too. Returned unmutated, an input is the very value the graph's
    placeholder holds, so the pair collapses to one name and the output's wins: prophetnet's decode returns
    its `encoder_outputs.last_hidden_state` as `encoder_last_hidden_state`, and its cross-attention cache
    (filled at prefill, untouched at decode) comes back under the name it went in with — both leaving the
    input name the runtime feeds absent from the session. A *mutated* input is a distinct value by then, so
    this only adds a copy where the graph would otherwise have lost a name.
    """
    seen = set() if seen is None else set(seen)

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
        # A decomposed component is a plain `nn.Module` and need not carry a config: an encoder-decoder's
        # `FSMTEncoder`, an RNN-T's `ParakeetRNNTDecoder`. Nothing to precompute from, so skip it — the
        # data-dependent tensors below are all config-derived.
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


def _lays_out_modality_spans(config: Any) -> bool:
    """Whether `config` describes a model that places modality spans, rather than the text model inside it.

    A model's text sub-config is declared in the same module as the multi-modal config it belongs to, so
    reaching the module is not enough: a component exported from the language model alone carries the text
    config, and the spans are not its to lay out. Declaring a vision or audio sub-config is what separates
    the two — including for an omni thinker, which lays out spans under its own config class rather than
    the outer model's.
    """
    return bool({"vision_config", "audio_config"} & set(getattr(config, "sub_configs", {}) or ()))


def _rope_index_owner(config: Any):
    """The class that defines `get_rope_index` for `config`'s model, or `None` if none does.

    A module can hold more than one (qwen3_omni_moe's thinker and talker each define their own). The class
    whose own `config_class` this is wins; failing that, the config's declared architecture picks, and
    failing that the first definition.
    """
    if not _lays_out_modality_spans(config):
        return None
    try:
        module = _resolve_modeling_module(config)
    except ImportError:
        return None
    owners = [obj for obj in vars(module).values() if inspect.isclass(obj) and "get_rope_index" in obj.__dict__]
    if len(owners) <= 1:
        return owners[0] if owners else None
    for owner in owners:
        if isinstance(config, getattr(owner, "config_class", ()) or ()):
            return owner
    for architecture in getattr(config, "architectures", None) or ():
        for base in getattr(getattr(module, architecture, None), "__mro__", ()):
            if base in owners:
                return base
    return owners[0]


def get_rope_index_from_config(config: Any, inputs: Mapping[str, Any]):
    """The model's own `get_rope_index`, run without the model: `(position_ids, rope_deltas)` or `None`.

    M-RoPE lays its modality spans out per architecture, and that layout lives on the model class
    (`Qwen2VLModel.get_rope_index` and its counterparts). Both callers here hold a config and nothing else
    — the export precompute, and `ExportedGenerator` driving a saved artifact — and the method reads its
    geometry off `self.config` alone, so it runs on an instance built without `__init__`: no module tree,
    no weights, no checkpoint.

    `None` means the positions are not this function's to build: the model defines no `get_rope_index`, or
    the inputs it places spans from are absent. That is the same gate the model's own forward applies, and
    it leaves the caller on the standard 1-D positions.
    """
    owner = _rope_index_owner(config)
    if owner is None or inputs.get("input_ids") is None:
        return None
    parameters = inspect.signature(owner.get_rope_index).parameters

    # The parameter names are the model's, the keys are the processor's; `audio_seqlens` is the one the
    # omni thinkers derive from the mel padding mask rather than receiving outright.
    candidates = dict(inputs)
    candidates.setdefault("second_per_grids", inputs.get("video_second_per_grid"))
    if inputs.get("audio_feature_lengths") is not None:
        candidates.setdefault("audio_seqlens", inputs["audio_feature_lengths"])
    elif inputs.get("feature_attention_mask") is not None:
        candidates.setdefault("audio_seqlens", inputs["feature_attention_mask"].sum(-1))
    call_kwargs = {name: value for name, value in candidates.items() if name in parameters and value is not None}

    # `attention_mask` here means the 2-D padding mask the layouts index positions with. `generate`
    # carries the per-layer form instead (a dict, or a `BlockMask`), which is not that, and some layouts
    # index the mask unconditionally with no `None` branch (the omni thinkers) — so anything that is not
    # the 2-D mask becomes the all-valid one, which is what an absent mask meant here all along.
    if "attention_mask" in parameters:
        mask = call_kwargs.get("attention_mask")
        if not (isinstance(mask, torch.Tensor) and mask.dim() == 2):
            call_kwargs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    # Spans are placed from a grid or from audio lengths; with none of them present there is nothing to
    # lay out — a text-only prompt through a multi-modal model lands here.
    if not {"image_grid_thw", "video_grid_thw", "audio_seqlens"} & call_kwargs.keys():
        return None
    # There is multi-modal data but nothing saying which tokens it covers. The model raises here rather
    # than guessing, and so do we: falling back to 1-D positions would run and be quietly wrong.
    if "mm_token_type_ids" in parameters and "mm_token_type_ids" not in call_kwargs:
        raise ValueError(
            "Multi-modal data was passed but `mm_token_type_ids` is missing, so the M-RoPE positions "
            f"{owner.__name__} expects cannot be built. Pass the `mm_token_type_ids` the processor returns "
            "alongside `input_ids`."
        )

    model = owner.__new__(owner)
    object.__setattr__(model, "config", config)
    # Most layouts read `self.config` alone, but a few reach for a value the model's `__init__` copies off
    # it (the omni thinkers' `spatial_merge_size`). Fill those in as the method asks for them; a name the
    # config does not carry is a genuine error and re-raises, as does one filling did not fix.
    while True:
        try:
            return owner.get_rope_index(model, **call_kwargs)
        except AttributeError as missing:
            name = getattr(missing, "name", None)
            value = _find_config_attr(config, name) if name else None
            if value is None or hasattr(model, name):
                raise
            object.__setattr__(model, name, value)


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


@register_export_input_preparer("input_features", "feature_attention_mask")
def _prepare_masked_omni_audio_inputs(config: Any, inputs: dict[str, Any]) -> None:
    """The Omni `get_audio_features` seam carries the padded features and their padding mask rather than
    the packed `feature_lens` pair — pack them the way the getter's own masking branch does (eagerly,
    outside the trace) and hand the packed pair to `_prepare_omni_audio_inputs`; its precompute rides in
    as extra graph inputs while the graph keeps taking the raw features and mask.

    Unlike `feature_lens`, this marker pair is not omni-specific (qwen2_audio carries it too), so fire
    only for a model whose own modeling module has the chunked-audio helpers."""
    if not hasattr(_resolve_modeling_module(config), "chunk_and_pad_features"):
        return
    mask = inputs["feature_attention_mask"]
    derived = dict(inputs)
    derived["input_features"] = inputs["input_features"].permute(0, 2, 1)[mask.bool()].permute(1, 0)
    derived["feature_lens"] = mask.sum(-1)
    _prepare_omni_audio_inputs(config, derived)
    for key in ("padded_feature", "chunk_lengths", "cu_seqlens", "valid_indices", "pool_indices", "max_seqlen"):
        if key in derived:
            inputs[key] = derived[key]


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
    - Outer LLM M-RoPE positions, via [`get_rope_index_from_config`] — the model's own `get_rope_index`,
      called without the model.
    - Per-encoder preparer dispatched by marker kwargs present in `inputs` (e.g. `grid_thw`,
      `target_sizes`, `(input_features, feature_lens)`) — see `register_export_input_preparer`.
      A preparer fires only when every one of its markers is present in `inputs`.
    """
    inputs = dict(inputs)

    # Outer-model M-RoPE positions. Placing the spans reads the token ids, so this is a no-op on
    # encoder-only components (an exported `get_image_features`) that carry no `input_ids`, and on a
    # text-only model, whose class defines no `get_rope_index`.
    if inputs.get("position_ids") is None and inputs.get("input_ids") is not None:
        attn_mask = inputs.get("attention_mask")
        is_prefill = attn_mask is None or inputs["input_ids"].shape[1] == attn_mask.shape[1]
        if is_prefill and (rope_index := get_rope_index_from_config(config, inputs)) is not None:
            inputs["position_ids"] = rope_index[0]

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
        # A model-specific cache may keep its state in fields of its own rather than a `layers` list
        # (xLSTM's `rnn_state`); there is nothing layer-shaped to fill in that case.
        if not hasattr(cache_half, "layers"):
            continue
        # `Cache.early_initialization` is the API's own answer to "export needs everything in advance": it
        # feeds each layer a rank-4 zero-length hint, which is exactly the shape a growing layer must be given
        # (its own lazy init would make a rank-1 empty and bake a rank-1 guard). Let it do the layers, and let
        # a cache with state of its own size that state by overriding it (`MiniMaxCache.linear_cache`).
        geometry = [
            kv_geometry.get(index) or _cache_kv_geometry(config, index) for index in range(len(cache_half.layers))
        ]
        if all(entry is not None for entry in geometry) and geometry:
            cache_half.early_initialization(
                batch_size,
                [entry[0] for entry in geometry],
                [entry[1] for entry in geometry],
                dtype,
                device,
                value_head_dim=[entry[2] for entry in geometry],
            )
        _materialize_layers(cache_half, batch_size, config, dtype, device, kv_geometry)


def _materialize_layers(cache, batch_size, config, dtype, device, kv_geometry) -> None:
    """`materialize_cache_layers` for one flat cache — see there."""
    for layer_idx, layer in enumerate(cache.layers):
        # Every tensor the layer built in `__init__` goes to the target device first, before any skip
        # below can `continue` past it. A layer relies on its own lazy initialization to move these, which
        # either never runs (the growing branch below skips it — its rank-1 empties would bake a rank-1
        # guard) or moves only the counters it knows about: `StaticLayer.lazy_initialization` moves
        # `cumulative_length`, not a sparse-index layer's `idx_cumulative_length` (minimax_m3_vl), and
        # `early_initialization` marks the layer initialized before either gets the chance. Left behind,
        # they are cpu leaves among cuda ones and the graph's input spec mismatches on device. Dtypes stay
        # as they are — those counters are `long`, not the cache dtype.
        for attribute, value in vars(layer).items():
            if isinstance(value, torch.Tensor) and attribute not in ("keys", "values"):
                setattr(layer, attribute, value.to(device))
        # A sparse-indexer layer (deepseek_v32, axk2, glm_moe_dsa) caches a *third* tensor beside keys and
        # values, and it is a graph input like the others — leave it lazy and every later cache leaf shifts
        # by one. Its own `lazy_initialization` covers only the main K/V, so this cannot wait behind the
        # `is_initialized` skip below: `early_initialization` marks the layer initialized while the indexer
        # (and its cpu `indexer_cumulative_length` counter) is still untouched. Rank-3 zero-length for the
        # same reason the others are rank-4: the indexer's lazy init makes a 1-D empty, which would bake a
        # rank-1 guard into the graph.
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
        else:
            layer.lazy_initialization(empty_keys, empty_values)


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
            # The signature differs by model: gemma4 takes `(input_ids, inputs_embeds)` with no defaults,
            # gemma3n only `(input_ids)`. Pass the ids, and the embeds slot only if there is one.
            takes_embeds = len(inspect.signature(self.model.get_per_layer_inputs).parameters) > 1
            per_layer_inputs = (
                self.model.get_per_layer_inputs(per_layer_ids, None)
                if takes_embeds
                else (self.model.get_per_layer_inputs(per_layer_ids))
            )
            return {"inputs_embeds": inputs_embeds, "per_layer_inputs": per_layer_inputs}
