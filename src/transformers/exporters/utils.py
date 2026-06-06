# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
  `_FX_PROGRAM_FIXES` populated via `@register_patch` / `@register_fx_node_fix` /
  `@register_fx_program_fix`, applied via `apply_patches` / `apply_fx_node_fixes` /
  `apply_fx_program_fixes`.
- **Recursive structure traversal** — internal helpers (`_map_leaf_tensors`,
  `_iter_leaf_tensors`) that drive every other tensor utility.
- **Public tensor utilities** — `get_leaf_tensors`, `duplicate_leaf_tensors`,
  `cast_leaf_tensors`, and `prepare_for_export` (sets attention/experts impl,
  patches non-exportable patterns, strips output flags).
- **Export input preparers** — `@register_export_input_preparer(model_type)`
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
import inspect
from typing import Any

from ..utils import logging
from ..utils.import_utils import is_torch_available


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

    from ..modeling_utils import PreTrainedModel
    from ..vision_utils import (
        get_vision_bilinear_indices_and_weights,
        get_vision_cu_seqlens,
        get_vision_merged_shape,
        get_vision_nearest_position_ids,
        get_vision_position_ids,
        get_vision_window_index,
    )


# ── Patch and fix registries ────────────────────────────────────────────────
# Single contract across exporters: `_PATCHES[backend]` lists `(obj, attr, factory)` triples
# to install reversibly, and `_FX_NODE_FIXES[backend]` lists `(gm, node) -> bool` fixers to
# apply in place. Each exporter populates its slot at module load (via `@register_patch` /
# `@register_fx_node_fix` decorators, or direct list-append for cases that can't be expressed
# as dotted paths). The export pipeline drives them via the backend-keyed helpers below.

_PATCHES: dict[str, list[tuple[Any, str, callable]]] = {}
_FX_NODE_FIXES: dict[str, list[callable]] = {}
_FX_PROGRAM_FIXES: dict[str, list[callable]] = {}


@contextlib.contextmanager
def patch_attr(obj: Any, attr: str, factory: Any):
    """Swap `obj.attr` with `factory(original)` for the duration of the block."""
    original = getattr(obj, attr)
    setattr(obj, attr, factory(original))
    try:
        yield
    finally:
        setattr(obj, attr, original)


@contextlib.contextmanager
def patch_attrs(patches: list[tuple[Any, str, callable]]):
    """Install `(obj, attr, factory)` patches for the duration of the block.

    Plural form of `patch_attr` — each `factory(original)` returns the replacement
    callable. Originals are restored on exit, even if the body raises.
    """
    with contextlib.ExitStack() as stack:
        for obj, attr, factory in patches:
            stack.enter_context(patch_attr(obj, attr, factory))
        yield


@contextlib.contextmanager
def apply_patches(backend: str):
    """Install `_PATCHES[backend]` for the duration of the block."""
    with patch_attrs(_PATCHES.get(backend, [])):
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


def register_patch(backend: str, path: str):
    """Append the decorated `factory(original)` to `_PATCHES[backend]`.

    `path` is a dotted Python path like `"torch.where"` or `"torch.Tensor.unsqueeze"`. The
    rightmost segment is the attribute to swap; the rest is the object that owns it. The
    path is resolved at decoration time — submodules are imported lazily, falling back to
    `getattr` for class attributes. If the chain can't be resolved (e.g. the backend isn't
    installed), registration is silently skipped so the module still imports.
    """

    def decorator(fn):
        obj_path, _, attr = path.rpartition(".")
        try:
            obj = _resolve_dotted_path(obj_path)
        except (ImportError, AttributeError):
            return fn
        _PATCHES.setdefault(backend, []).append((obj, attr, fn))
        return fn

    return decorator


def _resolve_dotted_path(path: str):
    """Resolve a dotted Python path to the actual object — importing submodules where
    possible, falling back to `getattr` for class attributes (e.g. `torch.Tensor`)."""
    import importlib

    parts = path.split(".")
    obj = importlib.import_module(parts[0])
    for part in parts[1:]:
        try:
            obj = importlib.import_module(f"{obj.__name__}.{part}")
        except (ImportError, AttributeError):
            obj = getattr(obj, part)
    return obj


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

    Traverses dicts, lists, tuples, sets, and objects with `__dict__` (e.g. cache objects).
    Skips non-traversable leaf types (enum, SymInt, etc.).
    """
    if isinstance(obj, _LEAF_SKIP_TYPES):
        return obj
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(_map_leaf_tensors(item, fn) for item in obj)
    if isinstance(obj, dict):
        return type(obj)({k: _map_leaf_tensors(v, fn) for k, v in obj.items()})
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
        return tensor.to(dtype=dtype, device=device) if tensor.is_floating_point() else tensor.to(device=device)

    return _map_leaf_tensors(obj, _cast)


# Output flags that should be set on `model.config`, not passed as forward() kwargs.
_OUTPUT_FLAGS = ("use_cache", "output_attentions", "output_hidden_states", "return_dict", "return_loss")


def prepare_for_export(
    model: PreTrainedModel | torch.nn.Module, inputs: dict[str, Any]
) -> tuple[PreTrainedModel | torch.nn.Module, dict[str, Any]]:
    """Configure the model for export. Mutates both `model` and `inputs` in-place:

    - Sets optimal attention/experts implementations.
    - Patches non-exportable module behaviours (mamba masks, classifier casts, …).
    - Strips label inputs (`labels`, `future_values`) — loss computation is unsupported.
    - Strips output flags (`use_cache`, `return_dict`, …) from inputs and bakes non-`None`
      values into `model.forward` via `functools.partial` so they are constant at trace time.
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

    # Strip output flags from inputs. Set on config when possible, otherwise bake into
    # the forward via functools.partial so the value is constant at trace time.
    # Submodule captures often inject these from the parent model's forward; they must not
    # appear as traced kwargs or the exported signature will mismatch at runtime.
    for output_flag in _OUTPUT_FLAGS:
        if output_flag in inputs:
            value = inputs.pop(output_flag)
            if value is not None:
                model.forward = functools.partial(model.forward, **{output_flag: value})

    # set experts implementation to batched_mm for export
    if isinstance(model, PreTrainedModel) and model._can_set_experts_implementation():
        model.set_experts_implementation("batched_mm")

    # set attention implementation to sdpa for export
    if isinstance(model, PreTrainedModel) and model._can_set_attn_implementation():
        try:
            model.set_attn_implementation("sdpa")
        except Exception as e:
            logger.warning("Could not set attention implementation to sdpa for %s: %s", model.config.model_type, e)

    # Idefics2/3's vision encoder uses boolean indexing to filter padding images, creating
    # an unbacked symbolic batch dim. SDPA's CPU kernel guards on Eq(batch, 1) when a mask
    # is provided, which fails with unbacked dims. Keep the vision part on eager.
    if (
        isinstance(model, PreTrainedModel)
        and model._can_set_attn_implementation()
        and model.config.model_type in ("idefics2", "idefics3", "smolvlm")
        and model.device.type == "cpu"
    ):
        model.set_attn_implementation({"vision_config": "eager"})

    for module in model.modules():
        if hasattr(module, "config"):
            # disable returning loss for every submodel
            if hasattr(module.config, "return_loss"):
                module.config.return_loss = False
            # disable mamba kernels for every submodel (mamba, jamba)
            if hasattr(module.config, "use_mamba_kernels"):
                module.config.use_mamba_kernels = False
        # Reset internal caches that are not part of past_key_values (e.g. DSA indexer in glm_moe_dsa)
        if hasattr(module, "_cached_keys"):
            module._cached_keys = None

    # Pre-compute data-dependent vision/audio tensors that use loops, .tolist(),
    # repeat_interleave, or itertools.groupby — untraceable by torch.export.
    with torch.no_grad():
        _precompute_export_inputs(model, inputs)

    # Cast all input tensors to match the model's dtype and device (e.g. cache objects
    # created before the model was moved to bfloat16/CUDA by a backend preparation step).
    try:
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
        inputs = cast_leaf_tensors(inputs, dtype=model_dtype, device=model_device)
    except StopIteration:
        pass

    return model, inputs


# ── Export input preparers ────────────────────────────────────────────────────
# Registry of `model_type -> (model, inputs) -> None` callables that precompute the
# data-dependent tensors (cu_seqlens, position_ids, padded audio chunks, …) the model
# would otherwise compute in its forward via `.tolist()` / `nonzero()` / etc. Inject
# the results into `inputs` so the forward skips the untraceable branch.


def _find_submodule_attr(model: torch.nn.Module, name: str) -> Any | None:
    """Return the first non-None value of `name` found on `model` or any of its submodules."""
    for module in model.modules():
        if (value := getattr(module, name, None)) is not None:
            return value
    return None


_EXPORT_INPUT_PREPARERS: dict[str, callable] = {}


def register_export_input_preparer(*model_types: str):
    """Register `fn(model, inputs) -> None` as the export-input preparer for these model_types."""

    def decorator(fn):
        for mt in model_types:
            _EXPORT_INPUT_PREPARERS[mt] = fn
        return fn

    return decorator


@register_export_input_preparer(
    "qwen2_vl_vision",
    "qwen2_5_vl_vision",
    "qwen3_vl_vision",
    "qwen3_vl_moe_vision",
    "qwen3_5_vision",
    "qwen3_5_moe_vision",
    "qwen2_5_omni_vision_encoder",
    "qwen3_omni_moe_vision_encoder",
    "glm4v_vision",
    "glm4v_moe_vision",
    "glm46v",
    "glm_image_vision",
    "glm_ocr_vision",
    "paddleocr_vl_vision",
    "video_llama_3_vision",
    "ernie4_5_vl_moe_vision",
    "exaone4_5_vision",
)
def _prepare_grid_thw_vision_inputs(model: torch.nn.Module, inputs: dict[str, Any]) -> None:
    """Precompute helpers driven by `grid_thw`: `cu_seqlens`, `position_ids`, plus optional
    `window_index`/`cu_window_seqlens` (XNet-style window attn) and
    `bilinear_indices`/`bilinear_weights` (interpolation-based merging).

    Optional helpers are gated by the presence of their config attribute on the encoder
    (`window_size`+`patch_size` for window attention, `num_grid_per_side` for bilinear),
    so a model that doesn't use that feature won't get its kwarg injected.
    """
    grid_thw = inputs.get("grid_thw")
    if grid_thw is None:
        return

    spatial_merge_size = _find_submodule_attr(model, "spatial_merge_size")
    if spatial_merge_size is None:
        # Video-Llama-3 carries per-image merge sizes as an input tensor; PaddleOCR-VL has
        # none (its encoder hard-codes `1` because spatial merging happens in the projector).
        spatial_merge_size = inputs.get("merge_sizes", 1)

    inputs["cu_seqlens"] = get_vision_cu_seqlens(grid_thw)
    inputs["position_ids"] = get_vision_position_ids(grid_thw, spatial_merge_size)

    window_size = _find_submodule_attr(model, "window_size")
    patch_size = _find_submodule_attr(model, "patch_size")
    if window_size is not None and patch_size is not None:
        inputs["window_index"], inputs["cu_window_seqlens"] = get_vision_window_index(
            grid_thw, spatial_merge_size, window_size, patch_size
        )

    num_grid_per_side = _find_submodule_attr(model, "num_grid_per_side")
    if num_grid_per_side is not None:
        inputs["bilinear_indices"], inputs["bilinear_weights"] = get_vision_bilinear_indices_and_weights(
            grid_thw, num_grid_per_side, spatial_merge_size
        )


@register_export_input_preparer("minicpmv4_6_vision")
def _prepare_navit_vision_inputs(model: torch.nn.Module, inputs: dict[str, Any]) -> None:
    """NaViT-style packed encoders carry per-image `(h, w)` as `target_sizes` instead of `grid_thw`.
    Synthesise `grid_thw = [1, h, w]` and run the nearest-position-id / window-index /
    merged-shape helpers so the per-image Python loops move outside the traced graph."""
    target_sizes = inputs.get("target_sizes")
    if target_sizes is None:
        return
    device = target_sizes.device

    num_patches_per_side = _find_submodule_attr(model, "num_patches_per_side")
    if num_patches_per_side is not None:
        inputs["position_ids"] = get_vision_nearest_position_ids(target_sizes, num_patches_per_side).to(device)

    window_kernel_size = _find_submodule_attr(model, "window_kernel_size")
    if window_kernel_size is not None:
        grid_thw = torch.nn.functional.pad(target_sizes, (1, 0), value=1)
        window_index, cu_window_seqlens = get_vision_window_index(
            grid_thw, spatial_merge_size=1, window_size=window_kernel_size[0], patch_size=1
        )
        inputs["window_index"] = window_index.to(device)
        inputs["cu_window_seqlens"] = cu_window_seqlens.to(device)
        inputs["merged_shape"] = get_vision_merged_shape(target_sizes, window_kernel_size)


@register_export_input_preparer("qwen2_5_omni_audio_encoder")
def _prepare_qwen2_5_omni_audio_inputs(model: torch.nn.Module, inputs: dict[str, Any]) -> None:
    """Replace `input_features`/`feature_lens` with the precomputed `padded_feature`, `chunk_lengths`,
    `cu_seqlens`, `valid_indices`, `pool_indices` — so the encoder's `.split(.tolist(), dim=0)` and
    related data-dependent ops happen outside the traced graph."""
    from ..models.qwen2_5_omni.modeling_qwen2_5_omni import (
        chunk_and_pad_features,
        get_audio_cu_seqlens,
        get_pool_indices,
        get_valid_indices,
    )

    if "input_features" not in inputs or "feature_lens" not in inputs:
        return

    feature_lens = inputs.pop("feature_lens")
    input_features = inputs.pop("input_features")

    padded_feature, chunk_lengths = chunk_and_pad_features(input_features, feature_lens, model.n_window)
    inputs["padded_feature"] = padded_feature
    inputs["chunk_lengths"] = chunk_lengths
    inputs["cu_seqlens"] = get_audio_cu_seqlens(chunk_lengths)
    inputs["valid_indices"] = get_valid_indices(chunk_lengths)
    inputs["pool_indices"] = get_pool_indices(feature_lens)


@register_export_input_preparer("qwen3_omni_moe_audio_encoder")
def _prepare_qwen3_omni_moe_audio_inputs(model: torch.nn.Module, inputs: dict[str, Any]) -> None:
    """Same shape as `_prepare_qwen2_5_omni_audio_inputs` but `get_audio_cu_seqlens` takes
    `(chunk_lengths, feature_lens, n_window_infer, n_window)` instead of `(chunk_lengths,)`,
    and there is no `get_pool_indices`."""
    from ..models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        chunk_and_pad_features,
        get_audio_cu_seqlens,
        get_valid_indices,
    )

    if "input_features" not in inputs or "feature_lens" not in inputs:
        return

    feature_lens = inputs.pop("feature_lens")
    input_features = inputs.pop("input_features")

    padded_feature, chunk_lengths = chunk_and_pad_features(input_features, feature_lens, model.n_window)
    inputs["padded_feature"] = padded_feature
    inputs["chunk_lengths"] = chunk_lengths
    inputs["cu_seqlens"] = get_audio_cu_seqlens(chunk_lengths, feature_lens, model.n_window_infer, model.n_window)
    inputs["valid_indices"] = get_valid_indices(chunk_lengths)


def _precompute_export_inputs(model: torch.nn.Module, inputs: dict[str, Any]) -> None:
    """Inject precomputed tensors for data-dependent ops the model would otherwise hit during tracing.

    Two layers:
    - Outer LLM rope index (`get_rope_index`) — generic `hasattr` probe; covers Qwen-VL / GLM-4V etc.
    - Per-encoder preparer dispatched by `config.model_type` (see `register_export_input_preparer`).
    """
    # Outer-model: LLM rope index. Self-detecting via `hasattr` since model_type at this level
    # varies (qwen2_vl vs qwen2_5_omni_thinker vs ...) and the get_rope_index signature is stable.
    if inputs.get("position_ids") is None and hasattr(model, "get_rope_index"):
        input_ids = inputs.get("input_ids")
        attn_mask = inputs.get("attention_mask")
        is_prefill = attn_mask is None or input_ids is None or input_ids.shape[1] == attn_mask.shape[1]
        if is_prefill:
            rope_params = set(inspect.signature(model.get_rope_index).parameters)
            rope_inputs = {k: inputs[k] for k in rope_params if k in inputs}
            position_ids, _ = model.get_rope_index(**rope_inputs)
            inputs["position_ids"] = position_ids

    # Encoder-level: dispatch by model_type.
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if preparer := _EXPORT_INPUT_PREPARERS.get(model_type):
        preparer(model, inputs)


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


def decompose_prefill_decode(
    model: PreTrainedModel,
    inputs: dict[str, Any],
) -> dict[str, tuple[torch.nn.Module, dict]]:
    """Run `model.generate()` for 2 tokens and capture prefill and decode inputs.

    Reuses the full generation machinery so every architecture (decoder-only, SSM,
    encoder-decoder, multi-modal, …) gets correct inputs without reimplementing the loop.

    Returns:
        `dict[str, tuple[torch.nn.Module, dict]]`:
        `{"prefill": (model, prefill_inputs), "decode": (model, decode_inputs)}`
    """
    try:
        with _capture_forward(model) as calls, torch.no_grad():
            model.generate(**copy.deepcopy(inputs), max_new_tokens=2, min_new_tokens=2)
    except Exception as e:
        raise RuntimeError(
            f"decompose_prefill_decode failed for {type(model).__name__}. "
            f"Inputs passed: {list(inputs.keys())}. "
            f"Make sure the inputs are compatible with model.generate()."
        ) from e

    return {
        "prefill": (copy.copy(model), calls[0]),
        "decode": (copy.copy(model), calls[1]),
    }


# Projector attribute names — no canonical accessor on `PreTrainedModel`, kept as a heuristic.
# Encoders and language model are resolved via `get_encoder(modality)` / `get_decoder()`.
_MULTIMODAL_PROJECTOR_NAMES = ("multi_modal_projector", "connector", "embed_vision", "embed_audio")
_MULTIMODAL_LM_HEAD_NAMES = ("lm_head",)


def _find_multimodal_submodules(model: PreTrainedModel) -> dict[str, torch.nn.Module]:
    """Return `{attr_name: module}` for multi-modal submodules found on `model`.

    Uses the canonical `PreTrainedModel.get_encoder("image"/"audio")` and `get_decoder()`
    accessors for encoders and the language model. Projectors and `lm_head` are looked
    up by name on `model` and its `base_model` (e.g. `LlavaModel` under `LlavaForConditionalGeneration`).

    Only returns results when at least one modal encoder AND a language model are found —
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
        found["language_model"] = decoder

    for root in {model, model.base_model}:
        for name in _MULTIMODAL_PROJECTOR_NAMES + _MULTIMODAL_LM_HEAD_NAMES:
            if name not in found and getattr(root, name, None) is not None:
                found[name] = getattr(root, name)

    if not has_encoder or "language_model" not in found:
        return {}

    return found


def is_multimodal(model: PreTrainedModel) -> bool:
    """Returns `True` if the model is multi-modal with modal encoders and a language model."""
    return bool(_find_multimodal_submodules(model))


def decompose_multimodal(model: PreTrainedModel, inputs: dict[str, Any]) -> dict[str, tuple[torch.nn.Module, dict]]:
    """Capture inputs to each multi-modal submodule via a single forward pass.

    Detects all known multi-modal submodules by attribute name (vision tower, projector,
    language model, lm_head, …) and captures their forward kwargs during one
    `model(**inputs)` call.

    Each submodule is returned as a separate `name: (module, inputs)` entry for
    independent export. The token-merge step (e.g. `masked_scatter` for multi-modal models)
    is intentionally left outside the exported graphs — it is the caller's responsibility
    to assemble `inputs_embeds` from the encoder outputs before running the decoder.

    Returns:
        `dict[str, tuple[torch.nn.Module, dict]]`: One `name: (module, inputs)`
        entry per detected submodule (image/audio encoder, projector, language model, lm_head).

    Raises:
        `ValueError`: if no known multi-modal submodules are found on the model.
    """
    submodules = _find_multimodal_submodules(model)
    if not submodules:
        raise ValueError(
            f"decompose_multimodal found no multi-modal submodules on {type(model).__name__}. "
            f"Expected an image/audio encoder + language model, found neither."
        )

    try:
        with contextlib.ExitStack() as stack, torch.no_grad():
            submodule_inputs = {
                name: stack.enter_context(_capture_forward(module)) for name, module in submodules.items()
            }
            model(**copy.deepcopy(inputs))
    except Exception as e:
        raise RuntimeError(
            f"decompose_multimodal failed for {type(model).__name__}. Inputs passed: {list(inputs.keys())}."
        ) from e

    return {
        name: (module, submodule_inputs[name][-1])
        for name, module in submodules.items()
        if submodule_inputs[name]  # skip submodules not called (e.g. lm_head on base models)
    }


def decompose_for_generation(
    model: PreTrainedModel, inputs: dict[str, Any]
) -> dict[str, tuple[torch.nn.Module, dict]]:
    """Decompose a generative model into independently exportable `(model, forward_inputs)` pairs.

    Runs `decompose_prefill_decode` to capture prefill and decode forward kwargs from a real
    `model.generate(**inputs, max_new_tokens=2)`. If the prefill is multi-modal (per `is_multimodal`),
    further splits it into one entry per submodule (vision/audio encoder, projector, language model,
    `lm_head`) via `decompose_multimodal`.

    Args:
        model: Generative model. Must support `model.generate(**inputs)`.
        inputs: **Generate** kwargs — what you'd pass to `model.generate(**inputs)`.

    Returns:
        `{component_name: (submodel, forward_inputs)}`. Keys are `"prefill"` / `"decode"` for
        plain generative models and `"<modality>_encoder"` / `"multi_modal_projector"` /
        `"language_model"` / `"lm_head"` / `"decode"` for multi-modal generative models.
    """
    stages = decompose_prefill_decode(model, inputs)
    prefill_model, prefill_inputs = stages["prefill"]

    if not is_multimodal(prefill_model):
        return stages

    components = decompose_multimodal(prefill_model, prefill_inputs)
    components["decode"] = stages["decode"]
    return components
