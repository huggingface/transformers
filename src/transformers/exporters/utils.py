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

Organised into sections (search for the `# ── Name ──` banners):

- **Patch and fix registries** — backend-keyed `_PATCHES` / `_FX_NODE_FIXES` /
  `_FX_PROGRAM_FIXES` populated via `@register_patch(backend, *paths)` /
  `@register_fx_node_fix` / `@register_fx_program_fix`, applied via
  `apply_patches` / `apply_fx_node_fixes` / `apply_fx_program_fixes`.
- **Recursive structure traversal** — internal helpers (`_map_leaf_tensors`,
  `_iter_leaf_tensors`) that drive every other tensor utility.
- **Public tensor utilities** — `get_leaf_tensors`, `duplicate_leaf_tensors`,
  `cast_leaf_tensors`, and `prepare_for_export` (sets attention/experts impl,
  patches non-exportable patterns, strips output flags).
- **Cross-backend patches and fixes** shared by several backends.

The inputs a model would compute with data-dependent ops are precomputed in `precompute.py`, and a generative model
is split into the graphs an export needs in `decompose.py`.
"""

from __future__ import annotations

import contextlib
import enum
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any

from packaging import version

from ..utils import logging
from ..utils.import_utils import _is_package_available, is_torch_available
from .precompute import precompute_export_inputs


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

    try:
        from torch._prims_common import is_contiguous_or_false
    except ImportError:  # `is_contiguous_or_false` was added in torch 2.9
        from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

        def is_contiguous_or_false(a: torch.Tensor) -> bool:
            # Like torch>=2.9's version, answer False instead of raising on data-dependent shapes.
            try:
                return bool(a.is_contiguous())
            except GuardOnDataDependentSymNode:
                return False

    from .. import masking_utils
    from ..modeling_utils import PreTrainedModel


# ── Patch and fix registries ────────────────────────────────────────────────
# Single contract across exporters: `_PATCHES[backend]` lists `(obj, attribute, factory)` triples
# to install reversibly, and `_FX_NODE_FIXES[backend]` lists `(gm, node) -> bool` fixers to
# apply in place. Each exporter populates its slot at module load (via `@register_patch` /
# `@register_fx_node_fix` decorators, or direct list-append for cases that can't be expressed
# as dotted paths). The export pipeline drives them via the backend-keyed helpers below.

_PATCHES: dict[str, list[tuple[str, str, callable]]] = {}
_FX_NODE_FIXES: dict[str, list[callable]] = {}
_FX_PROGRAM_FIXES: dict[str, list[callable]] = {}


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
    resolved = [
        (obj, attribute, fn)
        for obj_path, attribute, fn in _PATCHES.get(backend, [])
        if (obj := _resolve_dotted_path(obj_path)) is not None
    ]
    with patch_attributes(resolved):
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
    Paths are resolved lazily at export time (inside `apply_patches`) — submodules are
    imported as needed, falling back to `getattr` for class attributes. A path that fails
    to resolve (e.g. the backend isn't installed) is silently skipped.

    Passing multiple paths registers the SAME factory against each — useful for swapping
    the same method or torch op across several call sites (e.g. ``torch.unsqueeze`` +
    ``torch.Tensor.unsqueeze``, or one vision-attention forward across N model classes).
    """

    def decorator(fn):
        for path in paths:
            obj_path, _, attribute = path.rpartition(".")
            _PATCHES.setdefault(backend, []).append((obj_path, attribute, fn))
        return fn

    return decorator


def _resolve_dotted_path(path: str):
    """Resolve a dotted Python path to the actual object — importing submodules where
    possible, falling back to `getattr` for class attributes (e.g. `torch.Tensor`).
    Returns `None` if the path can't be resolved (e.g. the backend isn't installed)."""
    import importlib

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
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            yield from _iter_leaf_tensors(value, path)
    elif hasattr(obj, "__dict__"):
        yield from _iter_leaf_tensors(vars(obj), prefix)


def check_packages(
    owner: str, required_packages: Iterable[str], min_versions: dict[str, str], tested_versions: dict[str, str]
) -> None:
    """Raise if one of `required_packages` is missing or older than its `min_versions` entry, and warn when one differs
    from its `tested_versions` entry. Shared by the exporters and the quantizers; `owner` names the one checking.
    Missing and outdated packages are reported together, in one error."""
    installed = {}
    for package in {*required_packages, *min_versions}:
        exists, package_version = _is_package_available(package, return_version=True)
        if exists:
            # The local build suffix (`+cu126`, `+cpu`) doesn't change the API
            installed[package] = package_version.split("+", 1)[0]

    missing = [package for package in dict.fromkeys([*required_packages, *min_versions]) if package not in installed]
    outdated = [
        f"{package}>={minimum} (found {installed[package]})"
        for package, minimum in min_versions.items()
        if package in installed
        and (installed[package] == "N/A" or version.parse(installed[package]) < version.parse(minimum))
    ]
    if missing or outdated:
        fixes = []
        if missing:
            fixes.append(
                "install "
                + ", ".join(
                    f"{package}=={tested_versions[package]}" if package in tested_versions else package
                    for package in missing
                )
            )
        if outdated:
            fixes.append(f"upgrade {', '.join(outdated)}")
        raise ImportError(f"To use {owner}, please {' and '.join(fixes)}.")

    drift = [
        f"{package}: installed {installed[package]}, tested {tested.split('+', 1)[0]}"
        for package, tested in tested_versions.items()
        if package in required_packages and installed[package] not in ("N/A", tested.split("+", 1)[0])
    ]
    if drift:
        logger.warning(
            f"{owner} is experimental and patches many backend internals; behaviour may differ from what was "
            f"validated. Version drift detected — {', '.join(drift)}. If you hit issues, try the tested versions."
        )


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
    # An int `logits_to_keep` is specialized by the trace anyway, so it is fixed rather than taken as an input.
    if isinstance(inputs.get("logits_to_keep"), int):
        output_flags["logits_to_keep"] = inputs.pop("logits_to_keep")

    # Pre-compute data-dependent vision/audio tensors that use loops, .tolist(),
    # repeat_interleave, or itertools.groupby — untraceable by dynamo.
    # TODO: use the collator API once it covers these cases.
    with torch.no_grad():
        # A decomposed component (e.g. `FSMTEncoder`) may carry no config.
        if (config := getattr(model, "config", None)) is not None:
            inputs.update(precompute_export_inputs(config, inputs))

    # Cast all input tensors to match the model's dtype and device (e.g. cache objects
    # created before the model was moved to bfloat16/CUDA by a backend preparation step).
    dtype = module_dtype(model)
    device = module_device(model)
    if dtype is not None or device is not None:
        inputs = cast_leaf_tensors(inputs, dtype=dtype, device=device)

    return model, inputs, output_flags


# ── Cross-backend patches ───────────────────────────────────────────────────
# Registered against every backend that needs them, so a workaround lives in one place even when the
# backends break the op for different reasons.


@register_patch("onnx", "transformers.models.blt.modeling_blt.byte_group_hash_function")
@register_patch("openvino", "transformers.models.blt.modeling_blt.byte_group_hash_function")
def _patch_byte_group_hash(original):
    """Evaluate BLT's rolling hash in base-256 limbs, which both backends need for different reasons.

    The hash multiplies each byte of a group by ``prime ** k`` and relies on int64 semantics. ONNX
    Runtime multiplies int64 exactly but reduces through a float: ``sum`` turns `7000000049` into
    `7000000000`, losing everything below fp32's mantissa. OpenVINO's CPU plugin is narrower still —
    it executes every internal node in `i32`, so `1000000007 ** 2` saturates at `2147483647`. Either
    way the hash reads the wrong embedding rows.

    The powers are compile-time constants and the hash reaches the model only as ``hash % max_hash``,
    so it is computed here as a sum of 8-bit limbs: the single pass below carries each lane and folds
    it into the running remainder, keeping every intermediate small enough to be exact in both
    backends. Each limb is taken with `BitwiseAnd`, the only division is by 256 of a value already a
    multiple of it, and dropping the last carry is the int64 wraparound.
    """
    limb_bits = 8
    limb = 1 << limb_bits
    limb_count = 64 // limb_bits

    def patch(token_ids, group_size: int = 2, prime: int = 1000000007, max_hash: int = 30000):
        # Beyond a 16-bit table the limb products leave the window where the plugin's `%` is exact
        if max_hash > (1 << 16):
            return original(token_ids, group_size=group_size, prime=prime, max_hash=max_hash)

        powers = [pow(prime, index, 1 << 64) for index in range(group_size)]
        limbs = torch.tensor(
            [[(power >> (limb_bits * position)) % limb for power in powers] for position in range(limb_count)],
            dtype=torch.int64,
            device=token_ids.device,
        )
        padding = torch.zeros(token_ids.shape[0], group_size - 1, dtype=torch.int64, device=token_ids.device)
        windows = torch.cat([padding, token_ids.to(torch.int64)], dim=1).unfold(1, group_size, 1)
        lanes = (windows.unsqueeze(-2) * limbs).sum(-1)

        value = carry = torch.zeros_like(lanes[..., 0])
        for position in range(limb_count):
            total = lanes[..., position] + carry
            digit = torch.bitwise_and(total, limb - 1)
            carry = (total - digit) // limb
            value = (value + digit * ((1 << (limb_bits * position)) % max_hash)) % max_hash
        # Those limbs spell the *unsigned* value; eager took the remainder of a signed int64, which
        # is 2**64 lower whenever the top bit -- the last digit's -- is set.
        negative = (digit >= limb // 2).to(torch.int64)
        return (value - negative * ((1 << 64) % max_hash) + max_hash) % max_hash

    return patch


@register_patch("onnx", "torch.histc")
@register_patch("openvino", "torch.histc")
def _patch_histc(original):
    """Replace `torch.histc` with a statically-shaped, deterministic `zeros` + `scatter_add_`.

    torchlib's `aten_histc` rejects integer input and casting to float calls the nondeterministic
    `_histc_cuda`; OV has no `aten.histc` lowering at all. `bincount`, the obvious replacement, has
    an unbacked SymInt output that trips downstream meta-shape guards (grouped_mm's `offs` check).
    """

    def patch(input, bins=100, min=0, max=0, *, out=None):
        flat = input.reshape(-1)
        if max == min == 0:
            min_val = flat.min().float()
            max_val = flat.max().float()
        else:
            min_val = torch.tensor(float(min), device=flat.device)
            max_val = torch.tensor(float(max), device=flat.device)
        bin_width = (max_val - min_val) / bins
        idx = ((flat.float() - min_val) / bin_width).long().clamp_(0, bins - 1)
        out_dtype = input.dtype if input.is_floating_point() else torch.float
        counts = torch.zeros(bins, dtype=out_dtype, device=input.device)
        return counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=out_dtype))

    return patch


@register_patch("onnx", "torch.searchsorted")
@register_patch("openvino", "torch.searchsorted")
@register_patch("executorch", "torch.searchsorted")
def _patch_searchsorted(original):
    """Decompose `searchsorted` into a broadcast comparison + sum — the insertion index into a sorted
    sequence is the count of entries below. O(N*M) instead of O(M log N), but no backend has the op:
    ONNX lacks it, OV rejects the node when `sorter`/`out` trace as `None`, and the portable runtime
    ships no kernel.
    """

    def patch(sorted_sequence, values, *, out_int32=False, right=False, side=None, out=None, sorter=None):
        if side is not None:
            right = side == "right"
        if right:
            mask = sorted_sequence.unsqueeze(-1) <= values.unsqueeze(-2)
        else:
            mask = sorted_sequence.unsqueeze(-1) < values.unsqueeze(-2)
        result = mask.sum(-2)
        result = result.to(torch.int32) if out_int32 else result
        return out.copy_(result) if out is not None else result

    return patch


@register_patch("onnx", "torch.bucketize")
@register_patch("executorch", "torch.bucketize")
def _patch_bucketize(original):
    """Decompose `bucketize` into a broadcast comparison + sum — `searchsorted` with the arguments the
    other way round. ONNX's own decomposition materialises scalar constants that become
    `alias`/`detach_` and break aot's functional-graph assertion; the portable runtime has no
    `bucketize.Tensor_out` kernel.
    """

    def patch(input, boundaries, *, out_int32=False, right=False, out=None):
        if boundaries.numel() == 0:
            result = torch.zeros_like(input, dtype=torch.int64)
        else:
            below = boundaries <= input.unsqueeze(-1) if right else boundaries < input.unsqueeze(-1)
            result = below.sum(dim=-1)
        result = result.to(torch.int32) if out_int32 else result
        return out.copy_(result) if out is not None else result

    return patch


@register_patch("onnx", "transformers.masking_utils._vmap_expansion_sdpa")
@register_patch("openvino", "transformers.masking_utils._vmap_expansion_sdpa")
@register_patch("executorch", "transformers.masking_utils._vmap_expansion_sdpa")
def _patch_broadcast_mask_expansion(_original):
    """Replace vmap-based mask expansion with broadcast expansion.

    No backend traces `torch.vmap`: OV's frontend sees inputs that "escaped" the vmap context,
    and `aot_autograd`/`gen_vmap_plumbing` reject vmap-built masks under ExecuTorch's lowering.
    """

    def patch(mask_function):
        def _expanded(batch_arange, head_arange, q_arange, kv_arange):
            broadcasted = masking_utils._non_vmap_expansion_sdpa(batch_arange, head_arange, q_arange, kv_arange)
            return mask_function(*broadcasted).expand(
                batch_arange.shape[0], head_arange.shape[0], q_arange.shape[0], kv_arange.shape[0]
            )

        return _expanded

    return patch


@register_patch("onnx", "torch.reshape", "torch.Tensor.reshape", "torch.Tensor.view")
@register_patch("executorch", "torch.reshape", "torch.Tensor.reshape", "torch.Tensor.view")
def _patch_reshape(original):
    """Materialise a non-contiguous input before `reshape`/`view`.

    Both backends refuse the resulting `aten.view`, and both eat a plain `.contiguous()` (the ONNX
    optimizer folds it, functionalization drops it) — a clone survives. `contiguous_format` is not
    optional: a bare `.clone()` preserves a transposed dim-order ExecuTorch can't map to a
    `torch.memory_format`.
    """

    def patch(input, *shape, **kwargs):
        if isinstance(input, torch.Tensor) and not is_contiguous_or_false(input):
            input = input.clone(memory_format=torch.contiguous_format)
        return original(input, *shape, **kwargs)

    return patch


# ── Cross-backend FX fixes ──────────────────────────────────────────────────


@register_fx_node_fix("onnx")
@register_fx_node_fix("openvino")
def _fix_scatter_reduce(gm, node):
    """Lower ``aten.scatter_reduce.two`` at the FX level — OV's frontend has no translation,
    and its ``ScatterElementsUpdate`` op can't accept the ``reduce`` string as a constant input.

    Handles two patterns the MoE/SSM models use:
      * ``reduce="sum", include_self=True`` → ``aten.scatter_add`` (BLT/JetMoe/NemotronH router).
      * ``reduce="amax"/"amin", include_self=False`` → masked extremum over a one-hot expansion of
        ``index`` (BLT byte-pooling, tapas segment reduction).
      * ``reduce="sum"/"mean", include_self=False`` → ``scatter_add`` onto zeros, divided by a
        scattered count for the mean (tapas segment reductions).

    Other combinations fall through to the generic OpConversionFailure.
    """
    if node.target is not torch.ops.aten.scatter_reduce.two:
        return False
    if len(node.args) < 5:
        return False
    reduce = node.args[4]
    include_self = node.kwargs.get("include_self", True)
    self_arg, dim, index, src = node.args[0:4]

    if reduce == "sum" and include_self is True:
        with gm.graph.inserting_before(node):
            new = gm.graph.call_function(torch.ops.aten.scatter_add.default, args=(self_arg, dim, index, src))
            new.meta.update(node.meta)
        node.replace_all_uses_with(new)
        gm.graph.erase_node(node)
        return True

    if reduce in ("sum", "mean") and include_self is False:
        # ``include_self=False``: a position that receives at least one source element reduces over
        # *only* those elements, while a position nothing scatters to keeps ``self``. Scattering onto
        # zeros gives the former, and scattering ones alongside counts the contributors — which both
        # divides the mean and says which positions were touched at all.
        self_val = self_arg.meta.get("val")
        src_val = src.meta.get("val")
        if self_val is None or src_val is None:
            return False
        with gm.graph.inserting_before(node):
            zeros = gm.graph.call_function(torch.ops.aten.zeros_like.default, args=(self_arg,))
            sums = gm.graph.call_function(torch.ops.aten.scatter_add.default, args=(zeros, dim, index, src))
            ones = gm.graph.call_function(torch.ops.aten.ones_like.default, args=(src,))
            counts = gm.graph.call_function(torch.ops.aten.scatter_add.default, args=(zeros, dim, index, ones))
            values = sums
            if reduce == "mean":
                # clamped so untouched positions divide by 1 instead of 0 — `where` discards them anyway
                divisor = gm.graph.call_function(torch.ops.aten.clamp_min.default, args=(counts, 1))
                values = gm.graph.call_function(torch.ops.aten.div.Tensor, args=(sums, divisor))
            # OV's frontend has no ``gt.Scalar`` translation, so compare against a 0-dim tensor
            zero_tensor = gm.graph.call_function(
                torch.ops.aten.scalar_tensor.default,
                args=(0,),
                kwargs={"dtype": src_val.dtype, "device": src_val.device},
            )
            touched = gm.graph.call_function(torch.ops.aten.gt.Tensor, args=(counts, zero_tensor))
            result = gm.graph.call_function(torch.ops.aten.where.self, args=(touched, values, self_arg))
            result.meta.update(node.meta)
        node.replace_all_uses_with(result)
        gm.graph.erase_node(node)
        return True

    if reduce in ("amax", "amin") and include_self is False:
        # ``amax``/``amin`` with ``include_self=False``: each source element competes for the extremum
        # at ``index[j]``; positions no source scatters to keep ``self``'s original value. Decompose to
        # a broadcast comparison + reduction: build a one-hot mask ``(index.unsqueeze(dim) ==
        # arange(K))``, reduce ``src`` where the mask is set (the opposite extreme elsewhere, so it
        # never wins), then fall back to ``self`` for positions with no scatter.
        self_val = self_arg.meta.get("val")
        src_val = src.meta.get("val")
        if self_val is None or src_val is None or not src_val.dtype.is_floating_point:
            return False
        ndim = self_val.ndim
        d = dim if dim >= 0 else dim + ndim
        k_size = self_val.shape[d]
        # the identity for the reduction: an element that never wins
        finfo = torch.finfo(src_val.dtype)
        fill_value = finfo.min if reduce == "amax" else finfo.max
        reduction = torch.ops.aten.amax.default if reduce == "amax" else torch.ops.aten.amin.default
        k_shape = [1] * (ndim + 1)
        k_shape[d] = -1
        with gm.graph.inserting_before(node):
            # ``k_size`` is symbolic under dynamic shapes (e.g. BLT's ``max_num_patches``); baking
            # the ``SymInt`` as an ``arange`` literal makes OV decode it as a malformed inlined
            # constant. Feed the dimension through a ``sym_size`` node so it stays a real Range input.
            arange_size = (
                k_size
                if isinstance(k_size, int)
                else gm.graph.call_function(torch.ops.aten.sym_size.int, args=(self_arg, d))
            )
            arange = gm.graph.call_function(
                torch.ops.aten.arange.default, args=(arange_size,), kwargs={"device": self_val.device}
            )
            k_range = gm.graph.call_function(torch.ops.aten.view.default, args=(arange, k_shape))
            index_unsq = gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(index, d))
            mask = gm.graph.call_function(torch.ops.aten.eq.Tensor, args=(index_unsq, k_range))
            src_unsq = gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(src, d))
            # OV's frontend has no ``where.ScalarOther`` translation, so materialise the scalar
            # branches as 0-dim tensors and use ``where.self`` (broadcasts the same way).
            scalar_kwargs = {"dtype": src_val.dtype, "device": src_val.device}
            fill_tensor = gm.graph.call_function(
                torch.ops.aten.scalar_tensor.default, args=(fill_value,), kwargs=scalar_kwargs
            )
            masked = gm.graph.call_function(torch.ops.aten.where.self, args=(mask, src_unsq, fill_tensor))
            extrema = gm.graph.call_function(reduction, args=(masked, [d + 1]))
            any_match = gm.graph.call_function(torch.ops.aten.any.dim, args=(mask, d + 1))
            result = gm.graph.call_function(torch.ops.aten.where.self, args=(any_match, extrema, self_arg))
            result.meta.update(node.meta)
        node.replace_all_uses_with(result)
        gm.graph.erase_node(node)
        return True

    return False
