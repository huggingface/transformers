"""Driving an ExecuTorch program: `ExecutorchModelRunner`."""

from __future__ import annotations

import re
from pathlib import Path

from ..utils.import_utils import is_torch_available
from .base import ModelRunner
from .cache import _cache_tensors
from .metadata import (
    EXPORT_METADATA_KEY,
    ExportMetadata,
)
from .utils import (
    get_leaf_tensors,
)


if is_torch_available():
    import torch


def _executorch_constant(value):
    """A constant slot's value as ExecuTorch can take it.

    The graph specialized on the constant, so the slot is vestigial — it declares a position but the value is
    already baked in. A type the runtime cannot represent as an `EValue` (a `str`: llava_next's
    `vision_feature_select_strategy`) therefore goes in as `None` rather than failing the call."""
    return value if value is None or isinstance(value, (int, float, bool, torch.Tensor)) else None


def _feed_mismatches(method, feed: tuple, names: list[str]) -> list[str]:
    """Slots whose fed tensor cannot fit the method's declared signature: wrong rank, wrong dtype, or an
    axis past its bound. ExecuTorch reports every one of these as a bare `execute() failed with error 0x12`
    without naming the argument, which makes an otherwise one-line shape bug unattributable.
    """
    metadata = method.metadata
    mismatches = []
    for index, (name, value) in enumerate(zip(names, feed)):
        if not isinstance(value, torch.Tensor) or index >= metadata.num_inputs():
            continue
        declared = metadata.input_tensor_meta(index)
        sizes, dtype = tuple(declared.sizes()), declared.dtype()
        reason = None
        if len(sizes) != value.ndim:
            reason = f"rank {value.ndim}, declared rank {len(sizes)}"
        elif isinstance(dtype, torch.dtype) and dtype != value.dtype:
            reason = f"dtype {value.dtype}, declared {dtype}"
        else:
            over = [axis for axis, (got, bound) in enumerate(zip(value.shape, sizes)) if got > bound]
            if over:
                reason = f"shape {tuple(value.shape)} exceeds declared {sizes} on axis {over}"
        if reason is not None:
            mismatches.append(f"  - {name} (slot {index}): {reason}")
    return mismatches


def _baked_export_metadata(program) -> str | None:
    """The export metadata baked into a `.pte`, or `None` for a program exported without one."""
    try:
        return program.load_method(EXPORT_METADATA_KEY).execute(())[0]
    except Exception:
        return None


class ExecutorchModelRunner(ModelRunner):
    """`ModelRunner` backed by a loaded ExecuTorch runtime program
    (`Runtime.get().load_program(...)`). Unlike ONNX, a `.pte` carries no `input.`/`output.` convention:
    inputs are **positional** (in the source graph's flat order) and `execute` returns the lowering's
    mutated-input copies first, the model's own outputs last. A loaded method's metadata carries only counts
    and tensor shapes, so the exporter bakes the input names and the user-output count into the `.pte`
    itself as one metadata constant method (`build_export_metadata`) — the program is self-describing, exactly
    like an ONNX session or a `torch.export` module, and this runner needs nothing else. Cache inputs are
    the `past_key_values*` ones, each named by its flat leaf index; the model's outputs are
    `[logits, *cache_updates]`, in cache-input order.

    Multi-token decode works: XNNPACK needs a *bounded* dynamic sequence dim, which `_fix_range_constraints`
    already supplies (it caps unbounded `Dim.AUTO` extents), so one graph serves prefill and decode.
    """

    # Cache inputs are matched by name; the model's own outputs come back under the names the trace recorded
    # (`logits`, `past_key_values.layers.0.keys`, …), the same mapping the other backends return.

    def __init__(self, program, export_metadata=None):
        self._method = program.load_method("forward")
        # A `.pte` binds inputs positionally and reports only counts and shapes, so everything about what it
        # takes and returns comes from the metadata the exporter baked in.
        self.export_metadata = self.resolve_metadata(
            export_metadata, lambda: ExportMetadata.from_json(_baked_export_metadata(program))
        )
        self._output_names = self.export_metadata.output_names
        # The model's own outputs are the LAST that many: the lowering emits its mutated-input copies first.
        total_outputs = self._method.metadata.num_outputs()
        recorded_user_outputs = self.export_metadata.num_user_outputs
        num_user_outputs = total_outputs if recorded_user_outputs is None else recorded_user_outputs
        self._user_output_indices = range(total_outputs - num_user_outputs, total_outputs)
        # Per cache kwarg, the leaf inputs it declares. Matched exactly (`<kwarg>_<N>`) rather than by
        # prefix, so two caches whose names share one cannot claim each other's leaves.
        # What the method declares, for choosing among the names a pytree kwarg could go in under.
        self._session_input_names = set(self.input_names)
        self._cache_names = {
            cache_input: [name for name in self.input_names if re.fullmatch(rf"{re.escape(cache_input)}_\d+", name)]
            for cache_input in self.cache_inputs
        }
        # Same contract as the other runners': a graph that took a *dict* of masks declares one input per
        # attention type, so the generation loop has the ranks to build it rather than assuming a single mask.

    @classmethod
    def from_artifact(cls, artifact, export_metadata=None, device=None, **kwargs) -> ExecutorchModelRunner:
        """Load an in-memory `ExecutorchProgramManager` through its serialized buffer, which is what the
        runtime accepts — there is no path to hand it."""
        from executorch.runtime import Runtime, Verification

        program = Runtime.get().load_program(artifact.buffer, verification=Verification.Minimal)
        return cls(program, export_metadata=export_metadata, **kwargs)

    @classmethod
    def from_pretrained(cls, path, export_metadata=None, device=None, **kwargs) -> ExecutorchModelRunner:
        """Load a saved `.pte` into the ExecuTorch runtime. `Verification.Minimal` matches what the export
        tests load with — full verification walks the whole program and buys nothing here, since the file
        was just written by us."""
        from executorch.runtime import Runtime, Verification

        program = Runtime.get().load_program(Path(path), verification=Verification.Minimal)
        return cls(program, export_metadata=export_metadata, **kwargs)

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        for cache_input, declared in self._cache_names.items():
            cache = kwargs.pop(cache_input, None)
            if cache is None:
                continue
            # Cache inputs are named by flat leaf index (`<kwarg>_<N>`) — index rather than zip,
            # since the lowering may have pruned leaves it left unused (sliding caches' scalars).
            leaves = _cache_tensors(cache)
            kwargs.update({name: leaves[int(name.rsplit("_", 1)[-1])] for name in declared})
        # Remaining non-tensor kwargs are pytrees (`encoder_outputs`, mask dicts) — the graph names their
        # leaves by underscore-joined path (`encoder_outputs_last_hidden_state`).
        for name in [n for n, v in kwargs.items() if not isinstance(v, torch.Tensor)]:
            value = kwargs.pop(name)
            leaves = get_leaf_tensors(value)
            # Leaf names by path (`encoder_outputs_last_hidden_state`), with a positional fallback
            # (`image_0`): a container the graph flattened by *index* rather than by attribute — an
            # ImageList holding one `.tensor` — declares the slot under a name no path spells.
            for index, (leaf, tensor) in enumerate(leaves.items()):
                by_path = f"{name}_{leaf.replace('.', '_')}"
                kwargs[by_path if by_path in self._session_input_names else f"{name}_{index}"] = tensor
        # Positional bind, so every declared slot is filled. A slot the trace recorded as a *constant* (a
        # `None` mask entry it kept) takes that value whatever the feed carries — the program baked it, so a
        # caller offering a tensor there is offering one the graph never had.
        constants = self.export_metadata.constant_inputs
        feed = tuple(
            _executorch_constant(constants[name]) if name in constants else kwargs[name].contiguous()
            for name in self.input_names
        )
        try:
            outputs = self._method.execute(feed)
        except RuntimeError as error:
            mismatches = _feed_mismatches(self._method, feed, self.input_names)
            if not mismatches:
                raise
            raise RuntimeError(f"{error}\nFed inputs the method does not accept:\n" + "\n".join(mismatches)) from error
        return dict(zip(self._output_names, (outputs[i] for i in self._user_output_indices)))
