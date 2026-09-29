"""Driving an ExecuTorch program: `ExecutorchModelRunner`."""

from __future__ import annotations

import re
from pathlib import Path

from ..utils.import_utils import is_torch_available
from .base import ModelRunner
from .cache import _cache_tensors, _read_cache_entry
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
    """`ModelRunner` backed by a loaded ExecuTorch program (`Runtime.get().load_program(...)`).

    A `.pte` binds its inputs positionally and returns the lowering's mutated-input copies before the model's
    own outputs, reporting only counts and shapes; the names and the user-output count come from the metadata
    the exporter baked in as a constant method. Cache inputs are named by flat leaf index
    (`past_key_values_<N>`), and outputs come back under the names the trace recorded.
    """

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
        # What the method declares, for choosing among the names a pytree kwarg could go in under.
        self._session_input_names = set(self.input_names)
        # Per cache kwarg, the leaf inputs it declares, matched exactly (`<kwarg>_<N>`) so two caches whose
        # names share a prefix cannot claim each other's leaves.
        self._cache_names = {
            cache_input: [name for name in self.input_names if re.fullmatch(rf"{re.escape(cache_input)}_\d+", name)]
            for cache_input in self.cache_inputs
        }
        # The model's cache outputs (`past_key_values.layers.0.keys`), by their leaf path in the cache, where the
        # runtime lets us point them at our own tensors: the export left them unplanned (`alloc_graph_output=False`)
        # and the runtime can bind them (`Method.set_output`).
        can_bind = hasattr(self._method, "set_output")
        self._cache_outputs = [
            (index, kwarg, path.split("."))
            for index, name in zip(self._user_output_indices, self._output_names)
            for kwarg, _, path in [name.partition(".")]
            if can_bind and path and kwarg in self.cache_inputs and not _is_memory_planned(self._method, index)
        ]

    @classmethod
    def from_artifact(cls, artifact, export_metadata=None, device=None, **kwargs) -> ExecutorchModelRunner:
        """Load an in-memory `ExecutorchProgramManager` through its serialized buffer — the runtime takes no
        program object."""
        return cls._load(artifact.buffer, export_metadata, **kwargs)

    @classmethod
    def from_pretrained(cls, path, export_metadata=None, device=None, **kwargs) -> ExecutorchModelRunner:
        """Load a saved `.pte`."""
        return cls._load(Path(path), export_metadata, **kwargs)

    @classmethod
    def _load(cls, source, export_metadata, **kwargs) -> ExecutorchModelRunner:
        from executorch.runtime import Runtime, Verification

        # Minimal verification, as the export tests load with: a full walk of a program we just wrote buys nothing.
        program = Runtime.get().load_program(source, verification=Verification.Minimal)
        return cls(program, export_metadata=export_metadata, **kwargs)

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        caches = {}
        for cache_input, declared in self._cache_names.items():
            cache = kwargs.pop(cache_input, None)
            if cache is None:
                continue
            caches[cache_input] = cache
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
        self._bind_cache_outputs(caches)
        try:
            outputs = self._method.execute(feed)
        except RuntimeError as error:
            mismatches = _feed_mismatches(self._method, feed, self.input_names)
            if not mismatches:
                raise
            raise RuntimeError(f"{error}\nFed inputs the method does not accept:\n" + "\n".join(mismatches)) from error
        return dict(zip(self._output_names, (outputs[i] for i in self._user_output_indices)))

    def _bind_cache_outputs(self, caches: dict) -> None:
        """Point each of the model's cache outputs at the cache tensor it updates, so the step writes the cache
        where it lives and hands back those very tensors, leaving the generation loop nothing to copy. The
        runtime keeps a mutated input's own write-back on the fed tensor by itself; the model's copy of it is
        a separate output, matched here by its leaf path the way ONNX matches `output.<name>`."""
        for index, kwarg, path in self._cache_outputs:
            tensor = _read_cache_entry(caches[kwarg], path) if kwarg in caches else None
            if isinstance(tensor, torch.Tensor) and tensor.is_contiguous():
                self._method.set_output(tensor, index)


def _is_memory_planned(method, index: int) -> bool:
    """Whether output `index` lives in the method's planned arena, where no caller tensor can be bound. A
    non-tensor output has no tensor metadata and counts as planned."""
    try:
        return method.metadata.output_tensor_meta(index).is_memory_planned()
    except Exception:
        return True
