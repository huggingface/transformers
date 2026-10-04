"""Driving an ExecuTorch program: `ExecutorchModelRunner`."""

from __future__ import annotations

import re
from pathlib import Path

from ..utils.import_utils import is_torch_available
from .base import ModelRunner
from .cache import _cache_tensors, _read_cache_entry
from .metadata import ExportMetadata
from .utils import get_leaf_tensors


if is_torch_available():
    import torch


def _executorch_constant(value):
    """A constant slot's value as ExecuTorch can take it; the value is baked in, so a non-`EValue` is `None`."""
    return value if value is None or isinstance(value, (int, float, bool, torch.Tensor)) else None


def _feed_mismatches(method, feed: tuple, names: list[str]) -> list[str]:
    """Slots whose fed tensor does not fit the method's signature (rank, dtype, bound).

    ExecuTorch reports all of these as a bare `execute() failed with error 0x12`, without naming the argument.
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


class ExecutorchModelRunner(ModelRunner):
    """`ModelRunner` backed by a loaded ExecuTorch program (`Runtime.get().load_program(...)`).

    A `.pte` binds inputs positionally and reports only counts and shapes; names come from the recorded export
    metadata. Cache inputs are named by flat leaf index (`past_key_values_<N>`).
    """

    def __init__(self, program, export_metadata=None):
        self._method = program.load_method("forward")
        self.export_metadata = ExportMetadata.from_dict(export_metadata)
        self._output_names = self.export_metadata.output_names
        # The model's own outputs are the LAST that many: the lowering emits its mutated-input copies first.
        total_outputs = self._method.metadata.num_outputs()
        recorded_user_outputs = self.export_metadata.num_user_outputs
        num_user_outputs = total_outputs if recorded_user_outputs is None else recorded_user_outputs
        self._user_output_indices = range(total_outputs - num_user_outputs, total_outputs)
        self._session_input_names = set(self.input_names)
        # Matched exactly so two caches whose names share a prefix cannot claim each other's leaves.
        self._cache_names = {
            cache_input: [name for name in self.input_names if re.fullmatch(rf"{re.escape(cache_input)}_\d+", name)]
            for cache_input in self.cache_inputs
        }
        # Cache outputs the export left unplanned (`alloc_graph_output=False`), bindable via `Method.set_output`.
        can_bind = hasattr(self._method, "set_output")
        self._cache_outputs = [
            (index, kwarg, path.split("."))
            for index, name in zip(self._user_output_indices, self._output_names)
            for kwarg, _, path in [name.partition(".")]
            if can_bind and path and kwarg in self.cache_inputs and not _is_memory_planned(self._method, index)
        ]

    @classmethod
    def from_artifact(cls, artifact, export_metadata=None, device=None, **kwargs) -> ExecutorchModelRunner:
        """Load an in-memory `ExecutorchProgramManager` through its serialized buffer."""
        return cls._load(artifact.buffer, export_metadata, **kwargs)

    @classmethod
    def from_pretrained(cls, path, export_metadata=None, device=None, **kwargs) -> ExecutorchModelRunner:
        """Load a saved `.pte`."""
        return cls._load(Path(path), export_metadata, **kwargs)

    @classmethod
    def _load(cls, source, export_metadata, **kwargs) -> ExecutorchModelRunner:
        from executorch.runtime import Runtime, Verification

        program = Runtime.get().load_program(source, verification=Verification.Minimal)
        return cls(program, export_metadata=export_metadata, **kwargs)

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        caches = {}
        for cache_input, declared in self._cache_names.items():
            cache = kwargs.pop(cache_input, None)
            if cache is None:
                continue
            caches[cache_input] = cache
            # By index, not zip: the lowering may have pruned unused leaves (sliding caches' scalars).
            leaves = _cache_tensors(cache)
            kwargs.update({name: leaves[int(name.rsplit("_", 1)[-1])] for name in declared})
        for name in [n for n, v in kwargs.items() if not isinstance(v, torch.Tensor)]:
            value = kwargs.pop(name)
            leaves = get_leaf_tensors(value)
            # Pytree leaves by underscore-joined path, else by position (`image_0`) for containers flattened by index.
            for index, (leaf, tensor) in enumerate(leaves.items()):
                by_path = f"{name}_{leaf.replace('.', '_')}"
                kwargs[by_path if by_path in self._session_input_names else f"{name}_{index}"] = tensor
        # A slot recorded as a constant takes that value whatever the feed carries: the program baked it.
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
        """Point each cache output at the cache tensor it updates, so the step writes the cache in place."""
        for index, kwarg, path in self._cache_outputs:
            tensor = _read_cache_entry(caches[kwarg], path) if kwarg in caches else None
            if isinstance(tensor, torch.Tensor) and tensor.is_contiguous():
                self._method.set_output(tensor, index)


def _is_memory_planned(method, index: int) -> bool:
    """Whether output `index` lives in the planned arena, where no caller tensor can be bound."""
    try:
        return method.metadata.output_tensor_meta(index).is_memory_planned()
    except Exception:
        return True
