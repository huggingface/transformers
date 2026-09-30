"""Driving a `torch.export` program: `DynamoModelRunner`."""

from __future__ import annotations

import json

from ..utils.import_utils import is_torch_available
from .base import ModelRunner
from .metadata import (
    EXPORT_METADATA_KEY,
    ExportMetadata,
)
from .utils import (
    get_leaf_tensors,
)


if is_torch_available():
    import torch


class DynamoModelRunner(ModelRunner):
    """`ModelRunner` backed by a `torch.export` unlifted module (`ExportedProgram.module()`).

    Kwargs pass straight through (the cache stays a `Cache` pytree); outputs are flattened to named tensor leaves."""

    def __init__(self, module, export_metadata=None):
        self._module = module
        # `graph_module.meta` survives `.module()`, but not `torch.export.save`.
        self.export_metadata = self.resolve_metadata(
            export_metadata,
            lambda: ExportMetadata.from_dict(getattr(module, "meta", {}).get(EXPORT_METADATA_KEY)),
        )
        # The pytree spec, not the metadata: the module also requires baked scalars (`max_seqlen`).
        self.input_names = tuple(module._in_spec.child(1).context)
        weight = next(self._module.parameters(), None)
        if weight is not None:
            self.device, self.dtype = weight.device, weight.dtype

    def to(self, device) -> DynamoModelRunner:
        """Move the unlifted module."""
        self._module.to(device)
        weight = next(self._module.parameters(), None)
        self.device = weight.device if weight is not None else torch.device(device)
        return self

    @classmethod
    def from_artifact(cls, artifact, export_metadata=None, device=None, **kwargs) -> DynamoModelRunner:
        """Run an `ExportedProgram` straight from memory, moved to `device` if given."""
        module = artifact.module()
        if device is not None:
            module = module.to(device)
        return cls(module, export_metadata=export_metadata, **kwargs)

    @classmethod
    def from_pretrained(cls, path, export_metadata=None, device=None, **kwargs) -> DynamoModelRunner:
        """Load a saved `.pt2` and unlift it, restoring the metadata the exporter parked in `extra_files`."""
        extra_files = {EXPORT_METADATA_KEY: ""}
        exported_program = torch.export.load(str(path), extra_files=extra_files)
        payload = extra_files.get(EXPORT_METADATA_KEY)
        if payload:
            exported_program.graph_module.meta[EXPORT_METADATA_KEY] = json.loads(payload)
        return cls.from_artifact(exported_program, export_metadata=export_metadata, device=device, **kwargs)

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        # A prefill graph traced before the cache existed rejects `past_key_values`.
        if "past_key_values" not in self.input_names:
            kwargs.pop("past_key_values", None)
        return get_leaf_tensors(self._module(**kwargs))
