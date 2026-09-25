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
    """`ModelRunner` backed by a `torch.export` unlifted module (`ExportedProgram.module()`) — the
    runnable, like ORT's session. Kwargs pass straight through (the KV-cache stays a `Cache` pytree the
    graph consumes natively); the output object is flattened to its named tensor leaves."""

    def __init__(self, module, export_metadata=None):
        self._module = module
        # `graph_module.meta` survives `.module()`, so the exporter's account of the trace rides along here
        # exactly as it does inside an artifact — precision, cache layout, traced shapes and all.
        # Passed in by a load (the saved `export_metadata.json`), else whatever the artifact carries:
        # `torch.export.save` drops `meta`, so a program that has been through disk has none of its own.
        self.export_metadata = self.resolve_metadata(
            export_metadata,
            lambda: ExportMetadata.from_dict(getattr(module, "meta", {}).get(EXPORT_METADATA_KEY)),
        )
        # The one thing the metadata cannot give: this module rejects any kwarg set but the one it was traced
        # with, *including* baked scalars (`max_seqlen`) that never became graph placeholders — so the
        # recorded graph inputs are too few, and its own pytree spec is the contract.
        self.input_names = tuple(module._in_spec.child(1).context)
        # Outputs land wherever the exported weights live.
        weight = next(self._module.parameters(), None)
        if weight is not None:
            self.device, self.dtype = weight.device, weight.dtype

    def to(self, device) -> DynamoModelRunner:
        """Move the unlifted module -- a `torch.export` program is a module, so this is the ordinary move."""
        self._module.to(device)
        weight = next(self._module.parameters(), None)
        self.device = weight.device if weight is not None else torch.device(device)
        return self

    @classmethod
    def from_artifact(cls, artifact, export_metadata=None, device=None, **kwargs) -> DynamoModelRunner:
        """Run an `ExportedProgram` straight from memory: the module is what `.module()` unlifts. `device`
        moves it, which is also where its outputs then land."""
        module = artifact.module()
        if device is not None:
            module = module.to(device)
        return cls(module, export_metadata=export_metadata, **kwargs)

    @classmethod
    def from_pretrained(cls, path, export_metadata=None, device=None, **kwargs) -> DynamoModelRunner:
        """Load a saved `.pt2` and unlift it, putting the metadata back where an in-memory program carries it.

        `torch.export.save` drops `meta` keys it does not know, so the exporter parks the payload in
        `extra_files`; reading it back onto the graph module is what makes a reloaded runner answer about
        precision and cache layout the way the one built straight from `export` does.
        """
        extra_files = {EXPORT_METADATA_KEY: ""}
        exported_program = torch.export.load(str(path), extra_files=extra_files)
        payload = extra_files.get(EXPORT_METADATA_KEY)
        if payload:
            exported_program.graph_module.meta[EXPORT_METADATA_KEY] = json.loads(payload)
        module = exported_program.module()
        if device is not None:
            module = module.to(device)
        return cls(module, export_metadata=export_metadata, **kwargs)

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        # The module rejects kwargs it wasn't traced with — e.g. a prefill graph whose capture predates the
        # cache (models that create it inside the first forward); its cache still rides out on the outputs.
        if "past_key_values" not in self.input_names:
            kwargs.pop("past_key_values", None)
        output = self._module(**kwargs)
        if isinstance(output, torch.Tensor):
            return {"output": output}
        return get_leaf_tensors(output)
