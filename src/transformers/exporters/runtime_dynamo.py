"""Driving a `torch.export` program: `DynamoModelRunner`."""

from __future__ import annotations

import torch

from .base import ModelRunner
from .utils import EXPORT_METADATA_KEY, ExportMetadata, get_leaf_tensors


class DynamoModelRunner(ModelRunner):
    """`ModelRunner` backed by a `torch.export` unlifted module (`ExportedProgram.module()`) — the
    runnable, like ORT's session. Kwargs pass straight through (the KV-cache stays a `Cache` pytree the
    graph consumes natively); the output object is flattened to its named tensor leaves."""

    def __init__(self, module):
        self._module = module
        # `graph_module.meta` survives `.module()`, so the exporter's account of the trace rides along here
        # exactly as it does inside an artifact — precision, cache layout, traced shapes and all.
        self.export_metadata = ExportMetadata.from_dict(getattr(module, "meta", {}).get(EXPORT_METADATA_KEY))
        self.kv_geometry = self.export_metadata.kv_geometry
        # The one thing the metadata cannot give: this module rejects any kwarg set but the one it was traced
        # with, *including* baked scalars (`max_seqlen`) that never became graph placeholders — so the
        # recorded graph inputs are too few, and its own pytree spec is the contract.
        self.input_names = tuple(module._in_spec.child(1).context)
        # Outputs land wherever the exported weights live.
        weight = next(self._module.parameters(), None)
        if weight is not None:
            self.device, self.dtype = weight.device, weight.dtype

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        # The module rejects kwargs it wasn't traced with — e.g. a prefill graph whose capture predates the
        # cache (models that create it inside the first forward); its cache still rides out on the outputs.
        if "past_key_values" not in self.input_names:
            kwargs.pop("past_key_values", None)
        output = self._module(**kwargs)
        if isinstance(output, torch.Tensor):
            return {"output": output}
        return get_leaf_tensors(output)
