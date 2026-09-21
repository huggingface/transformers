"""Driving an AOTInductor package: `AotiModelRunner`."""

from __future__ import annotations

import io
from pathlib import Path

from ..utils.import_utils import is_torch_available
from .base import ModelRunner
from .metadata import EXPORT_METADATA_KEY, ExportMetadata
from .utils import get_leaf_tensors


if is_torch_available():
    import torch
    from torch.utils import _pytree as pytree


def _load_package(source, device):
    """Load a compiled package, on `device` when one is asked for.

    Which GPU is the caller's to choose — the kernels are the same on any of them — so a device index goes
    to the loader. Which *kind* of device is not: that was decided at compile time, and `__init__` refuses
    the mismatch rather than letting it surface inside the first call.
    """
    index = torch.device(device).index if device is not None else None
    return torch._inductor.aoti_load_package(source, device_index=-1 if index is None else index)


class AotiModelRunner(ModelRunner):
    """`ModelRunner` backed by a loaded AOTInductor package — the compiled counterpart of
    [`DynamoModelRunner`]'s unlifted module.

    It is called exactly as that one is: kwargs go in against the graph's own pytree spec (the KV cache
    stays a `Cache` object), and the output is flattened to its named tensor leaves. There is no `to`:
    the package holds kernels compiled for one device, so the base class's refusal is the honest answer.
    """

    def __init__(self, compiled, export_metadata=None, device=None):
        self._compiled = compiled
        # Baked into the package at compile time, so a runner built from a file in hand knows what one
        # built straight from the export knows — the device it was compiled for among it.
        self.export_metadata = self.resolve_metadata(
            export_metadata, lambda: ExportMetadata.from_json(compiled.get_metadata().get(EXPORT_METADATA_KEY))
        )
        # The package states the call it was compiled for, which is the contract here exactly as the
        # unlifted module's `_in_spec` is for `torch.export` — kwargs in the second child of `(args, kwargs)`.
        in_spec = pytree.treespec_loads(compiled.loader.get_call_spec()[0])
        self.input_names = tuple(in_spec.child(1).context)
        compiled_for = self.export_metadata.device or torch.device("cpu")
        if device is not None and torch.device(device).type != compiled_for.type:
            raise ValueError(
                f"This package holds kernels compiled for {compiled_for.type}, so it cannot run on "
                f"{device}. Export the model again with its weights on {device}."
            )
        self.device = torch.device(device) if device is not None else compiled_for

    @classmethod
    def from_artifact(cls, artifact: bytes, export_metadata=None, device=None, **kwargs) -> AotiModelRunner:
        """Load the package straight from the bytes an export handed back, without writing it out first."""
        return cls(_load_package(io.BytesIO(artifact), device), export_metadata=export_metadata, device=device)

    @classmethod
    def from_pretrained(cls, path: str | Path, export_metadata=None, device=None, **kwargs) -> AotiModelRunner:
        """Load a saved `.pt2` package."""
        return cls(_load_package(str(path), device), export_metadata=export_metadata, device=device)

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        return get_leaf_tensors(self._compiled(**kwargs))
