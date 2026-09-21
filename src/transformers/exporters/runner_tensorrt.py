"""Driving a TensorRT-compiled `torch.export` program: `TensorrtModelRunner`."""

from __future__ import annotations

from pathlib import Path

from .runner_dynamo import DynamoModelRunner


class TensorrtModelRunner(DynamoModelRunner):
    """`ModelRunner` for a graph whose convertible parts are TensorRT engines.

    Everything it does is [`DynamoModelRunner`]'s: the artifact is an `ExportedProgram`, and an engine is a
    `tensorrt.execute_engine` node inside it. The one difference is the import below -- that op is
    registered by importing `torch_tensorrt`, and without it a load fails on an operator it has never
    heard of rather than on anything to do with the model.

    Conversion folds every weight into an engine, so this is the backend whose program has no parameter to
    read a device off. It does not need one: the export recorded where it ran, and `ModelRunner.device`
    falls back to that.
    """

    @classmethod
    def from_pretrained(cls, path: str | Path, **kwargs) -> TensorrtModelRunner:
        import torch_tensorrt  # noqa: F401  (registers `tensorrt.execute_engine`)

        return super().from_pretrained(path, **kwargs)
