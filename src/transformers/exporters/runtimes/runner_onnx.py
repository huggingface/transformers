# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Runs an ONNX program in ONNX Runtime."""

from __future__ import annotations

from typing import Any

from ...utils.import_utils import is_onnxruntime_available, is_torch_available
from ..metadata import ExportMetadata
from .base import ModelRunner


if is_torch_available():
    import torch
    from torch.onnx import ONNXProgram

if is_onnxruntime_available():
    import onnxruntime as ort


def _providers(device=None) -> list:
    """The ONNX Runtime providers for `device`: CUDA when asked for, or by default when this build has it."""
    available = ort.get_available_providers()
    if device is None:
        return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    device = torch.device(device)
    if device.type == "cpu":
        return ["CPUExecutionProvider"]
    if device.type == "cuda" and "CUDAExecutionProvider" in available:
        return [("CUDAExecutionProvider", {"device_id": device.index or 0}), "CPUExecutionProvider"]
    raise ValueError(f"This ONNX Runtime build cannot run on {device} (providers: {available}).")


class OnnxModelRunner(ModelRunner):
    """Runs an `ONNXProgram` through its own `__call__`, in an ONNX Runtime session opened for the device. A loaded
    program's session opens the saved file; a fresh one opens the in-memory graph."""

    def __init__(self, program: ONNXProgram, metadata: ExportMetadata, session: ort.InferenceSession):
        super().__init__(metadata)
        self.program = program
        self.session = session

    @property
    def device(self) -> torch.device:
        provider, options = self.session.get_providers()[0], self.session.get_provider_options()
        if provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
            return torch.device("cuda", int(options[provider]["device_id"]))
        return torch.device("cpu")

    @classmethod
    def from_program(cls, program: ONNXProgram, metadata: ExportMetadata, device=None) -> OnnxModelRunner:
        sessions = []

        def open_session(model):
            sessions.append(ort.InferenceSession(model, providers=_providers(device)))
            return sessions[-1]

        program.initialize_inference_session(open_session)
        return cls(program, metadata, sessions[0])

    def __call__(self, **kwargs) -> Any:
        # The graph binds inputs by position, so they go in traced order.
        inputs = {name: kwargs[name] for name in self.metadata.kwargs if name in kwargs}
        if self.device.type == "cuda":
            return self.unflatten(self._run_on_device(inputs))
        return self.unflatten(list(self.program(**inputs)))

    def _run_on_device(self, inputs: dict) -> list:
        """Run with the inputs and outputs left on the GPU (IO binding), instead of copied through host arrays."""
        from torch.onnx._internal.exporter._core import torch_dtype_to_onnx_dtype

        device = self.device
        # Flattened as `ONNXProgram` flattens them: pytree leaves, `None`s dropped.
        leaves = [leaf for leaf in torch.utils._pytree.tree_leaves(inputs) if leaf is not None]
        binding, kept = self.session.io_binding(), []
        for spec, leaf in zip(self.session.get_inputs(), leaves):
            leaf = leaf if isinstance(leaf, torch.Tensor) else torch.tensor(leaf)
            # Integers bind on host: ORT reads them on its CPU, where a GPU pointer is an illegal access.
            on_host = not leaf.is_floating_point()
            tensor = (leaf.cpu() if on_host else leaf.to(device)).contiguous()
            # ORT refuses the null pointer of an empty tensor; a one-element stand-in serves.
            buffer = tensor if tensor.numel() else torch.empty(1, dtype=tensor.dtype, device=tensor.device)
            kept.append(buffer)
            binding.bind_input(
                spec.name,
                "cpu" if on_host else "cuda",
                0 if on_host else device.index,
                int(torch_dtype_to_onnx_dtype(tensor.dtype)),
                list(tensor.shape),
                buffer.data_ptr(),
            )
        for spec in self.session.get_outputs():
            binding.bind_output(spec.name, "cuda", device.index)
        # ORT runs on its own stream: sync both ways, or it reads half-written inputs.
        torch.cuda.current_stream(device).synchronize()
        self.session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        return [torch.from_dlpack(value._ortvalue.to_dlpack()) for value in binding.get_outputs()]
