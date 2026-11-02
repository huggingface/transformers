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
    """Runs an `ONNXProgram` in an ONNX Runtime session opened for the device, its inputs and outputs bound in place.
    A loaded program's session opens the saved file; a fresh one opens the in-memory graph."""

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
        """Run with the inputs and outputs bound in place on the session's device (IO binding), instead of copied
        through host arrays. The graph binds inputs by position, so they go in traced order."""
        from torch.onnx._internal.exporter._core import torch_dtype_to_onnx_dtype

        inputs = {name: kwargs[name] for name in self.metadata.kwargs if name in kwargs}
        device = self.device
        on_gpu = device.type == "cuda"
        # The tensor leaves: `torch.export` bakes Python scalars (`output_router_logits=False`) into the graph.
        leaves = [leaf for leaf in torch.utils._pytree.tree_leaves(inputs) if isinstance(leaf, torch.Tensor)]
        specs = self.session.get_inputs()
        if len(leaves) != len(specs):
            raise ValueError(
                f"The graph takes {len(specs)} inputs, but got {len(leaves)} tensors from {list(inputs)}."
            )
        binding, kept = self.session.io_binding(), []
        for spec, leaf in zip(specs, leaves):
            # Integers bind on host: ORT reads them on its CPU, where a GPU pointer is an illegal access.
            on_host = not (on_gpu and leaf.is_floating_point())
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
            binding.bind_output(spec.name, device.type, device.index or 0)
        # ORT runs on its own stream: sync both ways, or it reads half-written inputs.
        if on_gpu:
            torch.cuda.current_stream(device).synchronize()
        self.session.run_with_iobinding(binding)
        if on_gpu:
            binding.synchronize_outputs()
        outputs = [torch.from_dlpack(value._ortvalue.to_dlpack()) for value in binding.get_outputs()]
        return self.unflatten(outputs)
