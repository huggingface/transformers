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

import os
import tempfile
import weakref
from typing import Any

from ...utils.import_utils import is_onnxruntime_available, is_onnxscript_available, is_torch_available
from ..metadata import ExportMetadata
from ..utils import onnx_inputs
from .base import ModelRunner


if is_torch_available():
    import torch

if is_onnxruntime_available():
    import onnxruntime as ort

if is_onnxscript_available():
    import onnx_ir


# Past this much weight a graph opens from a file rather than serialized: protobuf caps a model at 2GB (torch's own
# threshold, leaving room for the graph itself).
_SERIALIZED_WEIGHTS_LIMIT = int(1.5 * 2**30)


def _execution_providers(device=None) -> list:
    """The ONNX Runtime providers for `device`: CUDA when asked for, or by default when this build and a GPU have it."""
    available = ort.get_available_providers()
    if device is None:
        device = "cuda" if torch.cuda.is_available() and "CUDAExecutionProvider" in available else "cpu"
    device = torch.device(device)
    if device.type == "cpu":
        return ["CPUExecutionProvider"]
    if device.type == "cuda" and "CUDAExecutionProvider" in available:
        return [("CUDAExecutionProvider", {"device_id": device.index or 0}), "CPUExecutionProvider"]
    raise ValueError(f"This ONNX Runtime build cannot run on {device} (providers: {available}).")


class OnnxModelRunner(ModelRunner):
    """Runs an ONNX graph in an ONNX Runtime session opened for the device, its inputs and outputs bound in place."""

    def __init__(self, metadata: ExportMetadata, session: ort.InferenceSession):
        super().__init__(metadata)
        self.session = session

        options = session.get_provider_options()
        self.replays_cuda_graph = (
            options.get("CUDAExecutionProvider", {}).get("enable_cuda_graph") == "1"
            or options.get("TensorrtExecutionProvider", {}).get("trt_cuda_graph_enable") == "1"
        )

        # ORT hands a bool output over DLPack as uint8.
        self.bool_outputs = [spec.type == "tensor(bool)" for spec in session.get_outputs()]

        # The CUDA graph's binding and input buffers, made on the first call.
        self.cuda_graph: tuple[ort.IOBinding, dict[str, torch.Tensor]] | None = None

    @property
    def device(self) -> torch.device:
        provider, options = self.session.get_providers()[0], self.session.get_provider_options()
        if provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
            return torch.device("cuda", int(options[provider]["device_id"]))
        return torch.device("cpu")

    @classmethod
    def from_program(
        cls,
        program: onnx_ir.Model,
        metadata: ExportMetadata,
        device=None,
        providers: list | None = None,
        session_options: ort.SessionOptions | None = None,
    ) -> OnnxModelRunner:
        """A loaded graph opens its saved file (`meta["path"]`), ONNX Runtime reading the weights from beside it; a
        fresh one opens serialized, or from a temporary file past protobuf's size limit. `providers`, ONNX Runtime's
        own list (TensorRT with its options, say), replaces the ones picked for `device`; `session_options` open the
        session (`intra_op_num_threads`: ONNX Runtime sizes its thread pool from every core of the machine, not the
        ones the process may run on)."""
        if device is not None and providers is not None:
            raise ValueError(
                "Pass either `device` or `providers`: the providers choose the device (`device_id` in the CUDA or "
                "TensorRT provider's options)."
            )
        providers = providers or _execution_providers(device)
        if "path" in program.meta:
            return cls(metadata, ort.InferenceSession(program.meta["path"], session_options, providers=providers))

        weights = sum(
            value.const_value.nbytes for value in program.graph.initializers.values() if value.const_value is not None
        )
        if weights < _SERIALIZED_WEIGHTS_LIMIT:
            serialized = onnx_ir.to_proto(program).SerializeToString()
            return cls(metadata, ort.InferenceSession(serialized, session_options, providers=providers))
        directory = tempfile.TemporaryDirectory()
        path = os.path.join(directory.name, "model.onnx")
        onnx_ir.save(program, path, external_data="model.onnx.data")
        session = ort.InferenceSession(path, session_options, providers=providers)
        # The CPU provider maps the weights file for as long as the session lives.
        weakref.finalize(session, directory.cleanup)
        return cls(metadata, session)

    def __call__(self, share_inputs: bool = True, share_outputs: bool = False, **kwargs) -> Any:
        """Run with the inputs and outputs bound in place on the session's device (IO binding), instead of copied
        through host arrays, so `share_inputs` and `share_outputs` change nothing (outside a CUDA graph's replay).
        Inputs bind by name, the leaf's path (with an `input.` prefix on a mutated one): the export prunes the ones the
        graph doesn't read (a linear-attention layer's mask)."""
        device = self.device
        on_gpu = device.type == "cuda"
        names = [spec.name for spec in self.session.get_inputs()]
        inputs = onnx_inputs(names, kwargs)
        missing = [name for name in names if name not in inputs]
        if missing:
            raise ValueError(f"The graph takes {missing}, which {list(kwargs)} don't hold.")
        if self.replays_cuda_graph:
            return self._replay_cuda_graph(inputs, share_outputs)
        binding, bound_buffers = self.session.io_binding(), []
        for name, leaf in inputs.items():
            # Integers bind on host: ORT reads them on its CPU, where a GPU pointer is an illegal access.
            on_host = not (on_gpu and leaf.is_floating_point())
            tensor = (leaf.cpu() if on_host else leaf.to(device)).contiguous()
            # ORT refuses the null pointer of an empty tensor; a one-element stand-in serves.
            buffer = tensor if tensor.numel() else torch.empty(1, dtype=tensor.dtype, device=tensor.device)
            bound_buffers.append(buffer)
            _bind_input(binding, name, tensor, buffer.data_ptr())
        for spec in self.session.get_outputs():
            binding.bind_output(spec.name, device.type, device.index or 0)
        # ORT runs on its own stream: sync both ways, or it reads half-written inputs.
        if on_gpu:
            torch.cuda.current_stream(device).synchronize()
        self.session.run_with_iobinding(binding)
        if on_gpu:
            binding.synchronize_outputs()
        return self.unflatten(self._read_outputs(binding))

    def _replay_cuda_graph(self, inputs: dict[str, torch.Tensor], share_outputs: bool) -> Any:
        """Run a session that replays a CUDA graph (`enable_cuda_graph`, `trt_cuda_graph_enable`). The graph reads and
        writes the device memory it was captured with, so its input buffers (integers on the device too) and the
        outputs ONNX Runtime allocated on its first run are kept: the inputs are copied in, and the outputs copied out
        unless `share_outputs`. Capture needs a static export, whose inputs keep one shape: a dynamic one computes its
        shapes in the graph."""
        device = self.device
        first_run = self.cuda_graph is None
        if first_run:
            binding = self.session.io_binding()
            buffers = {name: leaf.to(device).contiguous().clone() for name, leaf in inputs.items()}
            for name, buffer in buffers.items():
                _bind_input(binding, name, buffer, buffer.data_ptr())
            for spec in self.session.get_outputs():
                binding.bind_output(spec.name, "cuda", device.index)
            self.cuda_graph = (binding, buffers)
        binding, buffers = self.cuda_graph
        for name, buffer in buffers.items():
            leaf = inputs[name]
            # `copy_` would broadcast a smaller input into the captured buffer.
            if leaf.shape != buffer.shape:
                raise ValueError(
                    f"The CUDA graph was captured with `{name}` of shape {tuple(buffer.shape)}, got {tuple(leaf.shape)}."
                )
            buffer.copy_(leaf)
        # ORT runs on its own stream: sync both ways, or it reads half-written inputs.
        torch.cuda.current_stream(device).synchronize()
        self.session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        if first_run:
            # The graph writes where this run's outputs were allocated: they stay bound for every replay.
            for spec, value in zip(self.session.get_outputs(), binding.get_outputs()):
                binding.bind_ortvalue_output(spec.name, value)
        outputs = self._read_outputs(binding)
        return self.unflatten(outputs if share_outputs else [output.clone() for output in outputs])

    def _read_outputs(self, binding: ort.IOBinding) -> list[torch.Tensor]:
        """The bound outputs as torch tensors on ORT's memory."""
        outputs = [torch.from_dlpack(value._ortvalue.to_dlpack()) for value in binding.get_outputs()]
        return [output.view(torch.bool) if is_bool else output for output, is_bool in zip(outputs, self.bool_outputs)]


def _bind_input(binding: ort.IOBinding, name: str, tensor: torch.Tensor, data_ptr: int) -> None:
    """Bind input `name` to the memory at `data_ptr`, read as `tensor`'s dtype and shape on its device."""
    binding.bind_input(
        name,
        tensor.device.type,
        tensor.device.index or 0,
        int(onnx_ir.tensor_adapters.from_torch_dtype(tensor.dtype)),
        list(tensor.shape),
        data_ptr,
    )
