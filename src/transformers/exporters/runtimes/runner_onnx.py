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

from ...utils.import_utils import is_onnxruntime_available, is_torch_available
from ..metadata import ExportMetadata
from ..utils import get_leaf_tensors
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
    """Runs an `ONNXProgram` in an ONNX Runtime session. A loaded program's session opens the saved file; a fresh
    one opens the in-memory graph, as `ONNXProgram` itself does."""

    def __init__(self, session: ort.InferenceSession, metadata: ExportMetadata, program: ONNXProgram | None = None):
        super().__init__(metadata)
        self.session = session
        # Kept alive: a program past ~1.5GB opens its session on a temporary copy the program owns.
        self.program = program
        self.input_names = [node.name for node in session.get_inputs()]
        self.output_names = [node.name.removeprefix("output.") for node in session.get_outputs()]

    @classmethod
    def from_program(cls, program: ONNXProgram, metadata: ExportMetadata, device=None) -> OnnxModelRunner:
        sessions = []

        def open_session(model):
            sessions.append(ort.InferenceSession(model, providers=_providers(device)))
            return sessions[-1]

        program.initialize_inference_session(open_session)
        return cls(sessions[0], metadata, program)

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        leaves = get_leaf_tensors(kwargs)
        feed = {name: leaves[name].detach().cpu().numpy() for name in self.input_names}
        outputs = self.session.run(None, feed)
        return {name: torch.from_numpy(value) for name, value in zip(self.output_names, outputs)}
