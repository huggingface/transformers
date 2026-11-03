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
"""Runs an OpenVINO model, compiled for a device."""

from __future__ import annotations

from typing import Any

from ...utils.import_utils import is_openvino_available, is_torch_available
from ..metadata import ExportMetadata
from ..quantizers.utils import openvino_feed
from .base import ModelRunner


if is_torch_available():
    import torch

if is_openvino_available():
    import openvino


def _openvino_device(device=None) -> str:
    """The OpenVINO device for a torch-style one: `AUTO` by default, `CPU`, or `GPU` for an Intel GPU (`xpu`)."""
    if device is None:
        return "AUTO"
    device = torch.device(device)
    if device.type == "cpu":
        return "CPU"
    if device.type == "xpu":
        return "GPU" if device.index is None else f"GPU.{device.index}"
    raise ValueError(f"OpenVINO cannot run on {device}; pass `cpu`, `xpu` (Intel GPU), or no device for `AUTO`.")


def openvino_output(value, port) -> torch.Tensor:
    """An inference result as a torch tensor. OpenVINO's Python API returns a bf16 output as a float16 array of the
    bf16 bits (NumPy has no bf16), which read as float16 give wrong values: reinterpret those bits as bf16."""
    if port.get_element_type() == openvino.Type.bf16:
        return torch.from_numpy(value.view("int16")).view(torch.bfloat16)
    return torch.from_numpy(value)


class OpenVINOModelRunner(ModelRunner):
    """Runs an `openvino.Model`, compiled for a device, through one infer request."""

    def __init__(self, compiled_model: openvino.CompiledModel, metadata: ExportMetadata):
        super().__init__(metadata)
        self.compiled_model = compiled_model
        self.request = compiled_model.create_infer_request()

    @property
    def device(self) -> torch.device:
        # Inputs and outputs live in host memory whichever device OpenVINO compiled for.
        return torch.device("cpu")

    @classmethod
    def from_program(
        cls, program: openvino.Model, metadata: ExportMetadata, device=None, config: dict | None = None
    ) -> OpenVINOModelRunner:
        """`config`, OpenVINO compile properties, goes over the CPU defaults, which keep the export's precision:
        inference and KV cache in its dtype (OpenVINO otherwise keeps a stateful cache in uint8, and may run fp32 in
        bf16 on CPUs that have it), tuned for latency."""
        device = _openvino_device(device)
        defaults = {}
        if device == "CPU":
            precision = {"float32": "f32", "float16": "f16", "bfloat16": "bf16"}.get(metadata.dtype, "f32")
            defaults = {
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_PRECISION_HINT": precision,
                "KV_CACHE_PRECISION": precision,
            }
        return cls(openvino.Core().compile_model(program, device, {**defaults, **(config or {})}), metadata)

    def reset_state(self) -> None:
        """Start the state a stateful model keeps (its cache) over, for a new sequence."""
        self.request.reset_state()

    def __call__(self, share_inputs: bool = True, share_outputs: bool = False, **kwargs) -> Any:
        feed = openvino_feed(self.compiled_model, kwargs)
        outputs = self.request.infer(feed, share_inputs=share_inputs, share_outputs=share_outputs)
        return self.unflatten([openvino_output(outputs[port], port) for port in self.compiled_model.outputs])
