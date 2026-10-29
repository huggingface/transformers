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


def _output_name(port) -> str:
    """A port's readable name: not OV's own `<node>:<port>` or bare-id names, which compiling can merge into it."""
    names = sorted(port.get_names())
    given = [name for name in names if ":" not in name and not name.isdigit()]
    return (given[0] if given else names[0]).removeprefix("output.")


class OpenVINOModelRunner(ModelRunner):
    """Runs an `openvino.Model`, compiled for a device, through one infer request."""

    def __init__(self, compiled_model: openvino.CompiledModel, metadata: ExportMetadata):
        super().__init__(metadata)
        self.compiled_model = compiled_model
        self.request = compiled_model.create_infer_request()
        self.output_names = [_output_name(port) for port in compiled_model.outputs]

    @classmethod
    def from_program(cls, program: openvino.Model, metadata: ExportMetadata, device=None) -> OpenVINOModelRunner:
        return cls(openvino.Core().compile_model(program, _openvino_device(device)), metadata)

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        outputs = self.request.infer(openvino_feed(self.compiled_model, kwargs))
        # Copied: the request writes its next outputs into the same buffers.
        return {
            name: torch.from_numpy(outputs[port].copy())
            for name, port in zip(self.output_names, self.compiled_model.outputs)
        }
