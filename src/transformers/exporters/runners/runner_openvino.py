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

import os
from typing import Any

from ...utils.import_utils import is_openvino_available, is_torch_available
from ..metadata import ExportMetadata
from ..utils import openvino_feed
from .base import ModelRunner


if is_torch_available():
    import torch

if is_openvino_available():
    import openvino


def _openvino_device_name(device=None) -> str:
    """The OpenVINO device for a torch-style one: `AUTO` by default, `CPU`, or `GPU` for an Intel GPU (`xpu`)."""
    if device is None:
        return "AUTO"
    device = torch.device(device)
    if device.type == "cpu":
        return "CPU"
    if device.type == "xpu":
        return "GPU" if device.index is None else f"GPU.{device.index}"
    raise ValueError(f"OpenVINO cannot run on {device}; pass `cpu`, `xpu` (Intel GPU), or no device for `AUTO`.")


def _physical_cores() -> int:
    """The physical cores this process may run on: its CPU affinity, one logical CPU per core (OpenVINO schedules on
    physical cores)."""
    if not hasattr(os, "sched_getaffinity"):  # macOS: no affinity or topology to read
        return os.cpu_count() or 1
    cpus = os.sched_getaffinity(0)
    cores = set()
    for cpu in cpus:
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list", encoding="utf-8") as siblings:
                cores.add(siblings.read().strip())
        except OSError:
            cores.add(str(cpu))
    return len(cores)


def openvino_output_tensor(value, port) -> torch.Tensor:
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
        cls,
        program: openvino.Model,
        metadata: ExportMetadata,
        device=None,
        config: dict | None = None,
        torch_threads: int | None = None,
    ) -> OpenVINOModelRunner:
        """`config`, OpenVINO compile properties, goes over the CPU defaults (tuned for latency) and over the precision
        the export recorded on the model.

        `torch_threads` splits the CPU between torch and OpenVINO, for torch work around the model (pre- and
        post-processing, `generate`'s loop): torch keeps that many threads and OpenVINO runs pinned on the remaining
        physical cores. Both thread pools otherwise span every core, and torch's, which spins for a while after each op,
        slows the OpenVINO call next to it (EmbeddingGemma2 on 56 Xeon cores, torch pooling after each call: 69.5 ms
        with both pools on every core, 50.1 ms with 8 torch threads, OpenVINO alone 49.5 ms)."""
        device = _openvino_device_name(device)
        defaults = {}
        if device == "CPU":
            defaults = {"PERFORMANCE_HINT": "LATENCY"}
            if torch_threads is not None:
                torch.set_num_threads(torch_threads)
                defaults["INFERENCE_NUM_THREADS"] = max(1, _physical_cores() - torch_threads)
                defaults["ENABLE_CPU_PINNING"] = True
        return cls(openvino.Core().compile_model(program, device, {**defaults, **(config or {})}), metadata)

    def __call__(self, share_inputs: bool = True, share_outputs: bool = False, **kwargs) -> Any:
        feed = openvino_feed(self.compiled_model, kwargs)
        outputs = self.request.infer(feed, share_inputs=share_inputs, share_outputs=share_outputs)
        return self.unflatten([openvino_output_tensor(outputs[port], port) for port in self.compiled_model.outputs])
