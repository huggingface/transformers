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
"""Runs an MLX export: its graph rebuilt as MLX ops on every call, compiled once per input shape. Sizes are read from the
arrays the call gets, so one export runs on any shape its graph allows."""

from __future__ import annotations

from typing import Any

from ...utils.import_utils import is_mlx_available, is_torch_available
from ..metadata import ExportMetadata, load_treespec
from ..mlx_ops import MLXProgram, run_graph, to_mlx, to_torch
from .base import ModelRunner


if is_torch_available():
    import torch

if is_mlx_available():
    import mlx.core as mx


class MLXModelRunner(ModelRunner):
    """Runs an [`MLXProgram`] on its tensor inputs flattened as `torch.export` traced them, compiled by `mx.compile`
    once per input shape. Outputs are torch views of MLX's memory.

    While `generate` runs, the runner keeps the cache as its own state, as mlx-lm does: the arrays the graph updates
    (`program.graph["mutations"]`) stay in MLX from one step to the next, where nothing else holds them, so MLX writes
    them in place, and `generate` keeps its own cache object, whose lengths it advances itself."""

    def __init__(self, program: MLXProgram, metadata: ExportMetadata, device=None):
        super().__init__(metadata)
        self.in_spec = load_treespec(metadata.in_spec)
        # As mlx-lm generates: on a stream of its own, which other MLX work never queues behind, and on a Mac with the
        # weights wired, so they stay in GPU memory between steps
        self.stream = mx.new_stream(device or mx.default_device())
        self.output_device = self.device
        if mx.metal.is_available():
            mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        # Each distinct output computed and converted once: a mutated cache is also returned in the cache
        names = program.graph["outputs"]
        outputs = list(dict.fromkeys(names))
        self.output_positions = [outputs.index(name) for name in names]
        # The inputs the graph updates (the cache), each with the position of its new value among the outputs, which
        # come first in the program's
        inputs = program.graph["inputs"]
        self.state_positions = {
            inputs.index(name): outputs.index(names[index])
            for index, name in enumerate(program.graph.get("mutations", []))
        }
        self.state: dict[int, mx.array] = {}
        graph, weights = {**program.graph, "outputs": outputs}, program.weights
        self.function = lambda *arrays: run_graph(graph, weights, arrays)
        # A graph reading values on the host (`.item()`, sizes that depend on the data) runs op by op
        if not any(node["target"] == "aten._local_scalar_dense.default" for node in graph["nodes"]):
            self.function = mx.compile(self.function)

    @property
    def device(self) -> torch.device:
        """Where the outputs come back: the device the model was exported from, so that `generate` runs as it would on
        the model (its tensors, its cache). Host memory, which Apple Silicon's GPU shares, makes them views of MLX's;
        another device (an export traced on CUDA) gets copies, and one this machine lacks falls back to the CPU."""
        device = torch.device(self.metadata.export_device or "cpu")
        available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
        return device if available.get(device.type, device.type == "cpu") else torch.device("cpu")

    @property
    def stateful(self) -> bool:
        return bool(self.state_positions)

    @classmethod
    def from_program(cls, program: MLXProgram, metadata: ExportMetadata, device=None) -> MLXModelRunner:
        """`device` is MLX's: `"gpu"`, `"cpu"`, or `None` for its default (the GPU where there is one)."""
        return cls(program, metadata, None if device is None else mx.Device(getattr(mx, str(device))))

    def reset_state(self) -> None:
        """Start the cache over, from the one the next call passes."""
        self.state = {}

    def load_state(self, other: MLXModelRunner) -> None:
        """Take over the cache `other` holds: the one a prompt graph left, for the decode graph to continue from."""
        self.state = dict(other.state)

    def reorder_state(self, beam_idx: torch.Tensor) -> None:
        """Beam search's reorder of the cache, along its batch axis (its lengths have none)."""
        rows = to_mlx(beam_idx)
        self.state = {
            index: mx.take(array, rows, axis=0) if array.ndim > 0 and array.shape[0] == rows.size else array
            for index, array in self.state.items()
        }

    def __call__(self, share_inputs: bool = True, share_outputs: bool = False, **kwargs) -> Any:
        """`share_outputs`, which `generate` passes, keeps the cache as state: the call reads it from the last one's
        instead of from the cache passed, which keeps its old values, and returns that cache as it got it."""
        leaves = torch.fx._pytree.tree_flatten_spec(((), {**self.metadata.in_constants, **kwargs}), self.in_spec)
        tensors = [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]
        state, self.state = self.state if share_outputs else {}, {}
        arrays = [state.pop(index) if index in state else to_mlx(tensor) for index, tensor in enumerate(tensors)]
        with mx.stream(self.stream):
            outputs = self.function(*arrays)
            # Held by the graph alone from here on, the cache is written in place
            del arrays
            if share_outputs:
                self.state = {index: outputs[position] for index, position in self.state_positions.items()}
                stale = {position: tensors[index] for index, position in self.state_positions.items()}
            else:
                stale = {}
            mx.eval(outputs)
            outputs_torch = [
                stale[position] if position in stale else to_torch(output).to(self.output_device)
                for position, output in enumerate(outputs)
            ]
        result = self.unflatten([outputs_torch[position] for position in self.output_positions])
        if share_outputs and kwargs.get("cache") is not None and "cache" in result:
            result["cache"] = kwargs["cache"]
        return result
