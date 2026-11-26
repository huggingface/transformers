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

from dataclasses import dataclass
from typing import Any

from ...utils.import_utils import is_mlx_available, is_torch_available
from ..metadata import ExportMetadata, load_treespec
from ..mlx_ops import MLXProgram, run_graph, to_mlx, to_torch
from .base import ModelRunner


if is_torch_available():
    import torch

if is_mlx_available():
    import mlx.core as mx


def _tensor_key(tensor: torch.Tensor) -> tuple:
    return tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.dtype


def _next_mask(mask: mx.array) -> mx.array:
    """The padding or causal mask of the decode step after `mask`'s: one more slot seen."""
    if mask.ndim == 2:
        return mx.concatenate([mask, mx.ones((mask.shape[0], 1), dtype=mask.dtype)], axis=-1)
    # The last query's row, with the slot after its last seen one seen too
    row = mask[:, :, -1:, :]
    seen = mx.max(row, axis=-1, keepdims=True)
    slots = mx.arange(row.shape[-1])
    last = mx.max(mx.where(row == seen, slots, -1), axis=-1, keepdims=True)
    return mx.where(slots == last + 1, seen, row)


@dataclass
class _QueuedStep:
    """A decode step computing on the inputs greedy decoding gives it next, before `generate` asks for it."""

    inputs: list
    outputs: list


class MLXModelRunner(ModelRunner):
    """Runs an [`MLXProgram`] on its tensor inputs flattened as `torch.export` traced them, compiled by `mx.compile`
    once per input shape.

    Outputs are torch views of MLX's memory. An input that is one of the last call's outputs (the cache a decode step
    returned) runs on that MLX array as is, so state that loops through `generate` is never copied back.

    While `generate` decodes greedily, the runner queues each next step ahead, as mlx-lm does: the device computes it
    on the argmax token while `generate` picks that token on the host (`_queue_next_step`)."""

    def __init__(self, program: MLXProgram, metadata: ExportMetadata, device=None):
        super().__init__(metadata)
        self.in_spec = load_treespec(metadata.in_spec)
        # As mlx-lm generates: on a stream of its own, which other MLX work never queues behind, and on a Mac with the
        # weights wired, so they stay in GPU memory between steps
        self.stream = mx.new_stream(device or mx.default_device())
        self.output_device = self.device
        if mx.metal.is_available():
            mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        self.input_names = program.graph["inputs"]
        # Each distinct output computed and converted once: a mutated cache is also returned in the cache
        outputs = list(dict.fromkeys(program.graph["outputs"]))
        self.output_positions = [outputs.index(name) for name in program.graph["outputs"]]
        graph, weights = {**program.graph, "outputs": outputs}, program.weights
        self.function = lambda *arrays: run_graph(graph, weights, arrays)
        # A graph reading values on the host (`.item()`, sizes that depend on the data) runs op by op
        if not any(node["target"] == "aten._local_scalar_dense.default" for node in graph["nodes"]):
            self.function = mx.compile(self.function)
        # The last call's outputs, by their torch view
        self.outputs: list[mx.array] = []
        self.last_outputs: dict[tuple, int] = {}
        # Each input that loops back from an output (the cache), as `{input index: output position}`
        self.loops: dict[int, int] = {}
        self.logits_position: int | None = None
        self.queued_step: _QueuedStep | None = None
        self.queue_steps = True

    @property
    def device(self) -> torch.device:
        """Where the outputs come back: the device the model was exported from, so that `generate` runs as it would on
        the model (its tensors, its cache). Host memory, which Apple Silicon's GPU shares, makes them views of MLX's;
        another device (an export traced on CUDA) gets copies, and one this machine lacks falls back to the CPU."""
        device = torch.device(self.metadata.export_device or "cpu")
        available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
        return device if available.get(device.type, device.type == "cpu") else torch.device("cpu")

    @classmethod
    def from_program(cls, program: MLXProgram, metadata: ExportMetadata, device=None) -> MLXModelRunner:
        """`device` is MLX's: `"gpu"`, `"cpu"`, or `None` for its default (the GPU where there is one)."""
        return cls(program, metadata, None if device is None else mx.Device(getattr(mx, str(device))))

    def __call__(self, share_inputs: bool = True, share_outputs: bool = False, **kwargs) -> Any:
        leaves = torch.fx._pytree.tree_flatten_spec(((), {**self.metadata.in_constants, **kwargs}), self.in_spec)
        tensors = [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]
        looped = [self.last_outputs.get(_tensor_key(tensor)) for tensor in tensors]
        queued_step, self.queued_step = self.queued_step, None
        with mx.stream(self.stream):
            hit = queued_step is not None and self._is_queued_step(queued_step, tensors, looped)
            if hit:
                arrays, outputs = queued_step.inputs, queued_step.outputs
            else:
                arrays = [
                    to_mlx(tensor) if position is None else self.outputs[position]
                    for tensor, position in zip(tensors, looped)
                ]
                outputs = self.function(*arrays)
            mx.eval(outputs)
            # Read before the next step is queued, which reading could otherwise wait for
            outputs_torch = [to_torch(output).to(self.output_device) for output in outputs]
            self._queue_next_step(arrays, outputs, looped, missed=queued_step is not None and not hit)
        self.outputs = list(outputs)
        self.last_outputs = {_tensor_key(tensor): position for position, tensor in enumerate(outputs_torch)}
        result = self.unflatten([outputs_torch[position] for position in self.output_positions])
        if self.logits_position is None:
            logits = getattr(result, "logits", None)
            self.logits_position = next((p for p, tensor in enumerate(outputs_torch) if tensor is logits), -1)
        return result

    def _queue_next_step(self, arrays: list, outputs: list, looped: list, missed: bool) -> None:
        """Queue the decode step after this one on the inputs greedy decoding gives it: the logits' argmax as the next
        token, the next positions, the mask opening the next slot, and the outputs that loop back (the cache).

        The next call runs on it only if its inputs are those (`_is_queued_step`), so the outputs are always the ones
        of the inputs `generate` passes. After a miss (sampling, a logits processor changing the token), no step is
        queued until a fresh state starts the next `generate`."""
        if all(position is None for position in looped):
            # A fresh state starts a new `generate`, whatever the last one left queued
            self.queue_steps = True
        else:
            self.loops = {index: position for index, position in enumerate(looped) if position is not None}
            self.queue_steps = self.queue_steps and not missed
        if not self.queue_steps or not self.loops or self.logits_position in (None, -1):
            return
        inputs = []
        for index, (name, array) in enumerate(zip(self.input_names, arrays)):
            if index in self.loops:
                inputs.append(outputs[self.loops[index]])
            elif name == "input_ids":
                inputs.append(mx.argmax(outputs[self.logits_position][:, -1:], axis=-1).astype(array.dtype))
            elif name == "position_ids":
                inputs.append(array[..., -1:] + 1)
            elif name.startswith("attention_mask") and array.ndim in (2, 4):
                inputs.append(_next_mask(array))
            else:
                return
        self.queued_step = _QueuedStep(inputs, self.function(*inputs))
        mx.async_eval(self.queued_step.outputs)

    def _is_queued_step(self, queued_step: _QueuedStep, tensors: list, looped: list) -> bool:
        """Whether this call's inputs are the ones `queued_step` computes on."""
        if len(queued_step.inputs) != len(tensors):
            return False
        for index, (array, tensor, position) in enumerate(zip(queued_step.inputs, tensors, looped)):
            if index in self.loops:
                if position != self.loops[index]:
                    return False
            elif tuple(array.shape) != tuple(tensor.shape) or not torch.equal(
                to_torch(array).to(tensor.device), tensor
            ):
                return False
        return True
