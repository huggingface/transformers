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


def _tensor_key(tensor: torch.Tensor) -> tuple:
    return tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.dtype


class MLXModelRunner(ModelRunner):
    """Runs an [`MLXProgram`] on its tensor inputs flattened as `torch.export` traced them, compiled by `mx.compile`
    once per input shape.

    Outputs are torch views of MLX's memory. An input that is one of the last call's outputs (the cache a decode step
    returned) runs on that MLX array as is, so state that loops through `generate` is never copied back."""

    def __init__(self, program: MLXProgram, metadata: ExportMetadata, device=None):
        super().__init__(metadata)
        self.in_spec = load_treespec(metadata.in_spec)
        self.mlx_device = device
        # Each distinct output computed and converted once: a mutated cache is also returned in the cache
        outputs = list(dict.fromkeys(program.graph["outputs"]))
        self.output_positions = [outputs.index(name) for name in program.graph["outputs"]]
        graph, weights = {**program.graph, "outputs": outputs}, program.weights
        self.function = mx.compile(lambda *arrays: run_graph(graph, weights, arrays))
        self.input_names = program.graph["inputs"]
        self.last_outputs: dict[tuple, int] = {}
        self.outputs: list[mx.array] = []
        # Speculative decoding of the next step (see `_speculate`): which output each input loops back from, where the
        # logits are, and the step queued on the predicted inputs
        self.loops: dict[int, int] = {}
        self.logits_position: int | None = None
        self.speculation: tuple[list, list] | None = None
        self.speculating = True

    @property
    def device(self) -> torch.device:
        # Inputs and outputs cross through host memory, which Apple Silicon's GPU shares
        return torch.device("cpu")

    @classmethod
    def from_program(cls, program: MLXProgram, metadata: ExportMetadata, device=None) -> MLXModelRunner:
        """`device` is MLX's: `"gpu"`, `"cpu"`, or `None` for its default (the GPU where there is one)."""
        return cls(program, metadata, None if device is None else mx.Device(getattr(mx, str(device))))

    def __call__(self, share_inputs: bool = True, share_outputs: bool = False, **kwargs) -> Any:
        leaves = torch.fx._pytree.tree_flatten_spec(((), {**self.metadata.in_constants, **kwargs}), self.in_spec)
        tensors = [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]
        looped = [self.last_outputs.get(_tensor_key(tensor)) for tensor in tensors]
        speculation, self.speculation = self.speculation, None
        with mx.stream(self.mlx_device or mx.default_device()):
            if speculation is not None and self._predicted(speculation[0], tensors, looped):
                arrays, outputs = speculation
            else:
                if speculation is not None:
                    self.speculating = False
                arrays = [
                    to_mlx(tensor) if position is None else self.outputs[position]
                    for tensor, position in zip(tensors, looped)
                ]
                outputs = self.function(*arrays)
            mx.eval(outputs)
            # Read before the next step is queued, which reading could otherwise wait for
            outputs_torch = [to_torch(output) for output in outputs]
            if all(position is None for position in looped):
                # A fresh state (a new `generate`): speculate again
                self.speculating = True
            else:
                self.loops = {index: position for index, position in enumerate(looped) if position is not None}
            self.speculation = self._speculate(arrays, outputs)
            if self.speculation is not None:
                mx.async_eval(self.speculation[1])
        self.outputs = list(outputs)
        self.last_outputs = {_tensor_key(tensor): position for position, tensor in enumerate(outputs_torch)}
        result = self.unflatten([outputs_torch[position] for position in self.output_positions])
        if self.logits_position is None:
            logits = getattr(result, "logits", None)
            self.logits_position = next((p for p, t in enumerate(outputs_torch) if t is logits), -1)
        return result

    def _speculate(self, arrays: list, outputs: list) -> tuple[list, list] | None:
        """The next decode step, queued on the inputs greedy decoding would give it, so that the device computes it
        while `generate` picks the token: the logits' argmax as the next token, the next positions, the mask opening
        the next slot, and the outputs that loop back (the cache). `None` where an input can't be predicted.

        The next call runs on it only if its inputs are the predicted ones (`_predicted`), so the outputs are always
        those of the inputs `generate` passes; after a miss (sampling, a logits processor changing the token), the
        runner stops speculating until the next `generate`."""
        if not self.speculating or not self.loops or self.logits_position in (None, -1):
            return None
        predicted = []
        for index, (name, array) in enumerate(zip(self.input_names, arrays)):
            if index in self.loops:
                predicted.append(outputs[self.loops[index]])
            elif name == "input_ids":
                logits = outputs[self.logits_position][:, -1:]
                predicted.append(mx.argmax(logits, axis=-1).astype(array.dtype))
            elif name == "position_ids":
                predicted.append(array[..., -1:] + 1)
            elif name.startswith("attention_mask") and array.ndim == 4:
                # The last query's row, with the slot after its last seen one seen too
                row = array[:, :, -1:, :]
                seen = mx.max(row, axis=-1, keepdims=True)
                slots = mx.arange(row.shape[-1])
                last = mx.max(mx.where(row == seen, slots, -1), axis=-1, keepdims=True)
                predicted.append(mx.where(slots == last + 1, seen, row))
            elif name.startswith("attention_mask") and array.ndim == 2:
                predicted.append(mx.concatenate([array, mx.ones((array.shape[0], 1), dtype=array.dtype)], axis=-1))
            else:
                return None
        return predicted, self.function(*predicted)

    def _predicted(self, predicted: list, tensors: list, looped: list) -> bool:
        """Whether this call's inputs are the ones the queued step was predicted on."""
        if len(predicted) != len(tensors):
            return False
        for index, (array, tensor, position) in enumerate(zip(predicted, tensors, looped)):
            if index in self.loops:
                if position != self.loops[index]:
                    return False
            elif tuple(array.shape) != tuple(tensor.shape) or not torch.equal(to_torch(array), tensor):
                return False
        return True
