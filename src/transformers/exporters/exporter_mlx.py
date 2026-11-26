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
"""Exports a model to MLX: the `torch.export` program decomposed to core ATen, saved as its graph (JSON) and its weights
(safetensors), which the MLX runner rebuilds op by op as MLX ops (`mlx_ops`)."""

from __future__ import annotations

import json
from collections.abc import MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.import_utils import is_mlx_available, is_torch_available
from .configs import ExportFormat, MLXConfig
from .exporter_dynamo import DynamoExporter
from .mlx_ops import MLX_OPS, MLXProgram, to_mlx, to_torch
from .utils import apply_patches


if is_torch_available():
    import torch

if is_mlx_available():
    import mlx.core as mx

if TYPE_CHECKING:
    from torch.export import ExportedProgram

    from ..modeling_utils import PreTrainedModel
    from .metadata import ExportMetadata


class MLXExporter(DynamoExporter):
    """Exporter that converts a [`PreTrainedModel`] to an [`MLXProgram`]: its ATen graph and weights, which MLX runs on
    Apple Silicon GPUs, CUDA GPUs and CPUs.

    Example:

    ```python
    >>> from transformers.exporters import MLXConfig, MLXExporter

    >>> exported = MLXExporter().export(model, inputs, config=MLXConfig(dynamic=True))
    >>> outputs = exported(**inputs)  # on MLX
    >>> exported.save_pretrained("exported")  # `mlx_model.json` + `.safetensors`, its metadata and the model config
    ```
    """

    export_format = ExportFormat.MLX
    config_class = MLXConfig
    program_prefix = "mlx_"
    program_suffix = ".json"
    # A JSON graph and safetensors weights, loading them runs no code
    requires_trust_remote_code = False

    required_packages = ["torch", "mlx"]
    tested_versions = {"torch": "2.13.0", "mlx": "0.32.5"}

    def __init__(self):
        super().__init__()
        # The MLX copy of each torch tensor the exports read, by storage: the components of one model (a generation
        # export's prefill and decode) share their weights instead of each holding a copy
        self._mlx_arrays: dict[tuple, mx.array] = {}

    def export_program(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, Any],
        config: MLXConfig,
    ) -> tuple[MLXProgram, ExportMetadata]:
        with apply_patches("mlx"):
            exported_program, metadata = super().export_program(model, sample_inputs, config=config)
        exported_program = exported_program.run_decompositions(_mlx_decompositions())
        program, metadata.out_constants = _to_mlx_program(exported_program, self._mlx_arrays)
        return program, metadata

    @classmethod
    def save_program(cls, program: MLXProgram, path) -> None:
        path = Path(path)
        path.write_text(json.dumps(program.graph), encoding="utf-8")
        mx.save_safetensors(str(path.with_suffix(".safetensors")), program.weights)

    @classmethod
    def program_files(cls, filename: str) -> list[str]:
        """The `.json` graph and its `.safetensors` weights."""
        return [filename, f"{filename.removesuffix('.json')}.safetensors"]

    @classmethod
    def load_program(cls, path) -> MLXProgram:
        path = Path(path)
        graph = json.loads(path.read_text(encoding="utf-8"))
        return MLXProgram(graph, mx.load(str(path.with_suffix(".safetensors"))))


def _mlx_decompositions() -> dict:
    """The core ATen decompositions, keeping the ops MLX runs fused (`mx.fast`)."""
    table = torch.export.default_decompositions()
    table.pop(torch.ops.aten.scaled_dot_product_attention.default, None)
    return table


def _encode(value) -> Any:
    """A node's argument as JSON: `{"__node__": name}` for another node's value, `{"__dtype__": name}` for a dtype."""
    if isinstance(value, torch.fx.Node):
        return {"__node__": value.name}
    if isinstance(value, torch.dtype):
        return {"__dtype__": str(value).removeprefix("torch.")}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (torch.device, torch.layout, torch.memory_format)):
        # Placement and layout: MLX arrays have neither
        return None
    return value


def _target_name(target) -> str:
    if isinstance(target, torch._ops.OpOverload):
        return str(target)
    return f"{'operator' if target.__module__ == '_operator' else target.__module__}.{target.__name__}"


def _tensor_key(tensor: torch.Tensor) -> tuple:
    return tensor.data_ptr(), tensor.dtype, tuple(tensor.shape), tensor.stride()


def _shared_mlx_array(tensor: torch.Tensor, mlx_arrays: dict[tuple, mx.array]) -> mx.array:
    """The MLX copy of `tensor`, made once. A CPU tensor then reads it: its data becomes a view of the MLX array, so the
    model and its export share one copy of the weights."""
    if (array := mlx_arrays.get(_tensor_key(tensor))) is not None:
        return array
    array = to_mlx(tensor)
    mx.eval(array)
    mlx_arrays[_tensor_key(tensor)] = array
    if tensor.device.type == "cpu":
        tensor.data = to_torch(array)
        mlx_arrays[_tensor_key(tensor)] = array
    return array


def _to_mlx_program(exported_program: ExportedProgram, mlx_arrays: dict[tuple, mx.array]) -> tuple[MLXProgram, list]:
    """The program's graph and weights, and its non-tensor outputs as `[leaf index, value]`.

    The graph returns the mutated inputs' new values (a `StaticCache` write), then the tensor outputs, as the program
    does: runners unflatten the outputs from the end."""
    from torch.export.graph_signature import InputKind, OutputKind, TensorArgument

    signature = exported_program.graph_signature
    if any(spec.kind == OutputKind.BUFFER_MUTATION for spec in signature.output_specs):
        raise NotImplementedError("The MLX exporter can't export a model that mutates its own buffers yet.")
    graph = exported_program.graph
    nodes = [node for node in graph.nodes if node.op == "call_function"]
    unsupported = sorted({_target_name(node.target) for node in nodes} - set(MLX_OPS))
    if unsupported:
        raise NotImplementedError(f"The MLX exporter has no conversion for {unsupported}.")

    state = {**exported_program.state_dict, **exported_program.constants}
    weights, inputs, constants = {}, [], {}
    for spec in signature.input_specs:
        if spec.kind != InputKind.USER_INPUT:
            weights[spec.arg.name] = _shared_mlx_array(state[spec.target], mlx_arrays)
        elif isinstance(spec.arg, TensorArgument):
            inputs.append(spec.arg.name)
        else:
            # A non-tensor input the program was traced with as a constant
            constants[spec.arg.name] = spec.arg.value

    output_args = graph.output_node().args[0]
    num_mutations = sum(spec.kind == OutputKind.USER_INPUT_MUTATION for spec in signature.output_specs)
    user_outputs = output_args[num_mutations:]
    outputs = [node.name for node in output_args[:num_mutations]]
    outputs += [node.name for node in user_outputs if isinstance(node, torch.fx.Node)]
    out_constants = [
        [index, value] for index, value in enumerate(user_outputs) if not isinstance(value, torch.fx.Node)
    ]

    program_graph = {
        "inputs": inputs,
        "constants": constants,
        "nodes": [
            {
                "name": node.name,
                "target": _target_name(node.target),
                "args": _encode(node.args),
                "kwargs": _encode(node.kwargs),
            }
            for node in nodes
        ],
        "outputs": outputs,
    }
    return MLXProgram(program_graph, weights), out_constants
