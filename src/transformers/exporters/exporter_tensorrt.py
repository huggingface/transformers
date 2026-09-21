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
"""TensorRT export: the `torch.export` graph, with what TensorRT can take compiled into engines."""

from __future__ import annotations

import copy
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

import torch
from torch.utils import _pytree as pytree

from ..utils import logging
from .configs import ExportFormat, TensorrtConfig
from .exporter_dynamo import DynamoExporter, get_auto_dynamic_shapes
from .metadata import EXPORT_METADATA_KEY


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)


# Left to torch rather than converted. A cache write is an `index_put`, and Torch-TensorRT's converter for
# it refuses the shapes a cache write has -- it broadcasts the values against the *indexed* dimension order
# and reports `Cannot broadcast (1, 2, 4, 16) to (4, 1, 2, 16)`. Every static-cache export writes its cache
# that way, so the alternative to naming the op here is that none of them convert at all; and the write is
# a memory copy rather than compute, so leaving it in torch is what the partitioner is for.
DEFAULT_TORCH_EXECUTED_OPS = frozenset({"torch.ops.aten.index_put.default"})


def _reshape_scalar_engine_outputs(graph_module: torch.fx.GraphModule) -> None:
    """Reshape an engine's rank-0 outputs back to rank 0.

    TensorRT has no rank-0 tensor, so an engine hands a scalar back as `[1]` however the graph describes
    it -- and the graph describes it as rank 0, which is why this cannot be found by comparing shapes in
    the node metadata: they agree, and only the value at runtime disagrees. A cache's `cumulative_length`
    counter is such a scalar, and the `copy_` that writes it back refuses what it gets ("output with shape
    [] doesn't match the broadcast shape [1]").

    So every engine output the trace called rank 0 gets a reshape to rank 0: a no-op where TensorRT
    behaved, and the repair where it did not.
    """
    import operator

    for node in list(graph_module.graph.nodes):
        if node.op != "call_function" or node.target is not operator.getitem:
            continue
        engine = node.args[0]
        if getattr(engine, "target", None) is not torch.ops.tensorrt.execute_engine.default:
            continue
        value = node.meta.get("val", None)
        if getattr(value, "shape", None) is None or len(value.shape) != 0:
            continue
        with graph_module.graph.inserting_after(node):
            reshaped = graph_module.graph.call_function(torch.ops.aten.reshape.default, (node, []))
        reshaped.meta["val"] = value
        node.replace_all_uses_with(reshaped)
        # The replacement above rewrote the reshape's own argument to itself; point it back at the output
        # it reshapes.
        reshaped.update_arg(0, node)
    graph_module.graph.lint()
    graph_module.recompile()


class TensorrtExporter(DynamoExporter):
    """Compile a model's graph with TensorRT, through Torch-TensorRT's `dynamo` frontend.

    Torch-TensorRT takes the `ExportedProgram` the [`DynamoExporter`] already produces and replaces every
    subgraph TensorRT can build an engine for, leaving the rest as torch ops. Conversion is therefore
    partial by nature -- a KV cache's `aten.cat` is the usual leftover -- and the result is a graph of
    engines with torch segments between them, not one engine.

    What makes the backend this small is that the result is *still* an `ExportedProgram`: an engine rides
    in the graph as a `tensorrt.execute_engine` call, so saving, loading and running are the `torch.export`
    ones, inherited unchanged. Only the load has to differ, and only because that op has to be registered
    before a graph containing it can be read (see [`TensorrtModelRunner`]).
    """

    export_format = ExportFormat.TENSORRT
    artifact_suffix = ".pt2"

    required_packages = ["torch", "torch_tensorrt"]
    tested_versions = {"torch_tensorrt": "2.11.0"}

    def export_artifact(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, Any],
        config: TensorrtConfig | dict[str, Any],
    ) -> tuple[Any, dict]:
        import torch_tensorrt

        if isinstance(config, dict):
            config = TensorrtConfig(**config)
        elif not isinstance(config, TensorrtConfig):
            raise TypeError(f"Expected config to be a TensorrtConfig or dict, got {type(config)}")

        program, metadata = super().export_artifact(model, sample_inputs, config)
        # Kwargs, because that is how the graph was traced: the model's forward takes them by name. The
        # converter's input parser takes tensors and plain containers of them and refuses everything else,
        # so anything that is not a tensor -- a cache, a bare `use_cache=True` -- goes in as its leaves,
        # which is all the parser reads them for (their shapes and dtypes).
        # Deep copies throughout: the converter *runs* the graph to infer shapes, and a cache it ran
        # against comes back holding this step's keys and values. Retracing below against that mutated
        # cache would bake an input spec with more leaves than the graph declares, and the runtime's feed
        # is then refused for a structure it never had.
        example_inputs = {
            name: value if isinstance(value, torch.Tensor) else pytree.tree_leaves(value)
            for name, value in copy.deepcopy(dict(sample_inputs)).items()
        }
        compiled = torch_tensorrt.dynamo.compile(
            program,
            arg_inputs=(),
            kwarg_inputs=example_inputs,
            min_block_size=config.min_block_size,
            truncate_double=config.truncate_double,
            torch_executed_ops=set(
                DEFAULT_TORCH_EXECUTED_OPS if config.torch_executed_ops is None else config.torch_executed_ops
            ),
            # Torch-TensorRT's `force_causal_efficient_attention` pass throws the attention mask away and
            # asks for `is_causal=True` instead. That is the same thing only for a mask that is causal in
            # the top-left sense, which a decode step's is not: one query against a cache of many positions
            # means "attend to everything written so far", where causal means "attend to position 0". The
            # pass does not fail, it returns wrong logits -- so it is off unless the caller asks for it.
            **{"attn_bias_is_causal": False, **(config.compiler_options or {})},
        )
        # Back to an `ExportedProgram`. What comes out of the converter is a `GraphModule` holding engine
        # submodules, which nothing but Torch-TensorRT's own saver knows how to write; re-exporting turns
        # each engine into a `tensorrt.execute_engine` node, and from there the artifact is an ordinary
        # exported program that this backend's inherited `save_artifact` writes like any other.
        # Under the same dynamic shapes the trace used. A retrace against plain example tensors would
        # specialize every axis the export had left symbolic, and the graph would then refuse the first
        # step whose query length is not the captured one.
        dynamic_shapes = config.dynamic_shapes
        if config.dynamic and dynamic_shapes is None:
            dynamic_shapes = get_auto_dynamic_shapes(sample_inputs)
        program = torch.export.export(
            compiled, args=(), kwargs=copy.deepcopy(dict(sample_inputs)), dynamic_shapes=dynamic_shapes
        )
        # After the retrace, not before: re-exporting is what turns the graph's input mutations back into
        # `copy_` calls, so this is the first point at which the rank-0 writes exist to repair.
        _reshape_scalar_engine_outputs(program.graph_module)
        program.graph_module.meta[EXPORT_METADATA_KEY] = metadata
        return program, metadata
