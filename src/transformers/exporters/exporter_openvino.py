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
"""OpenVINO exporter: `DynamoExporter` followed by conversion to an `openvino.Model`.

1. **Patches** (`apply_patches("openvino")`): reversible swaps of `torch` ops the OV frontend cannot lower
   (`histc`, `searchsorted`, ...). The trace runs under `torch.no_grad()` so grad regions do not become
   HigherOrderOp subgraphs.
2. **Graph preparation**: OV's own decompositions run up front (`_run_openvino_decompositions`), then FX program
   fixes, per-node FX fixes, output deduplication and renames. Handing the raw program to `convert_model` would
   re-run the decompositions and lose these fixes.
3. **Conversion**: `TorchFXPythonDecoder` and `openvino.convert_model`, with `ConversionExtension`s for ops OV has
   no lowering for; ports are renamed to their dotted leaf paths.
4. **Stateful transformation** (`OpenVINOConfig.stateful`, on by default): KV cache and SSM states folded into OV
   variables with a fused `beam_idx` reorder.
"""

from __future__ import annotations

import math
import operator
import re
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.import_utils import is_openvino_available, is_torch_available
from .configs import ExportFormat, OpenVINOConfig
from .exporter_dynamo import DynamoExporter, is_cache_object
from .exporter_onnx import disambiguate_io_names, patch_model_outputs
from .utils import (
    BATCH_INPUTS,
    apply_fx_node_fixes,
    apply_fx_program_fixes,
    apply_patches,
    apply_rotary_pos_emb_pairs,
    drop_runtime_asserts,
    get_leaf_tensors,
    leaf_name,
    register_fx_node_fix,
    register_patch,
    zero_fully_masked_rows,
)


if is_torch_available():
    import torch
    from torch.export import ExportedProgram
    from torch.export.decomp_utils import CustomDecompTable


if is_openvino_available():
    import numpy as np
    import openvino
    import openvino.opset14 as ov_ops
    from openvino import Model, PartialShape, Type
    from openvino._offline_transformations import apply_make_stateful_transformation, compress_model_transformation
    from openvino.frontend.pytorch import ConversionExtension
    from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
    from openvino.frontend.pytorch.torchdynamo.export_decompositions import ops_to_not_decompose


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)


class OpenVINOExporter(DynamoExporter):
    """Exporter that converts a [`PreTrainedModel`] to an OpenVINO ``openvino.Model``.

    Example:

    ```python
    >>> from transformers.exporters.exporter_openvino import OpenVINOExporter, OpenVINOConfig

    >>> exporter = OpenVINOExporter()
    >>> ov_model = exporter.export(model, inputs, config=OpenVINOConfig(dynamic=True))
    >>> exporter.export(model, inputs, config=OpenVINOConfig(output_path="model.xml"))
    ```
    """

    required_packages = ["torch", "openvino"]
    tested_versions = {"torch": "2.12.0", "openvino": "2026.3.1"}
    export_format = ExportFormat.OPENVINO
    config_class = OpenVINOConfig
    artifact_suffix = ".xml"
    # A variable's `ReadValue` computes the cross-attention cache on a sequence's first step and keeps it.
    decoder_writes_cross_cache = True

    def export_artifact(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, Any],
        config: OpenVINOConfig | dict[str, Any],
    ) -> tuple[openvino.Model, dict]:
        config = self._as_config(config)

        # With grad enabled, modeling-internal ``torch.no_grad()`` regions trace as ``wrap_with_set_grad_enabled``
        # subgraphs OV's frontend can't lower.
        with torch.no_grad(), patch_model_outputs(model) as (inputs_names, outputs_names), apply_patches("openvino"):
            exported_program, metadata = super().export_artifact(model, sample_inputs, config=config)

        exported_program, graph_module = _fix_exported_program(exported_program)
        ov_model = _convert_to_openvino(graph_module)

        inputs_names = [name for name in inputs_names if name in get_leaf_tensors(sample_inputs)]
        inputs_names, outputs_names = disambiguate_io_names(inputs_names, outputs_names)
        _rename_model_ports(ov_model, graph_module, inputs_names, outputs_names)

        if config.dynamic:
            # Only where the trace left axes symbolic; a static export already declares them all.
            _pin_static_input_axes(ov_model, graph_module, inputs_names)

        if config.stateful:
            _make_stateful(ov_model, exported_program, graph_module, sample_inputs, inputs_names, outputs_names)

        if config.compress_to_fp16:
            # Before saving, so the in-memory model matches what lands on disk.
            compress_model_transformation(ov_model)

        if config.output_path is not None:
            # `save_model` would otherwise halve `f32` weights on its own.
            openvino.save_model(ov_model, config.output_path, compress_to_fp16=False)

        # Precision, cache layout and mask rank aren't in the IR; they travel in `metadata`.
        return ov_model, metadata

    @classmethod
    def save_artifact(cls, artifact, path) -> None:
        """Write the `.xml` graph and `.bin` weights at the precision the model already holds."""
        openvino.save_model(artifact, path, compress_to_fp16=False)


# ── Conversion helpers ──────────────────────────────────────────────────────


def _fix_exported_program(exported_program: ExportedProgram) -> tuple[ExportedProgram, Any]:
    """Decompose and repair the exported program, returning it with the module to convert.

    Both are returned because ``_make_stateful`` reads the program while the port fixes read the module.
    """
    drop_runtime_asserts(exported_program.graph_module)
    exported_program = _run_openvino_decompositions(exported_program)
    apply_fx_program_fixes("openvino", exported_program)
    graph_module = exported_program.module()
    drop_runtime_asserts(graph_module)
    _deduplicate_output_args(graph_module)
    apply_fx_node_fixes("openvino", graph_module)
    _rename_bare_node_names(graph_module)
    return exported_program, graph_module


def _prepare_tensors_for_conversion(graph_module) -> None:
    """Rebind the module's tensors to host float32 copies it owns, before OV reads them.

    OV builds constants with ``shared_memory=True``; for a non-host tensor it points at a freed
    ``numpy(force=True)`` temporary and weights read back as garbage. OV also reads ``bfloat16`` bits as
    ``float16`` (``1.0`` -> ``1.875``), so half-precision tensors are promoted. The caller's model is untouched.
    """
    half = (torch.bfloat16, torch.float16)

    def on_host_in_float32(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype in half:
            tensor = tensor.float()
        return tensor.detach().cpu()

    def needs_repair(tensor: torch.Tensor) -> bool:
        return tensor.device.type != "cpu" or tensor.dtype in half

    for module in graph_module.modules():
        for name, param in list(module._parameters.items()):
            if param is not None and needs_repair(param):
                module._parameters[name] = torch.nn.Parameter(on_host_in_float32(param), requires_grad=False)
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and needs_repair(buffer):
                module._buffers[name] = on_host_in_float32(buffer)
        for name, value in list(module.__dict__.items()):
            if isinstance(value, torch.Tensor) and needs_repair(value):
                setattr(module, name, on_host_in_float32(value))

    # Graph dtypes move with the weights, or SDPA refuses the mixed types. OV types a node from `tensor_meta`
    # before `val`, so both are promoted.
    def promoted(value):
        if isinstance(value, torch.dtype) and value in half:
            return torch.float32
        return value

    def promoted_meta(meta):
        if isinstance(meta, torch.Tensor) and meta.dtype in half:
            return meta.to(torch.float32)
        if hasattr(meta, "_replace") and getattr(meta, "dtype", None) in half:
            return meta._replace(dtype=torch.float32)
        if isinstance(meta, (tuple, list)) and not hasattr(meta, "_replace"):
            return type(meta)(promoted_meta(item) for item in meta)
        return meta

    graph = graph_module.graph
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    for node in graph.nodes:
        if node.op == "placeholder":
            continue
        node.args = tuple(promoted(arg) for arg in node.args)
        node.kwargs = {name: promoted(value) for name, value in node.kwargs.items()}
        for key in ("val", "tensor_meta"):
            if key in node.meta:
                node.meta[key] = promoted_meta(node.meta[key])

    # Inputs keep the type they are fed in and are cast once at entry.
    if placeholders:
        with graph.inserting_after(placeholders[-1]):
            for node in placeholders:
                value = node.meta.get("val")
                if not (isinstance(value, torch.Tensor) and value.dtype in half):
                    continue
                cast = graph.call_function(torch.ops.aten._to_copy.default, (node,), {"dtype": torch.float32})
                cast.meta["val"] = value.to(torch.float32)
                if "tensor_meta" in node.meta:
                    cast.meta["tensor_meta"] = promoted_meta(node.meta["tensor_meta"])
                node.replace_all_uses_with(cast, delete_user_cb=lambda user, cast=cast: user is not cast)

    graph_module.recompile()


def _convert_to_openvino(graph_module) -> openvino.Model:
    """Hand the repaired FX graph to OV's frontend and fix up the ports it produces."""
    _prepare_tensors_for_conversion(graph_module)
    decoder = TorchFXPythonDecoder(graph_module, dynamic_shapes=True)
    # OV may drop unused inputs, so ports are matched to placeholders by name, never positionally.
    decoder._input_signature = [node.name for node in graph_module.graph.nodes if node.op == "placeholder"]
    ov_model = openvino.convert_model(decoder, extension=_OV_CONVERSION_EXTENSIONS)
    _fix_non_tensor_inputs(ov_model, graph_module)
    return ov_model


def _lookup_by_port(port, by_name: Mapping[str, Any]):
    """Return the ``by_name`` entry keyed by one of ``port``'s tensor names, or ``None``."""
    return next((by_name[name] for name in port.get_names() if name in by_name), None)


def _port_named(port, names) -> bool:
    """Whether any of ``port``'s tensor names is one of ``names``."""
    return bool(port.get_names() & set(names))


def _leaf_names_by_placeholder(graph_module, inputs_names: list[str]) -> dict[str, str]:
    """Map each tensor placeholder's FX name to its dotted leaf-path name.

    Tensor placeholders follow kwargs-leaf order, the order ``patch_model_outputs`` captured ``inputs_names`` in.
    """
    tensor_placeholders = [
        node
        for node in graph_module.graph.nodes
        if node.op == "placeholder" and isinstance(node.meta.get("val"), torch.Tensor)
    ]
    return dict(zip((node.name for node in tensor_placeholders), inputs_names))


def _pin_static_input_axes(ov_model: openvino.Model, graph_module, inputs_names: list[str]) -> None:
    """Declare the input axes the trace settled on a number.

    OV's decoder marks every axis dynamic, and the stateful transformation may then resolve the batch from the
    wrong axis (an m-rope ``[sections, batch, positions]`` position_ids answers 3).
    """
    leaf_by_placeholder = _leaf_names_by_placeholder(graph_module, inputs_names)
    traced = {}
    for node in graph_module.graph.nodes:
        leaf = leaf_by_placeholder.get(node.name)
        value = node.meta.get("val") if leaf is not None else None
        if value is None:
            continue
        # Never the cache: its batch is reordered by beams and its length grows.
        leaf = leaf_name(leaf)
        if leaf.startswith(("past_key_values", "cache_params")):
            continue
        traced[leaf] = value.shape

    shapes, changed = {}, False
    for port in ov_model.inputs:
        shape = port.get_partial_shape()
        for name in port.get_names():
            traced_shape = traced.get(leaf_name(name))
            if traced_shape is None or not shape.rank.is_static:
                continue
            for axis, length in enumerate(traced_shape):
                if isinstance(length, int) and shape[axis].is_dynamic:
                    shape[axis] = openvino.Dimension(length)
                    changed = True
        shapes[port.get_any_name()] = shape
    if changed:
        ov_model.reshape(shapes)


def _rename_model_ports(
    ov_model: openvino.Model,
    graph_module,
    inputs_names: list[str],
    outputs_names: list[str],
) -> None:
    """Restore the dotted leaf-path names on the converted model's ports.

    OV's frontend has no ``output=`` and its ``input=`` rejects dotted names, so names are restored after
    conversion. Scalar ports keep their placeholder name.
    """
    leaf_names = _leaf_names_by_placeholder(graph_module, inputs_names)
    for port in ov_model.inputs:
        name = _lookup_by_port(port, leaf_names)
        if name is not None:
            port.get_tensor().set_names({name})
    # A passthrough output (T5's ``encoder_last_hidden_state``) shares its tensor with the input port; route it
    # through a no-op ``convert_like`` so renaming it doesn't clobber the input name.
    changed = False
    for port, name in zip(ov_model.outputs, outputs_names):
        tensor = port.get_tensor()
        if tensor.get_names() & set(inputs_names):
            result = port.get_node()
            source = result.input_value(0)
            copy = ov_ops.convert_like(source, source)
            result.input(0).replace_source_output(copy.output(0))
            copy.output(0).get_tensor().set_names({name})
            changed = True
        else:
            tensor.set_names({name})
    if changed:
        ov_model.validate_nodes_and_infer_types()


def _fix_non_tensor_inputs(ov_model: openvino.Model, graph_module) -> None:
    """Repair Parameters converted from FX non-tensor placeholders.

    OV makes them dynamic-rank, which the CPU plugin refuses to compile. Scalars get a static scalar shape;
    ``None`` and string kwargs (``attention_mask=None``) are removed.
    """
    scalar_types = {bool: openvino.Type.boolean, int: openvino.Type.i64, float: openvino.Type.f32}
    placeholders = {node.name: node for node in graph_module.graph.nodes if node.op == "placeholder"}
    to_remove, changed = [], False
    for port in ov_model.inputs:
        node = _lookup_by_port(port, placeholders)
        if node is None or not port.get_partial_shape().rank.is_dynamic:
            continue
        val = node.meta.get("val")
        if val is None or isinstance(val, str):
            to_remove.append(port.get_node())
            changed = True
        elif type(val) in scalar_types:
            parameter = port.get_node()
            parameter.set_partial_shape(openvino.PartialShape([]))
            parameter.set_element_type(scalar_types[type(val)])
            changed = True
    for parameter in to_remove:
        ov_model.remove_parameter(parameter)
    if changed:
        ov_model.validate_nodes_and_infer_types()


# ── Stateful transformation ─────────────────────────────────────────────────
# Folds round-tripped state (KV cache, SSM states, …) into OV ``ReadValue``/``Assign`` variables. A leaf path on
# both sides of the model that lives in a cache object is state; no per-model naming conventions.

_STATE_BATCH_DIM = 0  # transformers-native caches are batch-first


def _state_leaf_tensors(sample_inputs: MutableMapping[str, Any], state_roots: set) -> list:
    """The rank-4 cache tensors among `sample_inputs`, which are the ones a state pair is made of."""
    return [
        tensor
        for path, tensor in get_leaf_tensors(sample_inputs).items()
        if path.partition(".")[0] in state_roots and tensor.dim() >= 3
    ]


def _find_state_pairs(ov_model: openvino.Model, sample_inputs: MutableMapping[str, Any]) -> dict[str, str]:
    """Return ``{input_port_name: output_port_name}`` for every round-tripped state tensor.

    A leaf path on both sides is state only inside a cache object; a plain kwarg returned under the same name
    (Parakeet's ``attention_mask``) is a regular output.
    """
    state_roots = {key for key, value in sample_inputs.items() if is_cache_object(value)}
    # An all-empty cache is a prefill: written once, never read back. Folding it would pin the variable to the
    # prefill's output length against an empty initializer.
    if all(not tensor.shape[-2] for tensor in _state_leaf_tensors(sample_inputs, state_roots)):
        return {}
    input_names = {name for port in ov_model.inputs for name in port.get_names()}
    pairs = {}
    for port in ov_model.outputs:
        for name in port.get_names():
            prefix, _, path = name.partition(".")
            if prefix == "output" and f"input.{path}" in input_names and path.partition(".")[0] in state_roots:
                pairs[f"input.{path}"] = name
    return pairs


def _fuse_state_reorder(ov_model: openvino.Model, state_input_names: list[str], batchless: set[str]) -> None:
    """Insert a ``beam_idx`` parameter and a batch-dim ``Gather`` in front of every state input that has a batch.

    Beam search reorders the cache between steps; with state inside the model the reorder must be too.
    ``batchless`` states (counters, empty buffers) are skipped: gathering a ``[1]`` counter would give it a batch.
    """
    batch_port = _batch_bearing_input(ov_model, state_input_names)
    batch = batch_port.get_partial_shape()[_STATE_BATCH_DIM]
    beam_idx = ov_ops.parameter(name="beam_idx", dtype=np.int32, shape=openvino.PartialShape([batch]))
    beam_idx.output(0).get_tensor().set_names({"beam_idx"})
    ov_model.add_parameters([beam_idx])
    for input_name in state_input_names:
        if input_name in batchless:
            continue
        state_port = ov_model.input(input_name)
        consumers = state_port.get_target_inputs()
        gather = ov_ops.gather(state_port, beam_idx, ov_ops.constant(np.int64(_STATE_BATCH_DIM)))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def _state_init_dims(
    exported_program: ExportedProgram,
    graph_module,
    sample_inputs: MutableMapping[str, Any],
    pairs: dict[str, str],
    inputs_names: list[str],
    outputs_names: list[str],
) -> dict[str, list]:
    """Per-dim init spec for every state input: ``"batch"``, ``0`` (the growing dim), or a concrete length.

    A dim whose SymInt expression differs from the paired output's (``s2`` vs ``s2 + s3``) grows; the others are
    pinned to the sample tensors' lengths, which keeps this model-type-agnostic.
    """
    leaf_names = _leaf_names_by_placeholder(graph_module, inputs_names)
    input_vals = {
        leaf_names[node.name]: node.meta.get("val")
        for node in graph_module.graph.nodes
        if node.op == "placeholder" and node.name in leaf_names
    }
    # Keyed by trace-ordered leaf names: ``convert_model`` does not always preserve output order.
    node_by_name = {node.name: node for node in exported_program.graph.nodes}
    output_specs = [s for s in exported_program.graph_signature.output_specs if s.kind.name == "USER_OUTPUT"]
    output_vals = {}
    for spec, name in zip(output_specs, outputs_names):
        node = node_by_name.get(getattr(spec.arg, "name", None))
        output_vals[name] = node.meta.get("val") if node is not None else None

    # Counters and empty buffers have no batch axis. Under dynamic shapes a state has one when its leading axis
    # is symbolic; a static trace can only compare sizes.
    batch_val = next(
        (val for name in BATCH_INPUTS for leaf, val in input_vals.items() if leaf_name(leaf) == name), None
    )

    def has_batch(val) -> bool:
        if batch_val is None or batch_val.ndim == 0:
            return val.ndim > 0
        if val.ndim == 0:
            return False
        lead, batch_lead = val.shape[_STATE_BATCH_DIM], batch_val.shape[_STATE_BATCH_DIM]
        return lead == batch_lead if isinstance(batch_lead, int) else not isinstance(lead, int)

    sample_leaves = get_leaf_tensors(sample_inputs)
    cross_paths = _cross_cache_paths(sample_inputs)
    init_dims: dict[str, list] = {}
    for input_name, output_name in pairs.items():
        in_val, out_val = input_vals.get(input_name), output_vals.get(output_name)
        sample = sample_leaves.get(input_name.partition(".")[2])
        if in_val is None or out_val is None or sample is None:
            continue
        # A cross cache's encoder length is the sequence's, not the graph's; it starts empty like a growing axis.
        cross = input_name.partition(".")[2] in cross_paths
        dims = []
        for axis in range(in_val.ndim):
            if axis == _STATE_BATCH_DIM and has_batch(in_val):
                dims.append("batch")
            elif str(in_val.shape[axis]) == str(out_val.shape[axis]) and not (
                cross and not isinstance(in_val.shape[axis], int)
            ):
                dims.append(int(sample.shape[axis]))
            else:
                dims.append(0)
        # ``apply_make_stateful_transformation`` names each variable by its input + output tensor names.
        init_dims[f"{input_name}{output_name}"] = dims
    return init_dims


def _freeze_batchless_states(
    ov_model: openvino.Model,
    pairs: dict[str, str],
    sample_inputs: MutableMapping[str, Any],
) -> None:
    """Replace batch-less tensors the graph passes through unchanged with constants (updating ``pairs``).

    Such a tensor (Cohere2's ``sliding_window_tensor``) is config-derived, not state. One the graph writes (a
    static layer's ``cumulative_length``) stays state.
    """
    sample_leaves = get_leaf_tensors(sample_inputs)
    changed = False
    for input_name, output_name in list(pairs.items()):
        port = ov_model.input(input_name)
        if port.get_partial_shape().rank.get_length() > _STATE_BATCH_DIM:
            continue
        written = ov_model.output(output_name).get_node().input_value(0).get_node()
        if written.get_friendly_name() != port.get_node().get_friendly_name():
            continue
        sample = sample_leaves.get(input_name.partition(".")[2])
        if sample is None:
            continue
        parameter = port.get_node()
        constant = ov_ops.constant(sample.cpu().numpy())
        for consumer in port.get_target_inputs():
            consumer.replace_source_output(constant.output(0))
        ov_model.remove_parameter(parameter)
        del pairs[input_name]
        changed = True
    if changed:
        ov_model.validate_nodes_and_infer_types()


def _batch_bearing_input(ov_model: openvino.Model, exclude: list[str] | None = None):
    """An input whose leading axis is the batch, preferring the text input.

    The first port may be an m-rope ``[sections, batch, positions]`` position_ids (glm4v).
    """
    ports = {name: port for port in ov_model.inputs for name in port.get_names()}
    for name in BATCH_INPUTS:
        if name in ports:
            return ports[name]
    return next(port for port in ov_model.inputs if not _port_named(port, exclude or []))


def _build_state_initializers(ov_model: openvino.Model, init_dims: dict[str, list]) -> None:
    """Give every state variable a zero-filled init expression, so the first ``infer()`` needs no shapes.

    The variable's shape follows the same spec: batch as the inputs declare it, growing dim dynamic, pass-through
    dims concrete. When the update is fully static the batch is pinned to the update's: the CPU plugin can't build
    the Reorder between a static update and a dynamic variable.
    """
    variables = {variable.get_info().variable_id: variable for variable in ov_model.get_variables()}
    update_shapes = {sink.get_variable_id(): sink.input_value(0).get_partial_shape() for sink in ov_model.get_sinks()}
    batch_port = _batch_bearing_input(ov_model)
    batch = ov_ops.gather(
        ov_ops.shape_of(batch_port, output_type="i64"),
        ov_ops.constant([_STATE_BATCH_DIM]),
        ov_ops.constant(0),
    )
    # A dynamic state beside a static query is a pair the fused attention can't show equal.
    declared = batch_port.get_partial_shape()[_STATE_BATCH_DIM]
    batch_dim = declared.get_length() if declared.is_static else -1
    for op in ov_model.get_ops():
        if op.get_type_name() != "ReadValue":
            continue
        update_shape = update_shapes.get(op.get_variable_id())
        update_is_static = update_shape is not None and update_shape.is_static
        dims = init_dims.get(op.get_variable_id())
        if dims is None:
            continue
        if update_is_static:
            dims = [update_shape[axis].get_length() if d == "batch" else d for axis, d in enumerate(dims)]
        info = variables[op.get_variable_id()].get_info()
        info.data_shape = openvino.PartialShape([batch_dim if d == "batch" else -1 if d == 0 else d for d in dims])
        variables[op.get_variable_id()].update(info)
        # ``batch - batch`` rather than a literal ``0``: the CPU plugin fuses the init into ``ReadValueWithSubgraph``
        # and would bake a folded ``0`` into the state descriptor.
        zero = ov_ops.subtract(batch, batch)
        shape = (
            ov_ops.concat(
                [
                    batch if d == "batch" else zero if d == 0 else ov_ops.constant(np.array([d], dtype=np.int64))
                    for d in dims
                ],
                axis=0,
            )
            if dims
            else ov_ops.constant(np.array([], dtype=np.int64))
        )
        zero = ov_ops.constant(0.0, dtype=op.get_output_element_type(0))
        op.set_arguments([ov_ops.broadcast(zero, shape)])
    ov_model.validate_nodes_and_infer_types()


def _align_state_pair_types(ov_model: openvino.Model, pairs: dict[str, str]) -> None:
    """Give each state pair one CPU-friendly storage type, converting at the boundaries.

    ``Assign`` rejects an update typed unlike its variable, and the CPU plugin rejects i64 state; the variable
    stores the output's type, demoted from i64 to i32.
    """
    changed = False
    for input_name, output_name in pairs.items():
        input_port = ov_model.input(input_name)
        original_type = input_port.get_element_type()
        output_port = ov_model.output(output_name)
        output_type = output_port.get_element_type()
        storage_type = openvino.Type.i32 if output_type == openvino.Type.i64 else output_type
        if original_type != storage_type:
            parameter = input_port.get_node()
            consumers = input_port.get_target_inputs()
            parameter.set_element_type(storage_type)
            convert = ov_ops.convert(parameter, original_type)
            for consumer in consumers:
                consumer.replace_source_output(convert.output(0))
            changed = True
        if output_type != storage_type:
            result = output_port.get_node()
            convert = ov_ops.convert(result.input_value(0), storage_type)
            result.input(0).replace_source_output(convert.output(0))
            convert.output(0).get_tensor().set_names({output_name})
            changed = True
    if changed:
        ov_model.validate_nodes_and_infer_types()


def _make_stateful(
    ov_model: openvino.Model,
    exported_program: ExportedProgram,
    graph_module,
    sample_inputs: MutableMapping[str, Any],
    inputs_names: list[str],
    outputs_names: list[str],
) -> None:
    """Convert round-tripped state ports into internal OV variables (in place)."""
    pairs = _find_state_pairs(ov_model, sample_inputs)
    if not pairs:
        logger.debug("No round-tripped state tensors found — leaving the model stateless.")
        return

    # Before freezing: removing a Parameter breaks the placeholder↔port alignment.
    init_dims = _state_init_dims(exported_program, graph_module, sample_inputs, pairs, inputs_names, outputs_names)
    _freeze_batchless_states(ov_model, pairs, sample_inputs)
    if not pairs:
        return

    _align_state_pair_types(ov_model, pairs)
    batchless = {
        name for name, output in pairs.items() if init_dims.get(f"{name}{output}", ["batch"])[:1] != ["batch"]
    }
    _fuse_state_reorder(ov_model, list(pairs), batchless)
    apply_make_stateful_transformation(ov_model, pairs)
    _build_state_initializers(ov_model, init_dims)
    _initialize_cross_state_from_its_write(ov_model, sample_inputs)
    _pin_state_update_shapes(ov_model)


def _cross_cache_paths(sample_inputs: MutableMapping[str, Any]) -> set[str]:
    """Leaf paths of every growing cross-attention cache half, for a graph given `encoder_outputs`."""
    paths = set()
    if sample_inputs.get("encoder_outputs") is None:
        return paths
    for root, value in sample_inputs.items():
        half = getattr(value, "cross_attention_cache", None)
        if not is_cache_object(value) or half is None:
            continue
        if any(getattr(layer, "is_compileable", False) for layer in getattr(half, "layers", [])):
            continue
        paths |= set(get_leaf_tensors({root: {"cross_attention_cache": half}}))
    return paths


def _initialize_cross_state_from_its_write(ov_model: openvino.Model, sample_inputs: MutableMapping[str, Any]) -> None:
    """Make a write-only cross-attention variable's projection its initializer, so it runs once per sequence.

    The CPU plugin fuses a single-consumer init into ``ReadValueWithSubgraph``, which runs only while the
    variable is empty; later steps read what it stored.
    """
    cross_paths = _cross_cache_paths(sample_inputs)
    if not cross_paths:
        return
    wanted = {f"input.{path}output.{path}" for path in cross_paths}
    assigns = {op.get_variable_id(): op for op in ov_model.get_sinks() if op.get_type_name() == "Assign"}
    beam_idx = next((port for port in ov_model.inputs if port.get_any_name() == "beam_idx"), None)
    changed = False
    for read_value in ov_model.get_ordered_ops():
        if read_value.get_type_name() != "ReadValue" or read_value.get_variable_id() not in wanted:
            continue
        assign = assigns.get(read_value.get_variable_id())
        # A variable the graph reads is not a write-only cache, and one with no update has nothing to seed from.
        if assign is None or read_value.output(0).get_target_inputs():
            continue
        write = assign.input_value(0)
        readers = [target for target in write.get_target_inputs() if target.get_node().get_type_name() != "Assign"]
        read_value.set_arguments([write])
        read = read_value.output(0)
        # The CPU plugin refuses ``ReadValueWithSubgraph`` feeding a ``Concat`` directly (t5gemma2); a beam gather
        # in between compiles. Only there, since elsewhere it would block attention fusion.
        if beam_idx is not None and any(target.get_node().get_type_name() == "Concat" for target in readers):
            read = ov_ops.gather(read, beam_idx, ov_ops.constant(np.int64(_STATE_BATCH_DIM))).output(0)
        for target in readers:
            target.replace_source_output(read)
        assign.input(0).replace_source_output(read)
        _unfuse_mean_reductions(write.get_node())
        changed = True
    if changed:
        ov_model.validate_nodes_and_infer_types()


def _unfuse_mean_reductions(root) -> None:
    """Rewrite each ``ReduceMean`` only ``root``'s subgraph uses as ``ReduceSum × 1/n``.

    The CPU plugin refuses an RMSNorm inside a ``ReadValueWithSubgraph`` body (t5gemma2's cross ``k_norm``),
    whose dims are all dynamic; the sum-then-scale spelling keeps it from matching the norm.
    """
    subgraph, stack = {}, [root]
    while stack:
        node = stack.pop()
        name = node.get_friendly_name()
        if name in subgraph or node.get_type_name() in ("Parameter", "Constant", "ReadValue"):
            continue
        subgraph[name] = node
        stack.extend(node.input_value(i).get_node() for i in range(node.get_input_size()))
    for node in subgraph.values():
        if node.get_type_name() != "ReduceMean":
            continue
        # Only a reduction the initializer owns: rewriting a shared one would unfuse a norm outside it too.
        if any(target.get_node().get_friendly_name() not in subgraph for target in node.output(0).get_target_inputs()):
            continue
        data, axes = node.input_value(0), node.input_value(1).get_node()
        shape = data.get_partial_shape()
        if axes.get_type_name() != "Constant" or not shape.rank.is_static:
            continue
        dims = [int(axis) % shape.rank.get_length() for axis in axes.get_data().flatten()]
        if not all(shape[axis].is_static for axis in dims):
            continue
        count = int(np.prod([shape[axis].get_length() for axis in dims]))
        total = ov_ops.reduce_sum(data, node.input_value(1), keep_dims=node.get_keep_dims())
        scale = ov_ops.constant(np.array(1.0 / count, dtype=data.get_element_type().to_dtype()))
        mean = ov_ops.multiply(total, scale)
        for target in node.output(0).get_target_inputs():
            target.replace_source_output(mean.output(0))


def _pin_state_update_shapes(ov_model: openvino.Model) -> None:
    """Reconcile each ``Assign``'s update shape with its variable's, which the CPU plugin requires.

    A fully static update pins the variable; an under-specified update against a static variable (olmo_hybrid's
    conv state) gets a ``special_zero`` Reshape. Pinning refines shapes downstream, so this iterates to a fixpoint.
    """
    variables = {variable.get_info().variable_id: variable for variable in ov_model.get_variables()}
    read_values = {op.get_variable_id(): op for op in ov_model.get_ordered_ops() if op.get_type_name() == "ReadValue"}
    for _ in range(len(variables) + 1):
        changed = False
        for op in ov_model.get_ordered_ops():
            if op.get_type_name() != "Assign":
                continue
            variable = variables[op.get_variable_id()]
            variable_shape = variable.get_info().data_shape
            update = op.input_value(0)
            update_shape = update.get_partial_shape()
            if update_shape == variable_shape:
                continue
            if update_shape.is_static:
                info = variable.get_info()
                info.data_shape = update_shape
                variable.update(info)
                # The init expression gets the same static shape, which the variable's must relax.
                read_value = read_values.get(op.get_variable_id())
                if read_value is not None and read_value.get_input_size() > 0:
                    target = np.array([dim.get_length() for dim in update_shape], dtype=np.int64)
                    pinned_init = ov_ops.reshape(
                        read_value.input_value(0), ov_ops.constant(target), special_zero=False
                    )
                    read_value.set_arguments([pinned_init])
                changed = True
                continue
            if variable_shape.rank.is_dynamic:
                continue
            target = [dim.get_length() if dim.is_static else 0 for dim in variable_shape]
            if all(t == 0 for t in target):
                continue
            pinned = ov_ops.reshape(update, ov_ops.constant(np.array(target, dtype=np.int64)), special_zero=True)
            # Skip a pin that buys nothing, or each round re-pins the pin and the chain of identity reshapes
            # hides the attention from the plugin's fusion.
            if pinned.get_output_partial_shape(0) == update_shape:
                continue
            op.input(0).replace_source_output(pinned.output(0))
            changed = True
        if not changed:
            break
        ov_model.validate_nodes_and_infer_types()


# ── Graph preparation ───────────────────────────────────────────────────────


# Ops OV keeps for itself that we decompose anyway: its ``index_copy`` broadcasts the source against the index,
# which a ``[batch, 1, 1, dim]`` write into a static cache can't satisfy.
_DECOMPOSE_ANYWAY = frozenset({"aten.index_copy.default"})


def _run_openvino_decompositions(exported_program: ExportedProgram) -> ExportedProgram:
    """Run the decomposition pass ``TorchFXPythonDecoder.from_exported_program`` would, so later fixes stick."""
    decomp_table = CustomDecompTable()
    for op in ops_to_not_decompose():
        if str(op) not in _DECOMPOSE_ANYWAY:
            decomp_table.pop(op, None)
    return exported_program.run_decompositions(decomp_table)


def _deduplicate_output_args(graph_module) -> None:
    """Give repeated graph outputs their own node via a fold-resistant self-identity op.

    Two Results sharing one OV tensor crash the translate session's ``is_number`` check. OV folds ``clone`` and
    ``add(x, 0)`` back to ``x``; ``maximum(x, x)`` (``logical_and`` for bool) survives.
    """
    # OV translates these as pass-through, so an output behind one still aliases its source.
    passthrough = (torch.ops.aten.clone.default, torch.ops.aten.alias.default, torch.ops.aten.detach.default)
    output_node = next(node for node in graph_module.graph.nodes if node.op == "output")
    seen = set()

    def dedup(arg):
        source = arg
        while source.op == "call_function" and source.target in passthrough:
            source = source.args[0]
        if source is arg and source not in seen:
            seen.add(source)
            return arg
        with graph_module.graph.inserting_before(output_node):
            val = source.meta.get("val")
            if isinstance(val, torch.Tensor) and val.dtype == torch.bool:
                copy = graph_module.graph.call_function(torch.ops.aten.logical_and.default, args=(source, source))
            else:
                copy = graph_module.graph.call_function(torch.ops.aten.maximum.default, args=(source, source))
            copy.meta.update(source.meta)
        seen.add(source)
        return copy

    output_node.args = (torch.fx.node.map_arg(output_node.args[0], dedup),)
    graph_module.recompile()


def _rename_bare_node_names(graph_module) -> None:
    """Append a numeric suffix to FX node names that lack one, in every nested graph.

    OV's frontend strips a trailing ``_<digits>`` and rejects bare names (``mul``) with ``is_number(name)``.
    HigherOrderOp bodies have their own counters; only top-level placeholders keep their names.
    """
    name_has_suffix = re.compile(r"_\d+$")
    for module in graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        is_top_level = module is graph_module
        used = {n.name for n in module.graph.nodes}
        for n in module.graph.nodes:
            if n.op == "output" or (n.op == "placeholder" and is_top_level):
                continue
            if name_has_suffix.search(n.name):
                continue
            candidate = f"{n.name}_0"
            i = 0
            while candidate in used:
                i += 1
                candidate = f"{n.name}_{i}"
            used.discard(n.name)
            used.add(candidate)
            n._rename(candidate)


# ── FX node fixes ───────────────────────────────────────────────────────────
# Registered via `@register_fx_node_fix("openvino")`; a fix returns `True` when it consumed the node.


@register_fx_node_fix("openvino")
def _fix_varlen_attn_getitem(gm, node):
    """Drop the `getitem` that unpacks `_varlen_attn`'s first output.

    For a converted op OV reads `getitem` as a tensor index and gathers row 0; `_convert_varlen_attn` returns
    the output itself.
    """
    if node.target is not operator.getitem or node.args[1] != 0:
        return False
    source = node.args[0]
    if not isinstance(source, torch.fx.Node) or "_varlen_attn" not in str(source.target):
        return False
    node.replace_all_uses_with(source)
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("openvino")
def _fix_symbolic_pad(gm, node):
    """Decompose ``constant_pad_nd`` with symbolic pad amounts into ``full`` + ``cat``.

    OV can place a symbolic amount on the wrong axis (mamba2's chunked scan). Constant pads keep OV's translation.
    """
    if node.target is not torch.ops.aten.constant_pad_nd.default:
        return False
    x, pads = node.args[0], node.args[1]
    if all(isinstance(p, int) for p in pads):
        return False
    value = node.args[2] if len(node.args) > 2 else 0
    val = x.meta.get("val")
    if val is None:
        return False
    users = list(node.users)
    current = x
    with gm.graph.inserting_before(node):
        for pair_index in range(len(pads) // 2):
            dim = val.ndim - 1 - pair_index  # torch pad pairs run last-dim-first
            for amount, at_front in ((pads[2 * pair_index], True), (pads[2 * pair_index + 1], False)):
                if isinstance(amount, int) and amount == 0:
                    continue
                # A negative amount crops (mamba2's conv warmup); a symbolic one can be either, so build both a
                # ``max(amount, 0)`` filler and a ``min(amount, 0)`` crop.
                filler_size = amount if isinstance(amount, int) else gm.graph.call_function(max, args=(amount, 0))
                crop = 0 if isinstance(amount, int) else gm.graph.call_function(min, args=(amount, 0))
                if isinstance(amount, int) and amount < 0:
                    filler_size, crop = 0, amount
                dim_size = gm.graph.call_function(torch.ops.aten.sym_size.int, args=(current, dim))
                if crop != 0:
                    if at_front:
                        start = gm.graph.call_function(operator.sub, args=(0, crop))
                        current = gm.graph.call_function(
                            torch.ops.aten.slice.Tensor, args=(current, dim, start, dim_size)
                        )
                    else:
                        end = gm.graph.call_function(operator.add, args=(dim_size, crop))
                        current = gm.graph.call_function(torch.ops.aten.slice.Tensor, args=(current, dim, 0, end))
                    current.meta.update(node.meta)
                if not isinstance(filler_size, int) or filler_size > 0:
                    sizes = [
                        filler_size
                        if i == dim
                        else gm.graph.call_function(torch.ops.aten.sym_size.int, args=(current, i))
                        for i in range(val.ndim)
                    ]
                    filler = gm.graph.call_function(
                        torch.ops.aten.full.default,
                        args=(sizes, value),
                        kwargs={"dtype": val.dtype, "device": val.device},
                    )
                    filler.meta.update(node.meta)
                    operands = [filler, current] if at_front else [current, filler]
                    current = gm.graph.call_function(torch.ops.aten.cat.default, args=(operands, dim))
                    current.meta.update(node.meta)
    for user in users:
        user.replace_input_with(node, current)
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("openvino")
def _fix_gather_index_extent(gm, node):
    """Align ``aten.gather``'s index with the data's extent on non-axis dims.

    OV's ``GatherElements`` requires equal shapes off the axis, where torch allows a smaller index. The index is
    expanded and the output narrowed back; narrowing the data gets re-fused by decomposition (efficientloftr).
    """
    if node.target is not torch.ops.aten.gather.default:
        return False
    data, dim, index = node.args[:3]
    data_val, index_val = data.meta.get("val"), index.meta.get("val")
    if data_val is None or index_val is None:
        return False
    axis = dim if dim >= 0 else dim + data_val.ndim
    mismatched = [i for i in range(data_val.ndim) if i != axis and str(data_val.shape[i]) != str(index_val.shape[i])]
    if not mismatched:
        return False
    users = list(node.users)
    with gm.graph.inserting_before(node):
        sizes = [
            gm.graph.call_function(torch.ops.aten.sym_size.int, args=(data, i)) if i in mismatched else -1
            for i in range(data_val.ndim)
        ]
        expanded = gm.graph.call_function(torch.ops.aten.expand.default, args=(index, sizes))
        expanded.meta.update(index.meta)
    node.args = (data, dim, expanded) + tuple(node.args[3:])
    narrowed = node
    for i in mismatched:
        with gm.graph.inserting_after(narrowed):
            size = gm.graph.call_function(torch.ops.aten.sym_size.int, args=(index, i))
        with gm.graph.inserting_after(size):
            narrowed = gm.graph.call_function(torch.ops.aten.slice.Tensor, args=(narrowed, i, 0, size))
            narrowed.meta.update(node.meta)
    for user in users:
        user.replace_input_with(node, narrowed)
    return True


@register_fx_node_fix("openvino")
def _fix_narrow_int_item(gm, node):
    """Read a scalar out of an int64 tensor so shape arithmetic stays one element type.

    An ``.item()`` off an ``int32`` tensor (hunyuan_vl) reaches OV as ``i32`` and fails concatenation with
    ``i64`` shape constants (``Argument element types are inconsistent``).
    """
    if node.target is not torch.ops.aten._local_scalar_dense.default or not node.args:
        return False
    source = node.args[0]
    val = getattr(source, "meta", {}).get("val")
    if val is None or getattr(val, "dtype", None) not in (torch.int32, torch.int16, torch.int8, torch.uint8):
        return False

    with gm.graph.inserting_before(node):
        widened = gm.graph.call_function(
            torch.ops.aten._to_copy.default, args=(source,), kwargs={"dtype": torch.int64}
        )
        widened.meta.update(source.meta)
        widened.meta["val"] = val.to(torch.int64)
    node.args = (widened,) + tuple(node.args[1:])
    return True


@register_fx_node_fix("openvino")
def _fix_integer_last_axis_sum(gm, node):
    """Reduce an integer ``sum`` over the last axis through a rank-2 view.

    The CPU plugin miscompiles an integer last-axis ``ReduceSum`` after a size-1 axis and before an eltwise op
    (BLT's patch ids): batch rows past the first come out wrong.
    """
    if node.target is not torch.ops.aten.sum.dim_IntList or len(node.args) < 2:
        return False
    source, dims = node.args[0], node.args[1]
    in_val, out_val = getattr(source, "meta", {}).get("val"), node.meta.get("val")
    if in_val is None or out_val is None or in_val.dim() < 3:
        return False
    if out_val.dtype.is_floating_point or out_val.dtype.is_complex or out_val.dtype == torch.bool:
        return False
    rank = in_val.dim()
    if [dim % rank for dim in dims] != [rank - 1]:
        return False
    keepdim = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)

    def size_of(axis):
        size = in_val.shape[axis]
        if isinstance(size, int):
            return size
        return gm.graph.call_function(torch.ops.aten.sym_size.int, args=(source, axis))

    with gm.graph.inserting_before(node):
        flat = gm.graph.call_function(torch.ops.aten.view.default, args=(source, [-1, size_of(rank - 1)]))
        flat.meta["val"] = in_val.reshape(-1, in_val.shape[-1])
        summed = gm.graph.call_function(
            torch.ops.aten.sum.dim_IntList, args=(flat, [1], False), kwargs=dict(node.kwargs)
        )
        summed.meta["val"] = flat.meta["val"].sum(1, dtype=out_val.dtype)
        leading = [size_of(axis) for axis in range(rank - 1)] + ([1] if keepdim else [])
        restored = gm.graph.call_function(torch.ops.aten.view.default, args=(summed, leading))
        restored.meta.update(node.meta)
    node.replace_all_uses_with(restored)
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("openvino")
def _fix_empty_cat(gm, node):
    """Drop ``aten.cat([empty, x], dim)`` built by ``DynamicLayer`` for prefill.

    OV can't concatenate the rank-1 ``f32[0]`` operand with a 4D one (``Axis -2 out of the tensor rank range``).
    """
    if node.target is not torch.ops.aten.cat.default:
        return False

    operands = node.args[0]
    if not isinstance(operands, (list, tuple)) or len(operands) != 2:
        return False

    from torch.fx.experimental.symbolic_shapes import guard_or_false

    def _is_empty(n):
        val = n.meta.get("val") if hasattr(n, "meta") else None
        if val is None:
            return False
        # ``numel() == 0`` can be data-dependent (MinimaxM3VL); keeping the cat is always correct.
        return guard_or_false(val.numel() == 0)

    if _is_empty(operands[0]):
        keep = operands[1]
    elif _is_empty(operands[1]):
        keep = operands[0]
    else:
        return False

    node.replace_all_uses_with(keep)
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("openvino")
def _fix_empty_expand(gm, node):
    """Replace ``aten.expand`` of a statically-empty tensor with an explicitly-shaped ``full``.

    OV folds the expand's ``Tile`` with repeats ``output_dim / input_dim``; a zero dim makes that ``0 / 0`` and
    SIGFPEs the process (chmv2's empty ``register_tokens``).
    """
    if node.target is not torch.ops.aten.expand.default:
        return False
    tensor, sizes = node.args[0], node.args[1]
    val = tensor.meta.get("val") if hasattr(tensor, "meta") else None
    out_val = node.meta.get("val")
    if val is None or out_val is None:
        return False

    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if not guard_or_false(val.numel() == 0):
        return False
    users = list(node.users)
    offset = len(sizes) - val.ndim  # expand aligns sizes to the input's trailing dims
    with gm.graph.inserting_before(node):
        full_sizes = []
        for i, size in enumerate(sizes):
            if isinstance(size, int) and size == -1:  # -1 keeps the input's dim
                dim = val.shape[i - offset]
                size = (
                    int(dim)
                    if isinstance(dim, int)
                    else gm.graph.call_function(torch.ops.aten.sym_size.int, args=(tensor, i - offset))
                )
            full_sizes.append(size)
        full = gm.graph.call_function(
            torch.ops.aten.full.default,
            args=(full_sizes, 0),
            kwargs={"dtype": out_val.dtype, "device": out_val.device},
        )
        full.meta.update(node.meta)
    for user in users:
        user.replace_input_with(node, full)
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("openvino")
def _fix_view_inferred_dim(gm, node):
    """Replace the inferred ``-1`` in an ``aten.view`` target that also carries a symbolic dim.

    OV mis-resolves the ``-1`` beside a runtime ``sym_size`` dim (edgetam/sam3_tracker's mask decoder gets
    ``(64, 1, 8, 8)`` for ``(2, 32, 8, 8)``). The traced output's static size is substituted.
    """
    if node.target not in (torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default):
        return False
    shape = node.args[1]
    if not isinstance(shape, (list, tuple)):
        return False
    minus_one = [i for i, dim in enumerate(shape) if isinstance(dim, int) and dim == -1]
    if len(minus_one) != 1:
        return False
    index = minus_one[0]
    out_val = node.meta.get("val")
    if out_val is None:
        return False
    # Only a static size; pinning a symbolic one would be wrong. Left inferred, a decode step's head count goes
    # dynamic and the CPU plugin stops fusing attention with its KV cache.
    resolved = out_val.shape[index]
    if not isinstance(resolved, int):
        return False
    new_shape = list(shape)
    new_shape[index] = resolved
    node.args = (node.args[0], new_shape) + tuple(node.args[2:])
    return True


@register_fx_node_fix("openvino")
def _fix_index_put_as_where(gm, node):
    """Rewrite a non-accumulating ``aten.index_put`` into a broadcast ``where`` when its index is a mask.

    Two shapes are rewritten. A single boolean mask, which OV lowers through a ``nonzero``-style gather it can't
    convert (``SequenceMark``; t5gemma2); flattened per-row values, which ``where`` can't express, are left
    untouched. One 1-D index tensor among ``None`` entries with a scalar value, as OV turns each ``None`` into
    an untranslatable ``torch::None`` (chameleon's logit masking).
    """
    if node.target not in (torch.ops.aten.index_put.default, torch.ops.aten.index_put_.default):
        return False
    if len(node.args) < 3:
        return False
    self_arg, indices, values = node.args[0], node.args[1], node.args[2]
    accumulate = node.args[3] if len(node.args) > 3 else node.kwargs.get("accumulate", False)
    if accumulate or not isinstance(indices, (list, tuple)):
        return False
    self_val = self_arg.meta.get("val")
    values_val = values.meta.get("val") if hasattr(values, "meta") else None
    non_none = [(dim, ix) for dim, ix in enumerate(indices) if ix is not None]
    if self_val is None or values_val is None or len(non_none) != 1:
        return False
    dim, index = non_none[0]
    index_val = index.meta.get("val") if hasattr(index, "meta") else None
    if index_val is None:
        return False

    if len(indices) == 1 and index_val.dtype == torch.bool:
        # Otherwise ``values`` is a flattened selected-rows tensor.
        if index_val.ndim > self_val.ndim or values_val.ndim > self_val.ndim - index_val.ndim:
            return False
        with gm.graph.inserting_before(node):
            mask = index
            for _ in range(self_val.ndim - index_val.ndim):
                mask = gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(mask, -1))
    elif len(indices) > 1 and index_val.ndim == 1 and values_val.numel() == 1:
        size = self_val.shape[dim]
        if not isinstance(size, int):
            return False
        with gm.graph.inserting_before(node):
            iota = gm.graph.call_function(
                torch.ops.aten.arange.default, args=(size,), kwargs={"device": self_val.device}
            )
            iota = gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(iota, 1))
            index = gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(index, 0))
            eq = gm.graph.call_function(torch.ops.aten.eq.Tensor, args=(iota, index))
            mask = gm.graph.call_function(torch.ops.aten.any.dim, args=(eq, 1))
            broadcast_shape = [1] * self_val.ndim
            broadcast_shape[dim] = size
            mask = gm.graph.call_function(torch.ops.aten.view.default, args=(mask, broadcast_shape))
    else:
        return False

    with gm.graph.inserting_before(node):
        result = gm.graph.call_function(torch.ops.aten.where.self, args=(mask, values, self_arg))
        result.meta.update(node.meta)
    node.replace_all_uses_with(result)
    gm.graph.erase_node(node)
    return True


# ── Torch patches ───────────────────────────────────────────────────────────
# Reversibly swap torch ops the OV frontend can't lower, via `@register_patch("openvino", path)`.


@register_patch("openvino", "torch.nn.functional.layer_norm")
def _patch_layer_norm(original):
    """Substitute identity ``weight``/``bias`` when either is ``None``.

    OV refuses the ``torch::None`` constant of an unwired optional (Chameleon).
    """

    def patch(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        if weight is None:
            weight = torch.ones(normalized_shape, dtype=input.dtype, device=input.device)
        if bias is None:
            bias = torch.zeros(normalized_shape, dtype=input.dtype, device=input.device)
        return original(input, normalized_shape, weight, bias, eps)

    return patch


@register_patch("openvino", "torch.nn.functional.interpolate")
def _patch_interpolate(original):
    """Carry ``antialias=True`` resampling into the graph as explicit weights.

    OV's ``Interpolate`` silently ignores ``antialias`` (siglip2's position embeddings). The separable
    triangle-filter weights are built from tensor ops and applied as two matmuls, so extents stay symbolic.
    """

    def axis_weights(size_in, size_out, dtype, device):
        in_index = torch.arange(size_in, dtype=torch.float32, device=device)
        out_index = torch.arange(size_out, dtype=torch.float32, device=device)
        # The ratio stays a tensor: OV inverts a ``Divide`` of two reduced extents feeding only internal nodes.
        scale = torch.zeros((), dtype=torch.float32, device=device) + size_in / size_out
        support = scale.clamp(min=1.0)
        center = (out_index + 0.5) * scale
        distance = (in_index[None, :] + 0.5 - center[:, None]) / support
        weights = (1.0 - distance.abs()).clamp(min=0.0)
        return (weights / weights.sum(-1, keepdim=True)).to(dtype)

    def patch(input, size=None, scale_factor=None, mode="nearest", align_corners=None, **kwargs):
        height_out, width_out = (size, size) if isinstance(size, int) else (size or (None, None))
        supported = mode in ("bilinear", "bicubic") and input.dim() == 4 and size is not None
        if not kwargs.get("antialias", False) or not supported:
            return original(
                input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, **kwargs
            )
        height_in, width_in = input.shape[-2:]
        resized = torch.einsum("nchw,oh->ncow", input, axis_weights(height_in, height_out, input.dtype, input.device))
        return torch.einsum("ncow,pw->ncop", resized, axis_weights(width_in, width_out, input.dtype, input.device))

    return patch


@register_patch("openvino", "torch.nn.functional.scaled_dot_product_attention")
def _patch_sdpa(original):
    """Pre-expand K/V to Q's head count, and keep fully-masked rows finite and zeroed.

    OV's ``ScaledDotProductAttention`` rejects GQA shapes (``Key input shape not compatible with other inputs``).
    """

    def patch(query, key, value, attn_mask=None, *args, **kwargs):
        # OV's SDPA returns NaN for fully masked rows under a boolean mask
        # (https://github.com/openvinotoolkit/openvino/issues/31630); pass a finite additive mask instead.
        masked = attn_mask is not None
        if masked:
            masked_value = torch.finfo(query.dtype).min
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.where(
                    attn_mask,
                    torch.zeros((), dtype=query.dtype, device=attn_mask.device),
                    torch.full((), masked_value, dtype=query.dtype, device=attn_mask.device),
                )
            else:
                attn_mask = attn_mask.clamp_min(masked_value)
        elif not kwargs.get("is_causal", False):
            # OV's fused KV-cache SDPA rejects a call without a mask (`attention_mask do not match q and k`)
            attn_mask = query.new_zeros(query.shape[-2], key.shape[-2])
        q_heads, k_heads = query.shape[-3], key.shape[-3]
        if q_heads != k_heads and q_heads % k_heads == 0:
            reps = q_heads // k_heads
            key = key.repeat_interleave(reps, dim=-3)
            value = value.repeat_interleave(reps, dim=-3)
        # OV's SDPA drops the traced ``scale`` and applies ``head_dim**-0.5``; fold a non-default scale into
        # the query.
        scale = kwargs.pop("scale", None)
        if scale is not None:
            default_scale = query.shape[-1] ** -0.5
            if scale != default_scale:
                query = query * (scale / default_scale)
        attn_output = original(query, key, value, attn_mask, *args, **kwargs)
        # OV returns the uniform average for a fully-masked row; torch's fused kernels write zeros.
        return zero_fully_masked_rows(attn_output, attn_mask) if masked else attn_output

    return patch


@register_patch("openvino", "torch.matmul", "torch.Tensor.matmul")
def _patch_matmul(original):
    """Flatten a two-axis batch before ``MatMul``.

    OV folds leading axes into one batch per operand and refuses m-rope's ``[sections, batch, ...]`` matmul.
    """

    def patch(input, other, **kwargs):
        batched = (
            isinstance(other, torch.Tensor)
            and input.dim() == 4
            and other.dim() == 4
            and input.shape[:2] == other.shape[:2]
        )
        if not batched:
            return original(input, other, **kwargs)
        leading = input.shape[:2]
        product = original(input.flatten(0, 1), other.flatten(0, 1), **kwargs)
        return product.unflatten(0, leading)

    return patch


@register_patch("openvino", "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextModel._deepstack_process")
def _patch_qwen3vl_deepstack(original):
    """Rewrite qwen3-vl deepstack injection without data-dependent boolean-mask indexing.

    Uses ``cumsum`` + ``index_select`` and a ``float(mask)`` multiply.
    """

    def patch(self, hidden_states, visual_pos_masks, visual_embeds):
        visual_embeds = visual_embeds.to(hidden_states.dtype)
        batch, seq_len, dim = hidden_states.shape
        flat_mask = visual_pos_masks.reshape(-1)
        indices = torch.clamp(torch.cumsum(flat_mask.long(), dim=0) - 1, min=0)
        full_visual = torch.index_select(visual_embeds, 0, indices).reshape(batch, seq_len, dim)
        return hidden_states + full_visual * flat_mask.to(hidden_states.dtype).reshape(batch, seq_len, 1)

    return patch


@register_patch(
    "openvino",
    "transformers.models.wavlm.modeling_wavlm.WavLMPreTrainedModel._get_feature_vector_attention_mask",
    "transformers.models.data2vec.modeling_data2vec_audio.Data2VecAudioPreTrainedModel._get_feature_vector_attention_mask",
)
def _patch_feature_vector_attention_mask(original):
    """Build the downsampled attention mask as ``arange(seq) < output_lengths`` instead of a scatter.

    OV can't convert the original's ``aten.index_put`` (``SequenceMark``).
    """

    def patch(self, feature_vector_length, attention_mask, add_adapter=None):
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)
        positions = torch.arange(feature_vector_length, device=attention_mask.device)
        return positions.unsqueeze(0) < output_lengths.unsqueeze(1)

    return patch


@register_patch("openvino", "torch.polar")
def _patch_polar(original):
    """Build ``polar(abs, angle)`` via Euler's formula; OV has no ``aten.polar`` lowering."""

    def patch(abs, angle):
        return torch.complex(abs * angle.cos(), abs * angle.sin())

    return patch


@register_patch("openvino", "transformers.models.deepseek_v2.modeling_deepseek_v2.apply_rotary_emb")
def _patch_deepseek_rotary_emb(original):
    """Rewrite complex-arithmetic RoPE with the equivalent real re/im-pair math.

    OV's ``ComplexTypeMark`` and our ``[..., 2]`` real-pair complex representation can't be mixed in one mul.
    """

    def patch(xq, xk, freqs_cis):
        freqs_pairs = torch.view_as_real(freqs_cis).unsqueeze(1).to(xq.device)
        return apply_rotary_pos_emb_pairs(xq, freqs_pairs), apply_rotary_pos_emb_pairs(xk, freqs_pairs)

    return patch


@register_patch("openvino", "transformers.models.llama4.modeling_llama4.apply_rotary_emb")
def _patch_llama4_rotary_emb(original):
    """Same real-pair rewrite as ``_patch_deepseek_rotary_emb`` for llama4's text RoPE."""

    def patch(xq, xk, freqs_cis):
        freqs_pairs = torch.view_as_real(freqs_cis)[:, :, None, :, :]
        return apply_rotary_pos_emb_pairs(xq, freqs_pairs), apply_rotary_pos_emb_pairs(xk, freqs_pairs)

    return patch


@register_patch("openvino", "transformers.models.llama4.modeling_llama4.vision_apply_rotary_emb")
def _patch_llama4_vision_rotary_emb(original):
    """Same real-pair rewrite as ``_patch_deepseek_rotary_emb`` for llama4's vision RoPE."""

    def patch(query, key, freqs_ci):
        freqs_pairs = torch.view_as_real(freqs_ci)
        # Mirror ``reshape_for_broadcast``: keep dims 1 (seq) and -1 (d/2), plus the re/im pair.
        shape = [d if i == 1 else 1 for i, d in enumerate(query.shape[:-1])] + [freqs_pairs.shape[-2], 2]
        freqs_pairs = freqs_pairs.view(*shape).to(query.device)
        return apply_rotary_pos_emb_pairs(query, freqs_pairs), apply_rotary_pos_emb_pairs(key, freqs_pairs)

    return patch


@register_patch(
    "openvino",
    "transformers.models.phi3.modeling_phi3.Phi3RotaryEmbedding.forward",
    "transformers.models.phimoe.modeling_phimoe.PhimoeRotaryEmbedding.forward",
)
def _patch_longrope_rotary_emb(original):
    """Trace both LongRoPE frequency sets and select with ``torch.where`` on the sequence length.

    Eager LongRoPE branches in Python and mutates a buffer, both data-dependent on a dynamic ``position_ids``.
    """

    def patch(self, x, position_ids=None, layer_type=None):
        rope_type = self.rope_type if layer_type is None else self.rope_type[layer_type]
        if rope_type != "longrope":
            return (
                original(self, x, position_ids) if layer_type is None else original(self, x, position_ids, layer_type)
            )

        params = self.config.rope_parameters if layer_type is None else self.config.rope_parameters[layer_type]
        original_max = params["original_max_position_embeddings"]
        partial_rotary_factor = params.get("partial_rotary_factor", 1.0)
        head_dim = getattr(self.config, "head_dim", None) or self.config.hidden_size // self.config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        seq_len = torch.max(position_ids) + 1
        is_long = seq_len > original_max

        long_factors = torch.tensor(params["long_factor"], dtype=torch.float32, device=x.device)
        short_factors = torch.tensor(params["short_factor"], dtype=torch.float32, device=x.device)
        ext_factors = torch.where(is_long, long_factors, short_factors)
        inv_freq_shape = torch.arange(0, dim, 2, dtype=torch.int64, device=x.device).float() / dim
        inv_freq = 1.0 / (ext_factors * params["rope_theta"] ** inv_freq_shape)

        long_mscale, short_mscale = params.get("long_mscale"), params.get("short_mscale")
        if long_mscale is not None and short_mscale is not None:
            mscale = torch.where(
                is_long,
                torch.tensor(long_mscale, dtype=x.dtype, device=x.device),
                torch.tensor(short_mscale, dtype=x.dtype, device=x.device),
            )
        else:
            factor = params.get("factor") or self.config.max_position_embeddings / original_max
            mscale = params.get("attention_factor")
            if mscale is None:
                mscale = 1.0 if factor <= 1.0 else math.sqrt(1 + math.log(factor) / math.log(original_max))

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * mscale
            sin = emb.sin() * mscale
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    return patch


@register_patch("openvino", "torch.nn.functional.avg_pool2d")
def _patch_avg_pool2d(original):
    """Clamp oversize pooling kernels to the padded input; OV's ``AvgPool`` rejects larger ones (EfficientNet)."""

    def patch(
        input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None
    ):
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        ph, pw = (padding, padding) if isinstance(padding, int) else padding
        kh = torch.sym_min(kh, input.shape[-2] + 2 * ph)
        kw = torch.sym_min(kw, input.shape[-1] + 2 * pw)
        if stride is None:
            stride = (kh, kw)
        return original(input, (kh, kw), stride, padding, ceil_mode, count_include_pad, divisor_override)

    return patch


@register_patch("openvino", "torch.Tensor.unfold")
def _patch_unfold(original):
    """Decompose ``Tensor.unfold`` into ``index_select`` + reshape.

    OV's translator builds an invalid Transpose for 3D inputs (PatchTST).
    """

    def patch(self, dimension, size, step):
        dim = dimension if dimension >= 0 else dimension + self.dim()
        starts = torch.arange(0, self.shape[dim] - size + 1, step, device=self.device)
        indices = (starts.unsqueeze(1) + torch.arange(size, device=self.device)).flatten()
        windows = self.index_select(dim, indices).unflatten(dim, (starts.shape[0], size))
        return windows.movedim(dim + 1, -1)

    return patch


@register_patch("openvino", "torch.bernoulli")
@register_patch("openvino", "torch.randn", "torch.randn_like")
def _patch_randn(original):
    """Strip randomness: return zeros shaped like the argument or the requested size."""

    def patch(*args, **kwargs):
        if args and isinstance(args[0], torch.Tensor):
            return torch.zeros_like(args[0])
        return torch.zeros(*args, **kwargs)

    return patch


@register_patch("openvino", "torch.randperm")
def _patch_randperm(original):
    """Strip randomness from ``torch.randperm`` — return the identity permutation."""

    def patch(n, *, dtype=None, device=None, **kwargs):
        return torch.arange(n, dtype=dtype if dtype is not None else torch.int64, device=device)

    return patch


@register_patch("openvino", "torch.randint")
def _patch_randint(original):
    """Strip randomness from ``torch.randint`` — return zeros."""

    def patch(*args, **kwargs):
        # Signatures: ``randint(high, size, ...)`` or ``randint(low, high, size, ...)``.
        size = next((a for a in args if isinstance(a, (list, tuple, torch.Size))), kwargs.get("size"))
        return torch.zeros(size, dtype=kwargs.get("dtype", torch.int64), device=kwargs.get("device"))

    return patch


@register_patch("openvino", "torch.nn.functional.embedding_bag")
def _patch_embedding_bag(original):
    """Decompose the 2-D, offset-free ``embedding_bag`` into an embedding lookup and a reduction.

    The CPU plugin can't compile ``aten._embedding_bag`` (``to_shape was called on a dynamic shape``).
    """

    def patch(
        input,
        weight,
        offsets=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        mode="mean",
        sparse=False,
        per_sample_weights=None,
        include_last_offset=False,
        padding_idx=None,
    ):
        supported = input.dim() == 2 and offsets is None and max_norm is None and padding_idx is None
        if not supported or include_last_offset or mode not in ("sum", "mean", "max"):
            return original(
                input,
                weight,
                offsets,
                max_norm,
                norm_type,
                scale_grad_by_freq,
                mode,
                sparse,
                per_sample_weights,
                include_last_offset,
                padding_idx,
            )

        embedded = torch.nn.functional.embedding(input, weight)
        if mode == "sum":
            if per_sample_weights is not None:
                embedded = embedded * per_sample_weights.unsqueeze(-1).to(embedded.dtype)
            return embedded.sum(dim=1)
        return embedded.mean(dim=1) if mode == "mean" else embedded.amax(dim=1)

    return patch


@register_patch("openvino", "torch.cumsum", "torch.Tensor.cumsum")
def _patch_cumsum(original):
    """Promote integral inputs to ``int64`` the way torch does before summing.

    OV's ``CumSum`` keeps the input type, so a bool mask saturates at ``True`` (OPT's ``position_ids``).
    """

    def patch(input, *args, dtype=None, **kwargs):
        if dtype is None and input.dtype in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32):
            dtype = torch.int64
        return original(input, *args, dtype=dtype, **kwargs)

    return patch


@register_patch("openvino", "torch.bincount", "torch.Tensor.bincount")
def _patch_bincount(original):
    """Replace ``torch.bincount``, which OV can't lower, with ``zeros + scatter_add_``."""

    from torch.fx.experimental.symbolic_shapes import guard_or_true

    def patch(input, weights=None, minlength=0):
        flat = input.reshape(-1)
        # ``guard_or_true`` avoids a data-dependent guard (splinter); an empty input gives a zero-length
        # output anyway.
        if guard_or_true(flat.numel() > 0):
            max_val = flat.max().item()
            torch._check(max_val >= 0)
            bin_count = max_val + 1
        else:
            bin_count = 0
        bins = torch.sym_max(minlength, bin_count)
        out_dtype = weights.dtype if weights is not None else torch.long
        counts = torch.zeros(bins, dtype=out_dtype, device=input.device)
        src = weights.reshape(-1).to(out_dtype) if weights is not None else torch.ones_like(flat, dtype=out_dtype)
        return counts.scatter_add_(0, flat.long(), src)

    return patch


@register_patch("openvino", "torch.fft.irfft")
def _patch_irfft(original):
    """Compute ``irfft`` in real arithmetic against cos/sin DFT bases.

    OV's ``DFT`` rejects ``is_onesided`` with ``inverse``, and a complex decomposition mixes OV's
    ``ComplexTypeMark`` with our ``[..., 2]`` real-pair representation.
    """

    def patch(input, n=None, dim=-1, norm=None):
        if n is None:
            n = 2 * (input.shape[dim] - 1)
        if torch.is_complex(input):
            pairs = torch.view_as_real(input)
            real, imag = pairs[..., 0], pairs[..., 1]
        else:
            real, imag = input, torch.zeros_like(input)
        real = real.movedim(dim, -1)
        imag = imag.movedim(dim, -1)
        # Mirror to the full n-point spectrum via conjugate symmetry: X[n - k] = conj(X[k]).
        mirror = slice(1, n - real.shape[-1] + 1)
        real = torch.cat([real, real[..., mirror].flip(-1)], dim=-1)
        imag = torch.cat([imag, -imag[..., mirror].flip(-1)], dim=-1)
        # y[j] = scale * sum_k (real[k] cos(2 pi k j / n) - imag[k] sin(2 pi k j / n))
        k = torch.arange(n, device=input.device, dtype=real.dtype)
        angles = 2.0 * torch.pi * k.view(-1, 1) * k / n  # symmetric [n, n], no transpose needed
        scale = {"forward": 1.0, "ortho": n**-0.5}.get(norm, 1.0 / n)
        out = (real @ angles.cos() - imag @ angles.sin()) * scale
        return out.movedim(-1, dim)

    return patch


@register_patch("openvino", "torch.fft.rfft")
def _patch_rfft(original):
    """Replace ``rfft`` with a two-sided ``fft`` sliced to the one-sided half (see ``_patch_irfft``)."""

    def patch(input, n=None, dim=-1, norm=None):
        full = torch.fft.fft(input, n=n, dim=dim, norm=norm)
        n_full = full.shape[dim]
        slc = [slice(None)] * full.ndim
        slc[dim] = slice(0, n_full // 2 + 1)
        return full[tuple(slc)]

    return patch


def _dft(input, n, dim):
    """1-D DFT as a twiddle matmul; OV translates no ``aten._fft_c2c``."""
    if n is None:
        n = input.shape[dim]
    k = torch.arange(n, device=input.device, dtype=torch.float32)
    angles = -2.0 * torch.pi * k.view(-1, 1) * k / n
    twiddle = torch.complex(angles.cos(), angles.sin())
    x = input if torch.is_complex(input) else input.to(torch.complex64)
    out = x.movedim(dim, -1) @ twiddle.T
    return out.movedim(-1, dim)


@register_patch("openvino", "torch.fft.fft")
def _patch_fft(original):
    """``torch.fft.fft`` lowers to an ``aten._fft_c2c`` OV's frontend can't translate."""

    def patch(input, n=None, dim=-1, norm=None):
        return _dft(input, n, dim)

    return patch


@register_patch("openvino", "torch.fft.fftn")
def _patch_fftn(original):
    """Multi-dim FFT as successive 1-D ``torch.fft.fft`` calls (FNet)."""

    def patch(input, s=None, dim=None, norm=None):
        dims = list(range(input.ndim)) if dim is None else list(dim)
        sizes = [None] * len(dims) if s is None else list(s)
        out = input
        for d, n in zip(dims, sizes):
            out = torch.fft.fft(out, n=n, dim=d, norm=norm)
        return out

    return patch


# ── OpenVINO conversion extensions ──────────────────────────────────────────
# ``ConversionExtension`` translations for ops with no torch-level decomposition, see ``_OV_CONVERSION_EXTENSIONS``.


def _match_kv_heads(query, tensor):
    """Grouped K/V widened to the query's head count, for a ``[tokens, heads, dim]`` packed tensor.

    OV's SDPA refuses two head counts. Done outside the loop body, where the head counts are static.
    """
    query_shape, tensor_shape = query.get_partial_shape(), tensor.get_partial_shape()
    if query_shape.rank.get_length() != 3 or not (query_shape[1].is_static and tensor_shape[1].is_static):
        return tensor
    heads, kv_heads = query_shape[1].get_length(), tensor_shape[1].get_length()
    if heads == kv_heads or kv_heads == 0 or heads % kv_heads:
        return tensor
    if not tensor_shape[2].is_static:
        return tensor
    # ``repeat_interleave`` order: tile a size-1 axis beside the heads, then fold it in.
    widened = ov_ops.unsqueeze(tensor, ov_ops.constant(np.array([2], dtype=np.int64)))
    tiled = ov_ops.tile(widened, ov_ops.constant(np.array([1, 1, heads // kv_heads, 1], dtype=np.int64)))
    target = np.array([0, heads, tensor_shape[2].get_length()], dtype=np.int64)
    return ov_ops.reshape(tiled, ov_ops.constant(target), special_zero=True).output(0)


def _convert_varlen_attn(context):
    """Convert ``torch_attn::_varlen_attn`` (packed variable-length attention) into an OV ``Loop``.

    One iteration per ``cu_seqlens`` segment, each a dense SDPA written into a loop-carried output: ``sum(n_i^2)``
    work and no ``L x L`` mask. Segment sizes vary, so the output is merged rather than scanned.
    """
    query, key, value = (context.get_input(index) for index in range(3))
    key, value = _match_kv_heads(query, key), _match_kv_heads(query, value)
    cu_seqlens = ov_ops.convert(context.get_input(3), "i64")
    element_type = query.get_element_type()
    axis0 = ov_ops.constant(np.array([0], dtype=np.int64))
    one = ov_ops.constant(np.array([1], dtype=np.int64))

    # OV's SDPA is batch-first and heads-major; transpose once outside the body.
    heads_first = ov_ops.constant(np.array([1, 0, 2], dtype=np.int64))
    axis2 = ov_ops.constant(np.array([2], dtype=np.int64))
    query, key, value = (
        ov_ops.unsqueeze(ov_ops.transpose(tensor, heads_first), axis0).output(0) for tensor in (query, key, value)
    )

    # Body: (iteration, q, k, v, cu, carried output) -> (written output, keep going)
    iteration = ov_ops.parameter([], Type.i64)
    body_q, body_k, body_v = (ov_ops.parameter(PartialShape([-1, -1, -1, -1]), element_type) for _ in range(3))
    body_cu = ov_ops.parameter(PartialShape([-1]), Type.i64)
    carried = ov_ops.parameter(PartialShape([-1, -1, -1, -1]), element_type)

    index = ov_ops.reshape(iteration, one, False)
    start = ov_ops.gather(body_cu, index, axis0)
    stop = ov_ops.gather(body_cu, ov_ops.add(index, one), axis0)

    def _segment(tensor):
        """This iteration's tokens, still `[1, heads, n_i, dim]`."""
        return ov_ops.slice(tensor, start, stop, one, axis2)

    # The op's own `scale`, when given; OV's SDPA otherwise applies `head_dim ** -0.5`
    scale = context.get_values_from_const_input(8) if context.get_input_size() > 8 else None
    if scale is not None:
        scale = ov_ops.constant(np.array(scale, dtype=element_type.to_dtype()))
    segment = ov_ops.scaled_dot_product_attention(
        _segment(body_q), _segment(body_k), _segment(body_v), scale=scale, causal=False
    )
    # Scattered into a whole-length buffer: a loop-carried value keeps one shape, so concatenating would carry
    # only the last segment.
    rows = ov_ops.range(
        ov_ops.squeeze(start, axis0), ov_ops.squeeze(stop, axis0), ov_ops.constant(np.int64(1)), output_type="i64"
    )
    grown = ov_ops.scatter_update(carried, rows, segment, ov_ops.constant(np.int64(2)))
    # The condition result must be computed in the body: a body parameter nothing feeds stops the loop early
    body = Model(
        [ov_ops.result(grown), ov_ops.result(ov_ops.constant(np.array(True)))],
        [iteration, body_q, body_k, body_v, body_cu, carried],
    )

    # As many iterations as there are segments: one fewer than the boundaries `cu_seqlens` names.
    segments = ov_ops.squeeze(
        ov_ops.subtract(ov_ops.shape_of(cu_seqlens, "i64"), one), ov_ops.constant(np.array([0], dtype=np.int64))
    )
    loop = ov_ops.loop(segments, ov_ops.constant(np.array(True)))
    loop.set_function(body)
    # `[iteration parameter, condition result]` — which body ports carry the loop's own bookkeeping.
    loop.set_special_body_ports([0, 1])
    for parameter, source in ((body_q, query), (body_k, key), (body_v, value), (body_cu, cu_seqlens.output(0))):
        loop.set_invariant_input(parameter, source)
    blank = ov_ops.broadcast(ov_ops.constant(0.0, dtype=element_type), ov_ops.shape_of(query, output_type="i64"))
    loop.set_merged_input(carried, blank.output(0), grown.output(0))
    loop.validate_and_infer_types()
    written = ov_ops.squeeze(loop.get_iter_value(grown.output(0), -1), axis0)
    return [ov_ops.transpose(written, heads_first).output(0)]


def _convert_grouped_mm(context):
    """Convert ``aten._grouped_mm`` / ``transformers.grouped_mm_fallback`` to OV ops.

    ``out[offs[g-1]:offs[g]] = mat_a[offs[g-1]:offs[g]] @ mat_b[g]``, unrolled over the static expert count ``G``.
    """
    mat_a = context.get_input(0)
    mat_b = context.get_input(1)
    offs = context.get_input(2)

    G = mat_b.get_partial_shape()[0].get_length()
    offs_i64 = ov_ops.convert(offs, "i64")
    axes_0 = ov_ops.constant(np.array([0], dtype=np.int64))
    step_1 = ov_ops.constant(np.array([1], dtype=np.int64))
    prev_end = ov_ops.constant(np.array([0], dtype=np.int64))

    outputs = []
    for g in range(G):
        g_lo = ov_ops.constant(np.array([g], dtype=np.int64))
        g_hi = ov_ops.constant(np.array([g + 1], dtype=np.int64))
        end = ov_ops.slice(offs_i64, g_lo, g_hi, step_1, axes_0)  # (1,) — offs[g]
        a_g = ov_ops.slice(mat_a, prev_end, end, step_1, axes_0)  # (n_g, K)
        w_g_3d = ov_ops.slice(mat_b, g_lo, g_hi, step_1, axes_0)  # (1, K, N)
        w_g = ov_ops.squeeze(w_g_3d, axes_0)  # (K, N)
        outputs.append(ov_ops.matmul(a_g, w_g, transpose_a=False, transpose_b=False).output(0))
        prev_end = end

    return [ov_ops.concat(outputs, axis=0).output(0)]


def _convert_empty_permuted(context):
    """Convert ``aten.empty_permuted`` to a zero ``Broadcast`` of the requested shape."""
    size = context.get_input(0)
    # f32 is safe: in the MoE path the result feeds index ops or is overwritten before any read.
    zero = ov_ops.constant(np.float32(0.0))
    return [ov_ops.broadcast(zero, size).output(0)]


def _convert_index_add(context):
    """Convert ``aten.index_add`` as a sum-reduced ``ScatterElementsUpdate``.

    OV's translator expects 5 inputs and fails when ``alpha`` is defaulted (t5gemma, speecht5).
    """
    data = context.get_input(0)
    dim = int(context.get_values_from_const_input(1))
    index = context.get_input(2)
    source = context.get_input(3)
    # Fold a non-default ``alpha`` (FX input 4) into ``source``.
    if context.get_input_size() > 4 and context.get_input(4).get_node().get_type_name() == "Constant":
        alpha = context.get_values_from_const_input(4)
        if alpha != 1:
            source = ov_ops.multiply(
                source, ov_ops.convert(ov_ops.constant(np.array(alpha)), source.get_element_type())
            )
    # Broadcast the 1-D index to ``source``'s shape along ``dim``.
    src_shape = ov_ops.shape_of(source, output_type="i64")
    ndim = source.get_partial_shape().rank.get_length()
    ones = [1] * ndim
    ones[dim] = -1
    index_reshaped = ov_ops.reshape(
        ov_ops.convert(index, "i64"),
        ov_ops.constant(np.array(ones, dtype=np.int64)),
        special_zero=False,
    )
    index_bcast = ov_ops.broadcast(index_reshaped, src_shape)
    return [
        ov_ops.scatter_elements_update(
            data, index_bcast, source, ov_ops.constant(np.int64(dim)), reduction="sum"
        ).output(0)
    ]


def _convert_view_as_real(context):
    """Identity: ``_convert_complex`` already represents complex tensors as ``[..., 2]`` real."""
    return [context.get_input(0)]


def _convert_conj(context):
    """Convert ``aten._conj`` by negating the imaginary half of the ``[..., 2]`` representation."""
    data = context.get_input(0)
    axes_neg1 = ov_ops.constant(np.array([-1], dtype=np.int64))
    real_part = ov_ops.gather(data, ov_ops.constant(np.int64(0)), axes_neg1)
    imag_part = ov_ops.gather(data, ov_ops.constant(np.int64(1)), axes_neg1)
    neg_imag = ov_ops.negative(imag_part)
    return [
        ov_ops.concat(
            [ov_ops.unsqueeze(real_part, axes_neg1), ov_ops.unsqueeze(neg_imag, axes_neg1)],
            axis=-1,
        ).output(0)
    ]


def _convert_bitwise_not(context):
    """Convert ``aten.bitwise_not`` to ``LogicalNot`` on a boolean view.

    OV's own translator leaves a ``torch.sym_float`` call on the dynamic dims behind (deformable_detr).
    """
    data = context.get_input(0)
    return [ov_ops.logical_not(ov_ops.convert(data, "boolean")).output(0)]


def _convert_layer_norm(context):
    """Convert ``aten.layer_norm`` to ``MVN`` + affine.

    OV's translator emits ``native_layer_norm``, whose unused outputs are failing ``torch::None`` constants.
    """
    data = context.get_input(0)
    normalized_shape = context.get_values_from_const_input(1)
    weight = context.get_input(2)
    bias = context.get_input(3)
    eps = float(context.get_values_from_const_input(4)) if context.get_input_size() > 4 else 1e-5
    ndim = data.get_partial_shape().rank.get_length()
    axes_len = len(normalized_shape) if hasattr(normalized_shape, "__len__") else 1
    axes = ov_ops.constant(np.array(list(range(ndim - axes_len, ndim)), dtype=np.int64))
    normalized = ov_ops.mvn(data, axes, normalize_variance=True, eps=eps, eps_mode="inside_sqrt")
    scaled = ov_ops.multiply(normalized, weight)
    shifted = ov_ops.add(scaled, bias)
    return [shifted.output(0)]


def _convert_to_copy(context):
    """Convert ``aten._to_copy`` to an OV ``Convert``.

    OV's translator throws on ``complex64``; with the ``[..., 2]`` real representation that cast is a no-op.
    Real casts must stay (``bitwise_and`` needs a real ``bool`` mask).
    """
    data = context.get_input(0)
    if not context.has_attribute("dtype"):
        return [data]
    try:
        dtype = context.get_attribute("dtype")
    except Exception:
        # Complex dtypes throw.
        return [data]
    if dtype is None:
        return [data]
    return [ov_ops.convert(data, dtype).output(0)]


def _convert_bmm(context):
    """Translate ``aten.bmm``, shielding softmax-fed ones from OV's SDPA fusion.

    The fusion mis-shapes ``bmm -> softmax -> bmm`` with batch and heads flattened (SpeechT5's relative-position
    attention). A runtime-dependent ``Reshape(x, ShapeOf(x))`` no-op blocks it and is cleaned up later.
    """
    a, b = context.get_input(0), context.get_input(1)
    product = ov_ops.matmul(a, b, transpose_a=False, transpose_b=False)
    if a.get_node().get_type_name() != "Softmax":
        return [product.output(0)]
    identity = ov_ops.reshape(product, ov_ops.shape_of(product, output_type="i64"), special_zero=False)
    return [identity.output(0)]


def _convert_sdpa(context):
    """Convert ``aten.scaled_dot_product_attention``, casting integer masks to boolean.

    ``aten.expand`` promotes bool masks to ``i64`` under CUDA export, which OV's SDPA rejects.
    """
    q, k, v = context.get_input(0), context.get_input(1), context.get_input(2)
    # A ``None`` FX arg reaches the extension as an unconverted ``PtFrameworkNode``.
    mask = None
    if context.get_input_size() > 3:
        candidate = context.get_input(3)
        if candidate.get_node().get_type_name() != "PtFrameworkNode":
            # Float additive masks pass through; casting them to bool would destroy them.
            mask = candidate if candidate.get_element_type().is_real() else ov_ops.convert(candidate, "boolean")
    is_causal = False
    if context.get_input_size() > 5:
        # ``is_causal`` is a positional FX input (arg 5), not a node attribute.
        if context.get_input(5).get_node().get_type_name() == "Constant":
            is_causal = bool(context.get_values_from_const_input(5))
    kwargs = {"causal": is_causal}
    if mask is not None:
        kwargs["attention_mask"] = mask
    # Stated explicitly: the CPU plugin fuses attention with its KV cache only for the five-input form.
    head_dim = q.get_partial_shape()[-1]
    if head_dim.is_static:
        # In the query's type: OV's SDPA won't merge an ``f32`` scale with ``bf16`` queries.
        scale = np.array(head_dim.get_length() ** -0.5).astype(q.get_element_type().to_dtype())
        kwargs["scale"] = ov_ops.constant(scale, q.get_element_type())
    return [ov_ops.scaled_dot_product_attention(q, k, v, **kwargs).output(0)]


def _convert_complex(context):
    """Convert ``aten.complex(real, imag)`` by stacking them as a trailing ``[..., 2]`` axis."""
    real = context.get_input(0)
    imag = context.get_input(1)
    stacked = ov_ops.concat(
        [ov_ops.unsqueeze(real, ov_ops.constant(-1)), ov_ops.unsqueeze(imag, ov_ops.constant(-1))],
        axis=-1,
    )
    return [stacked.output(0)]


# ── SymInt builtin translations ─────────────────────────────────────────────
# torch.export keeps Python math on SymInts (``a % b``, ``min(a, b)``) as builtin ``call_function`` nodes with no
# OV translation; each gets one keyed on its ``str(target)``.


def _convert_sym_binop(op):
    """Build a 2-arg translator for SymInt binary builtins, promoting mixed int/float operands to float.

    ``mod`` must map to ``floor_mod``: Python's ``%`` is floored, OV's ``Mod`` truncates (LongT5).
    """

    def _convert(context):
        a, b = context.get_input(0), context.get_input(1)
        a_type, b_type = a.get_element_type(), b.get_element_type()
        if a_type != b_type:
            if a_type.is_integral() and not b_type.is_integral():
                a = ov_ops.convert_like(a, b)
            elif b_type.is_integral() and not a_type.is_integral():
                b = ov_ops.convert_like(b, a)
        return [op(a, b).output(0)]

    return _convert


def _convert_sym_unop(op, *, cast_to_i64=False):
    """Build a 1-arg translator for SymInt unary builtins.

    ``cast_to_i64``: OV's ``floor``/``ceiling`` keep a float type where Python returns an int (focalnet)."""

    def _convert(context):
        out = op(context.get_input(0))
        if cast_to_i64:
            out = ov_ops.convert(out, "i64")
        return [out.output(0)]

    return _convert


def _float_operands(context):
    """Both operands as floats — OV's ``Divide`` truncates on two integers, so a following
    ``floor`` is a no-op and negative operands round the wrong way (``-200 // 64`` → ``-3``)."""
    operands = [context.get_input(0), context.get_input(1)]
    return [ov_ops.convert(x, "f32") if x.get_element_type().is_integral() else x for x in operands]


def _convert_sym_floordiv(context):
    """``a // b`` over SymInts as ``floor(a / b)`` in float, cast to i64.

    Truncating division breaks the ceil-div idiom ``-(-x // n)`` (minimax_m3_vl's ``num_key_blocks``)."""
    a, b = _float_operands(context)
    return [ov_ops.convert(ov_ops.floor(ov_ops.divide(a, b)), "i64").output(0)]


def _convert_sym_truediv(context):
    """``a / b`` over SymInts as float division, like Python (granite_speech's ``ceil(seq / chunk)``)."""
    return [ov_ops.divide(*_float_operands(context)).output(0)]


_OV_CONVERSION_EXTENSIONS: list[Any] = []
if is_openvino_available():
    _OV_CONVERSION_EXTENSIONS.extend(
        [
            ConversionExtension("aten._grouped_mm.default", _convert_grouped_mm),
            ConversionExtension("transformers.grouped_mm_fallback.default", _convert_grouped_mm),
            ConversionExtension("aten.empty_permuted.default", _convert_empty_permuted),
            ConversionExtension("aten.index_add.default", _convert_index_add),
            ConversionExtension("aten.bmm.default", _convert_bmm),
            ConversionExtension("aten.complex.default", _convert_complex),
            ConversionExtension("aten.view_as_real.default", _convert_view_as_real),
            ConversionExtension("aten._conj.default", _convert_conj),
            ConversionExtension("aten._to_copy.default", _convert_to_copy),
            ConversionExtension("aten.layer_norm.default", _convert_layer_norm),
            ConversionExtension("aten.scaled_dot_product_attention.default", _convert_sdpa),
            ConversionExtension("torch_attn._varlen_attn.default", _convert_varlen_attn),
            ConversionExtension("aten.bitwise_not.default", _convert_bitwise_not),
            # SymInt builtins — see comment block above.
            ConversionExtension("<built-in function add>", _convert_sym_binop(ov_ops.add)),
            ConversionExtension("<built-in function sub>", _convert_sym_binop(ov_ops.subtract)),
            ConversionExtension("<built-in function mul>", _convert_sym_binop(ov_ops.multiply)),
            ConversionExtension("<built-in function truediv>", _convert_sym_truediv),
            ConversionExtension("<built-in function floordiv>", _convert_sym_floordiv),
            ConversionExtension("<built-in function mod>", _convert_sym_binop(ov_ops.floor_mod)),
            ConversionExtension("<built-in function pow>", _convert_sym_binop(ov_ops.power)),
            ConversionExtension("<built-in function floor>", _convert_sym_unop(ov_ops.floor, cast_to_i64=True)),
            ConversionExtension("<built-in function ceil>", _convert_sym_unop(ov_ops.ceiling, cast_to_i64=True)),
            ConversionExtension("<built-in function min>", _convert_sym_binop(ov_ops.minimum)),
            ConversionExtension("<built-in function max>", _convert_sym_binop(ov_ops.maximum)),
            # These reprs are address-based, so register them by their runtime str
            ConversionExtension(str(torch.sym_float), _convert_sym_unop(lambda x: ov_ops.convert(x, "f32"))),
            ConversionExtension(str(torch.sym_min), _convert_sym_binop(ov_ops.minimum)),
            ConversionExtension(str(torch.sym_max), _convert_sym_binop(ov_ops.maximum)),
        ]
    )
