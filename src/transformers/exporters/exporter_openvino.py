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
"""OpenVINO exporter.

Extends [`DynamoExporter`] with the stages that turn an ``ExportedProgram`` into an
``openvino.Model``:

1. **Torch patches** (``apply_patches("openvino")``): reversibly swap ``torch`` ops the OV
   frontend can't lower (``torch.histc``, ``torch.searchsorted``, …) with decomposed
   equivalents during the trace. The trace itself runs under ``torch.no_grad()`` so
   modeling-internal grad regions don't become HigherOrderOp subgraphs.
2. **Dynamo trace** (inherited from [`DynamoExporter`]): signature patch, model patches,
   pytree registration, dynamic shapes, state cleanup — same as for any other backend.
3. **Graph preparation**: run OV's own decomposition pass up front
   (``_run_openvino_decompositions``), then repair the resulting graph in place — FX program
   fixes, output-arg deduplication, per-node FX fixes, and bare-name renames. Preparing the
   decomposed graph ourselves is what makes these fixes stick: handing the raw
   ``ExportedProgram`` to ``convert_model`` would re-run decompositions internally and
   regenerate the graph.
4. **Conversion**: the prepared module is decoded via ``TorchFXPythonDecoder`` and handed to
   ``openvino.convert_model`` together with the custom ``ConversionExtension``\\ s for ops
   without a built-in OV lowering. Ports are then renamed to their dotted leaf paths and
   non-tensor inputs repaired.
5. **Stateful transformation** (``OpenVINOConfig.stateful``, on by default): fold
   round-tripped state tensors (KV cache, SSM states, …) into internal OV variables with a
   fused ``beam_idx`` reorder. Optionally written to disk via ``openvino.save_model`` when
   ``OpenVINOConfig.output_path`` is set.
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
    get_leaf_tensors,
    leaf_name,
    register_fx_node_fix,
    register_fx_program_fix,
    register_patch,
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
    artifact_suffix = ".xml"
    # A variable's `ReadValue` computes the cross-attention cache on a sequence's first step and keeps it.
    decoder_writes_cross_cache = True

    def export_artifact(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, Any],
        config: OpenVINOConfig | dict[str, Any],
    ) -> tuple[openvino.Model, dict]:
        if isinstance(config, dict):
            config = OpenVINOConfig(**config)
        elif type(config) is not OpenVINOConfig:
            raise TypeError(f"Expected config to be an OpenVINOConfig or dict, got {type(config)}")

        # ``torch.no_grad()``: with grad enabled, every modeling-internal ``torch.no_grad()``
        # region (frozen towers, VQ-VAEs) traces as a ``wrap_with_set_grad_enabled``
        # HigherOrderOp subgraph, which OV's frontend can't lower.
        with torch.no_grad(), patch_model_outputs(model) as (inputs_names, outputs_names), apply_patches("openvino"):
            exported_program, metadata = super().export_artifact(model, sample_inputs, config=config)

        exported_program, graph_module = _fix_exported_program(exported_program)
        ov_model = _convert_to_openvino(graph_module)

        inputs_names = [name for name in inputs_names if name in get_leaf_tensors(sample_inputs)]
        inputs_names, outputs_names = disambiguate_io_names(inputs_names, outputs_names)
        _rename_model_ports(ov_model, graph_module, inputs_names, outputs_names)

        if config.dynamic:
            # Only where the trace left axes symbolic — a static export already declares every one of them,
            # and reshaping a graph that is wholly pinned only disturbs what it settled.
            _pin_static_input_axes(ov_model, graph_module, inputs_names)

        if config.stateful:
            _make_stateful(ov_model, exported_program, graph_module, sample_inputs, inputs_names, outputs_names)

        if config.compress_to_fp16:
            # Here rather than at save time, so what runs in memory is what lands on disk — and so writing an
            # artifact out stays a matter of the format alone, with no recipe to carry along.
            compress_model_transformation(ov_model)

        if config.output_path is not None:
            # Whatever precision the model holds by now: `save_model` halves `f32` weights on its own, which
            # would compress an export that never asked for it and compress twice one that did.
            openvino.save_model(ov_model, config.output_path, compress_to_fp16=False)

        # What the trace meant, beside what the IR declares: an `openvino.Model` reports port names, shapes
        # and element types, and nothing about precision, cache layout or mask rank. It travels in the
        # artifacts rather than in the file, since OV's `rt_info` is not the place for it.
        return ov_model, metadata

    @classmethod
    def save_artifact(cls, artifact, path) -> None:
        """`save_model` writes the `.xml` graph and the `.bin` weights beside it, named after the `.xml`.

        Written at the precision the model holds: `save_model` would otherwise halve `f32` weights on its
        own, and whether that was wanted was settled at export.
        """
        openvino.save_model(artifact, path, compress_to_fp16=False)


# ── Conversion helpers ──────────────────────────────────────────────────────
# Small helpers for ``OpenVINOExporter.export`` — extracted for readability and so each stage
# of the conversion has a single responsibility.


def _fix_exported_program(exported_program: ExportedProgram) -> tuple[ExportedProgram, Any]:
    """Decompose and repair the exported program, returning it with the module to convert.

    OV's own decomposition pass runs up front and the RESULT is what gets decoded — handing the
    ``ExportedProgram`` to ``convert_model`` would re-run it internally, regenerating node names and
    discarding every fix applied here. Both are returned because ``_make_stateful`` reads the program
    while the port fixes read the module.
    """
    _drop_runtime_asserts(exported_program.graph_module)
    exported_program = _run_openvino_decompositions(exported_program)
    apply_fx_program_fixes("openvino", exported_program)
    graph_module = exported_program.module()
    _deduplicate_output_args(graph_module)
    apply_fx_node_fixes("openvino", graph_module)
    _rename_bare_node_names(graph_module)
    return exported_program, graph_module


def _prepare_tensors_for_conversion(graph_module) -> None:
    """Rebind the module's tensors to host copies it owns, in float32, before OV reads them.

    OV builds every constant with ``shared_memory=True``, so the constant points into the tensor's
    buffer instead of copying it. For a tensor that is not already on the host, OV first materialises
    one with ``Tensor.numpy(force=True)`` — a temporary that is freed as soon as the constant is
    built, leaving it aimed at memory that gets reused. Weights then read back as garbage (denormals)
    and the model's outputs collapse to zero.

    Half-precision tensors are promoted to float32 in the same pass. To build a constant from a
    ``bfloat16`` tensor OV views its bits as ``float16`` (``openvino/frontend/pytorch/utils.py``), so the
    values come back as the ``float16`` reading of those bits: ``1.0`` arrives as ``1.875``. optimum-intel
    promotes 16-bit modules for the same reason; an FX graph has no module boundaries left, so every
    half-precision tensor is promoted. The CPU plugin reports no ``bf16`` capability, so this costs
    constant size and no accuracy.

    Copies are rebound on the exported module only; the parameters the caller's model holds are left
    untouched.
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

    # The dtype the graph asks for has to move with the weights. A `to(bfloat16)` or a mask built as
    # `full(..., dtype=bfloat16)` left behind would meet the promoted weights at the first op that
    # requires one type on both sides, and scaled-dot-product attention refuses to merge them. OV types a
    # node from its `tensor_meta` before its `val`, so both are promoted.
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

    # Inputs keep the type they are fed in — a half-precision `pixel_values`, or a static cache — and are
    # cast once where they enter, so the IR takes what the model took while computing in float32 inside.
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
    # Name every input port after its FX placeholder — OV may drop unused inputs, so all
    # downstream port↔placeholder matching is done by name, never positionally.
    decoder._input_signature = [node.name for node in graph_module.graph.nodes if node.op == "placeholder"]
    ov_model = openvino.convert_model(decoder, extension=_OV_CONVERSION_EXTENSIONS)
    _fix_non_tensor_inputs(ov_model, graph_module)
    return ov_model


def _lookup_by_port(port, by_name: Mapping[str, Any]):
    """Return the ``by_name`` entry keyed by one of ``port``'s tensor names, or ``None``.

    Every input port carries its placeholder's name (via ``decoder._input_signature``), so
    port↔placeholder matching is by name — OV drops unused inputs, which would shift any
    positional pairing.
    """
    return next((by_name[name] for name in port.get_names() if name in by_name), None)


def _port_named(port, names) -> bool:
    """Whether any of ``port``'s tensor names is one of ``names``."""
    return bool(port.get_names() & set(names))


def _leaf_names_by_placeholder(graph_module, inputs_names: list[str]) -> dict[str, str]:
    """Map each tensor placeholder's FX name to its dotted leaf-path name.

    Tensor placeholders appear in the graph in kwargs-leaf order — the same order
    ``patch_model_outputs`` captured ``inputs_names`` in — so the two zip together.
    """
    tensor_placeholders = [
        node
        for node in graph_module.graph.nodes
        if node.op == "placeholder" and isinstance(node.meta.get("val"), torch.Tensor)
    ]
    return dict(zip((node.name for node in tensor_placeholders), inputs_names))


def _pin_static_input_axes(ov_model: openvino.Model, graph_module, inputs_names: list[str]) -> None:
    """Declare the input axes the trace settled on a number.

    OV's decoder is handed `dynamic_shapes=True` and marks every axis symbolic, including the ones
    `torch.export` had already resolved — an m-rope `position_ids`' section count, an embedding's hidden
    size. Two axes that are only *nominally* dynamic are then indistinguishable, and the stateful
    transformation resolves the batch from whichever it reaches first: a `[sections, batch, positions]`
    position_ids answers 3, and every state the graph sizes by it is built for the wrong batch.
    """
    leaf_by_placeholder = _leaf_names_by_placeholder(graph_module, inputs_names)
    traced = {}
    for node in graph_module.graph.nodes:
        leaf = leaf_by_placeholder.get(node.name)
        value = node.meta.get("val") if leaf is not None else None
        if value is None:
            continue
        # Named as the ports are, unprefixed. Never the cache: its batch is the beam's to reorder and its
        # length the sequence's to grow, so neither is the graph's to keep.
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
    """Restore the dotted leaf-path names on the converted model's input/output ports.

    OV's PyTorch frontend doesn't support an ``output=`` argument, and ``input=`` only accepts
    Python-identifier names (no dots) — so the dotted ``get_leaf_tensors`` form is restored
    post-conversion. Input ports are matched to their leaf names through their FX placeholder;
    scalar ports (no leaf) keep their placeholder name.
    """
    leaf_names = _leaf_names_by_placeholder(graph_module, inputs_names)
    for port in ov_model.inputs:
        name = _lookup_by_port(port, leaf_names)
        if name is not None:
            port.get_tensor().set_names({name})
    # A passthrough output (e.g. T5's ``encoder_last_hidden_state`` returning the
    # ``encoder_outputs.last_hidden_state`` input untouched) shares its tensor with the input
    # port — renaming it would clobber the input name. Give the Result its own tensor by
    # routing it through a no-op ``convert_like`` first.
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

    Non-tensor forward kwargs survive ``torch.export`` as placeholders that OV's frontend
    converts to dynamic-rank Parameters — and the CPU plugin refuses to compile any Parameter
    with dynamic rank. Scalars (``logits_to_keep: int``) are pinned to a static scalar shape;
    ``None`` and string kwargs (e.g. ``attention_mask=None`` in SSM decode captures, Blip's
    ``reduction="mean"``) produce a Parameter nothing translatable consumes, which is removed.
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
# Folds round-tripped state tensors (KV cache, SSM conv/ssm states, …) into internal OV
# ``ReadValue``/``Assign`` variables so the runtime carries them across ``infer()`` calls
# instead of marshalling them through inputs/outputs on every step. State pairs are derived
# STRUCTURALLY: a leaf path that appears on both sides of the model was traced from the same
# cache leaf (``disambiguate_io_names`` marks exactly these collisions with ``input.`` /
# ``output.`` prefixes) — no name conventions, no per-model-type branching, and any cache
# layout (KV, SSM, sliding-window, hybrid) is covered by construction.

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

    A leaf path appearing on both sides is only state when it lives inside a cache object
    (per [`~exporters.exporter_dynamo.is_cache_object`]) — a plain tensor kwarg the model
    happens to return under the same name (e.g. Parakeet's downsampled ``attention_mask``)
    is a regular output.
    """
    state_roots = {key for key, value in sample_inputs.items() if is_cache_object(value)}
    # A cache the trace found empty is written, not carried: that is a prefill, which runs once and reads
    # nothing back. Folding it in would pin the variable to the length the prefill *produces* while its
    # initializer stays the empty tensor it was given, and the two cannot be reconciled under static shapes.
    # The decode graph — the one called per token, where carrying the state in-plugin is the whole point —
    # is unaffected, since its cache arrives with the prompt already in it.
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

    Beam search reorders the cache between steps (`_reorder_cache`); once state lives inside the
    model that reorder must happen inside too. The runtime passes the beam permutation as
    ``beam_idx`` and the fused ``Gather`` applies it to each state variable — for greedy decoding
    ``beam_idx = arange(batch)`` makes it the identity.

    A counter (xLSTM's ``seqlen_offset``, a sparse indexer's ``idx_cumulative_length``) or an empty buffer
    (reformer's ``buckets_cache``) has no batch axis, and gathering it by ``beam_idx`` would give it one — a
    ``[1]`` counter comes back ``[batch]``. The converted graph cannot tell: every state is declared fully
    dynamic. The trace can (`_state_init_dims`), so the ``batchless`` states it named are left alone.
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
    """Compute a per-dim init spec for every state input: ``"batch"`` (follow the main input's
    batch at runtime), ``0`` (the growing dim — empty on first inference), or a concrete length.

    The growing dim is identified from the ``ExportedProgram``'s symbolic shapes: a state input
    dim whose SymInt expression differs from the paired output's (e.g. ``s2`` vs ``s2 + s3``)
    grows across steps; dims with identical expressions pass through unchanged and are pinned to
    their length in the sample tensors (heads, head_dim, conv width, …) — deriving them from the
    data rather than from ``model.config`` keeps this model-type-agnostic.
    """
    leaf_names = _leaf_names_by_placeholder(graph_module, inputs_names)
    input_vals = {
        leaf_names[node.name]: node.meta.get("val")
        for node in graph_module.graph.nodes
        if node.op == "placeholder" and node.name in leaf_names
    }
    # Output vals are keyed by the trace-ordered leaf names, NOT by zipping OV output ports —
    # ``convert_model`` does not always preserve output order.
    node_by_name = {node.name: node for node in exported_program.graph.nodes}
    output_specs = [s for s in exported_program.graph_signature.output_specs if s.kind.name == "USER_OUTPUT"]
    output_vals = {}
    for spec, name in zip(output_specs, outputs_names):
        node = node_by_name.get(getattr(spec.arg, "name", None))
        output_vals[name] = node.meta.get("val") if node is not None else None

    # A counter (xLSTM's ``seqlen_offset``) or an empty buffer (reformer's ``buckets_cache``) has no batch
    # axis. Under dynamic shapes the batch is a symbol, and a state carries it exactly when its leading
    # axis is symbolic too; a static trace has only the sizes to compare.
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
        # A cross-attention cache passes its encoder length through unchanged, yet that length is the
        # sequence's, not the graph's: pinned to the sample's, every other source length is refused. It
        # starts empty like a growing axis — it is always written before it is read.
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
        # ``apply_make_stateful_transformation`` names each variable by concatenating the input
        # and output tensor names — key the specs the same way for the ReadValue lookup.
        init_dims[f"{input_name}{output_name}"] = dims
    return init_dims


def _freeze_batchless_states(
    ov_model: openvino.Model,
    pairs: dict[str, str],
    sample_inputs: MutableMapping[str, Any],
) -> None:
    """Replace batch-less round-tripped tensors with baked constants (in place, updating ``pairs``).

    A round-tripped tensor without a batch dim that the graph hands back unchanged (e.g. Cohere2's scalar
    ``_sliding_window_tensor``) is config-derived, not per-sequence state — there is nothing to reorder
    between beams and nothing to reset between prompts, so it becomes a graph constant instead of an OV
    variable. One the graph writes is a counter (a static layer's ``cumulative_length``, where each step's
    keys land) and stays state: baked, every step would write the slot the trace wrote.
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
    """An input whose leading axis is the batch — what the state is sized and reordered by.

    The text input is the one that always answers that. Taking whichever port comes first reads an m-rope
    `position_ids` as `[sections, batch, positions]` and takes the section count for the batch (glm4v: 4
    rather than 2), which the first state update then refuses to concatenate against.
    """
    ports = {name: port for port in ov_model.inputs for name in port.get_names()}
    for name in BATCH_INPUTS:
        if name in ports:
            return ports[name]
    return next(port for port in ov_model.inputs if not _port_named(port, exclude or []))


def _build_state_initializers(ov_model: openvino.Model, init_dims: dict[str, list]) -> None:
    """Give every state variable a zero-filled init expression so the runtime can materialise
    empty state on the first ``infer()`` without the caller providing shapes.

    The variable's declared shape is relaxed/pinned from the same dim spec first: the batch follows the
    one the graph's inputs declare, the growing dim goes dynamic, and pass-through dims get their
    concrete sample length (a ``[B,0,0,D]``-style degenerate init would break the state-update concat
    downstream).

    Exception: when a variable's update expression (its ``Assign``'s input) is fully static —
    a static trace can bake the batch into a non-growing state's update while the decoder-level
    shapes stay dynamic — the batch dim is pinned to the update's instead. Updating a dynamic
    variable from a fully static expression makes the CPU plugin insert a Reorder between the
    two descriptors, which it can't build against the variable's dynamic one.
    """
    variables = {variable.get_info().variable_id: variable for variable in ov_model.get_variables()}
    update_shapes = {sink.get_variable_id(): sink.input_value(0).get_partial_shape() for sink in ov_model.get_sinks()}
    batch_port = _batch_bearing_input(ov_model)
    batch = ov_ops.gather(
        ov_ops.shape_of(batch_port, output_type="i64"),
        ov_ops.constant([_STATE_BATCH_DIM]),
        ov_ops.constant(0),
    )
    # The batch the graph's own inputs declare, where they declare one: a state left dynamic beside a
    # static query is a pair the fused attention cannot show equal (`B == B_state`).
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
        # `"batch"` and a growing `0` are different answers: the batch is whatever the inputs declare, the
        # growing axis is nobody's until a step writes it.
        info.data_shape = openvino.PartialShape([batch_dim if d == "batch" else -1 if d == 0 else d for d in dims])
        variables[op.get_variable_id()].update(info)
        # The growing dim's zero length is emitted as ``batch - batch`` rather than a literal
        # ``[0]``: the CPU plugin fuses single-consumer init subgraphs into
        # ``ReadValueWithSubgraph`` and re-infers the state descriptor from the init's static
        # shape — a folded ``0`` gets baked in and seeding the state is then rejected.
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
    """Give each state pair a single CPU-friendly storage type, converting at the boundaries.

    ``Assign`` rejects an update whose type differs from the variable's (e.g. Parakeet's
    streaming lengths enter as i64 but are recomputed as i32), and the CPU plugin's oneDNN
    path rejects i64 state outright (xLSTM's ``seqlen_offset``). The variable stores the
    output's compute type, demoted to i32 when it would be i64; the input Parameter is retyped
    to match (with a ``Convert`` restoring the original type for its consumers) and the output
    converted before its ``Result``.
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
        # Common benign case with stateful=True as the default: encoders and prefill-only
        # exports have no round-tripped state.
        logger.debug("No round-tripped state tensors found — leaving the model stateless.")
        return

    # Init specs must be computed before freezing — removing a Parameter breaks the
    # positional placeholder↔port alignment the spec derivation relies on.
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
    """The leaf paths of every growing cross-attention cache half among `sample_inputs` (an
    `EncoderDecoderCache`'s), for a graph that attends over `encoder_outputs`. A fixed-size half is written at
    positions of a buffer allocated in full, so it is left to the ordinary state handling."""
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
    """Compute a cross-attention variable the graph writes but never reads from that write, once per sequence.

    A decode graph captured writing its own cross cache (`decoder_writes_cross_cache`) projects the encoder's
    output into it on every call and never reads the variable back. Making the projection the variable's
    initializer — `ReadValue(init=projection)`, read by the attention in its place — computes it on a
    sequence's first step only: the CPU plugin fuses a single-consumer init into `ReadValueWithSubgraph`,
    which runs it while the variable is empty (after `reset_state()`), and later steps read what it stored.
    Only the cross half of an `EncoderDecoderCache`: it is the cache whose write is the same at every step.
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
        # The CPU plugin refuses a `ReadValueWithSubgraph` feeding a `Concat` directly (t5gemma2's merged self/cross
        # attention concatenates the two keys); the beam reorder every other state carries in between compiles.
        # Only there: without a `Concat` it would keep the plugin from fusing the variable into its attention.
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
    """Spell each `ReduceMean` only `root`'s subgraph uses as `ReduceSum × 1/n`, the same value.

    The CPU plugin fuses an initializer into `ReadValueWithSubgraph` with every axis of its body dynamic, and
    an RMSNorm there (t5gemma2's cross `k_norm`, matched from its `ReduceMean` of squares) is refused for
    want of a static last dimension. The graph knows that dimension; the sum-then-scale spelling keeps the
    plugin from matching the norm inside the body, and every norm outside it stays fused.
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
    """Reconcile each ``Assign``'s update shape with its variable's shape.

    The CPU plugin refuses to reorder dynamic descriptors into the state memory, so the two
    sides must agree. When the update is fully static and the variable is not (a static trace
    bakes the batch into the update while the variable was declared batch-dynamic), the variable
    is pinned to the update's shape. Conversely, when shape inference leaves the update
    under-specified against a static variable (olmo_hybrid's rolled conv state comes out
    ``[?,32,2..]`` against a ``[?,32,3]`` variable), a ``special_zero`` Reshape pins the
    statically-known dims and copies the dynamic ones from the input. Pinning a variable
    refines shapes downstream (other updates read from it), so this iterates to a fixpoint.
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
                # The variable's shape must relax its init expression's, so the init gets the
                # same static shape (its dims were runtime-derived but numerically identical).
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
            # Only when it buys a shape the update did not already have. The comparison above is against the
            # *variable*, which a `special_zero` pin cannot always reach — so without this the next round
            # pins the pin, once per variable per round, and the graph leaves with a chain of identity
            # reshapes long enough to hide the attention from the plugin's fusion.
            if pinned.get_output_partial_shape(0) == update_shape:
                continue
            op.input(0).replace_source_output(pinned.output(0))
            changed = True
        if not changed:
            break
        ov_model.validate_nodes_and_infer_types()


# ── Graph preparation ───────────────────────────────────────────────────────
# Turns the traced `ExportedProgram` into the exact `GraphModule` handed to OV's decoder:
# decompose with OV's own table, then repair the result in place. Doing this ourselves (rather
# than letting `convert_model` decompose internally) is what makes the repairs stick.


def _drop_runtime_asserts(graph_module) -> None:
    """Drop ``_assert_tensor_metadata`` / ``_assert_scalar`` runtime asserts before the replay.

    ``_assert_tensor_metadata`` re-checks trace-time dtypes/devices and fails the replay once
    other stages have legitimately changed them. ``_assert_scalar`` lowers a ``torch._check``
    on an unbacked symint (e.g. the image-token count in ``get_placeholder_mask``) into a
    ``cast_symbool_to_symint`` + ``eq`` chain whose ``Piecewise`` result OV's ``_ModuleStackTracer``
    cannot proxy, crashing the replay (``... is not tracked with proxy``). The range facts these
    asserts encode survive on ``exported_program.range_constraints``, so dropping the nodes (and
    the now-dead symint feeders via ``eliminate_dead_code``) is safe. ``_fix_drop_assert_ops``
    still removes any that reappear post-decomposition.
    """
    for module in graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in list(module.graph.nodes):
            if node.op == "call_function" and node.target in (
                torch.ops.aten._assert_tensor_metadata.default,
                torch.ops.aten._assert_scalar.default,
            ):
                module.graph.erase_node(node)
        module.graph.eliminate_dead_code()
        module.recompile()


# Ops OV keeps for its own conversion, which we hand back to torch's. `index_copy` writes one position into
# a cache and its direct conversion broadcasts the source against the index, which a `[batch, 1, 1, dim]`
# write into a `[batch, 1, length, dim]` static cache cannot satisfy.
_DECOMPOSE_ANYWAY = frozenset({"aten.index_copy.default"})


def _run_openvino_decompositions(exported_program: ExportedProgram) -> ExportedProgram:
    """Run the same decomposition pass ``TorchFXPythonDecoder.from_exported_program`` would.

    Decomposing up front lets the FX node fixes and the bare-name renames below operate on the
    graph OV actually decodes — ``convert_model(exported_program)`` would re-run decompositions
    internally, regenerating node names and silently discarding those fixes.
    """
    decomp_table = CustomDecompTable()
    for op in ops_to_not_decompose():
        if str(op) not in _DECOMPOSE_ANYWAY:
            decomp_table.pop(op, None)
    return exported_program.run_decompositions(decomp_table)


def _deduplicate_output_args(graph_module) -> None:
    """Give repeated graph outputs their own node via a fold-resistant self-identity op.

    Two Results sharing one OV tensor crash the translate session's ``is_number`` check: the
    results-cleanup pass erases the shared tensor's numeric id on the first visit and fails
    decoding the debug alias on the second. Repeats arise when decomposition collapses the
    distinction between two output nodes (and ``aten.clone`` is no protection — OV folds it to
    identity). The copy must survive OV's neutral-constant elimination: ``add(x, 0)`` gets folded
    back to ``x`` (so a duplicate output re-aliases the original — e.g. moshi's ``depth_past_key_values``
    ports collapsing onto the main-cache state buffer and losing their names), whereas ``maximum(x, x)``
    is a self-identity with no neutral constant and survives as a distinct tensor (mirroring the
    ``logical_and(x, x)`` used for the bool branch).
    """
    # Ops OV translates as pass-through — their output IS their input's tensor, so an output
    # arg behind one of these still aliases the underlying node.
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
    """Append a numeric suffix to FX node names that lack one (in every nested graph).

    OV's PyTorch frontend strips a trailing ``_<digits>`` from each tensor name to recover the op
    kind, then validates the remainder — aborting with ``GeneralFailure: is_number(name)`` for
    bare names (``mul``, ``clone``, ``linear``). The first node of any kind in an FX graph has
    no ``_<digits>`` suffix, so the strip is a no-op and OV rejects it. HigherOrderOp bodies
    (e.g. ``wrap_with_set_grad_enabled`` around frozen vision towers) are separate
    ``GraphModule``\\ s with their own name counters, and their placeholders are internal closure
    args (not user inputs) — everything except the top-level placeholders gets the suffix.
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
# Per-node in-place rewrites applied to the `ExportedProgram` graph after the Dynamo trace
# but before `openvino.convert_model`. Each `_fix_*(gm, node) -> bool` factory is registered
# via `@register_fx_node_fix("openvino")` and returns `True` when it consumed the node
# (no further fixes run against it). Use this for OV-frontend quirks that are easier to
# repair at the FX level than to patch around at the torch op level.
#
# To add a new fix: define a `_fix_*` callable and decorate it.


@register_fx_node_fix("openvino")
def _fix_varlen_attn_getitem(gm, node):
    """Drop the `getitem` that unpacks `_varlen_attn`'s first output.

    The op declares `(out, softmax_lse, rng_state)` and the traced graph reads `[0]`. OV's frontend selects
    a port that way only for an op it has not converted; for a converted one it reads `getitem` as an index
    into the tensor and gathers row 0. Erasing it leaves the conversion free to hand back the output itself
    (`_convert_varlen_attn`), which is the only one anything reads.
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
def _fix_sym_float(gm, node):
    """``torch.sym_float`` is a no-op at the OV layer (it's a Python-level SymInt→SymFloat cast).
    Replace it with its input — affects deformable_detr, focalnet, mask2former, deepseek_ocr2.
    """
    if node.target is not torch.sym_float:
        return False
    node.replace_all_uses_with(node.args[0])
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("openvino")
def _fix_sym_min_max(gm, node):
    """Rewrite ``torch.sym_min``/``torch.sym_max`` to the built-in ``min``/``max``.

    OV's FX decoder keys translations on ``str(target)``. ``torch.sym_min`` reprs to
    ``<function sym_min at 0xADDRESS>`` — the address varies per process so no
    ``ConversionExtension`` string can match. ``min``/``max`` repr to stable
    ``<built-in function min>``/``<built-in function max>``, which we register translators
    for. The numeric behaviour is identical for SymInts.
    """
    if node.target is torch.sym_min:
        node.target = min
        return True
    if node.target is torch.sym_max:
        node.target = max
        return True
    return False


@register_fx_program_fix("openvino")
def _fix_to_dtype_layout_in_subgraphs(exported_program):
    """Rewrite every ``aten.to.{dtype,dtype_layout,device,other}`` node to ``aten._to_copy``
    (keeping only the dtype kwarg), in the top-level graph and every submodule graph.

    OV's PyTorch frontend has no ``aten.to.*`` translators at all — an unhandled variant falls
    back to a dangling ``torch::None`` constant that fails conversion (e.g. the
    ``wrap_with_set_grad_enabled`` HigherOrderOp subgraph in Chameleon's rotary path hits
    ``aten.to.dtype_layout``). Rewriting the FX target here — before conversion — lets our
    ``_convert_to_copy`` override handle every case (it also swallows complex-dtype casts)."""
    # Walk the top-level graph AND every submodule's graph (higher-order-op subgraphs).
    graphs = [exported_program.graph_module]
    graphs.extend(m for _, m in exported_program.graph_module.named_children() if hasattr(m, "graph"))
    for gm_or_submod in graphs:
        for node in list(gm_or_submod.graph.nodes):
            if node.op != "call_function":
                continue
            target = node.target
            if target is torch.ops.aten.to.dtype:
                # ``aten.to.dtype(tensor, dtype)`` — dtype is positional arg[1].
                dtype = node.args[1] if len(node.args) > 1 else node.kwargs.get("dtype")
                node.target = torch.ops.aten._to_copy.default
                node.args = (node.args[0],)
                node.kwargs = {"dtype": dtype} if dtype is not None else {}
            elif target in (torch.ops.aten.to.dtype_layout, torch.ops.aten.to.device, torch.ops.aten.to.other):
                dtype = node.kwargs.get("dtype")
                node.target = torch.ops.aten._to_copy.default
                node.args = (node.args[0],)
                node.kwargs = {"dtype": dtype} if dtype is not None else {}
        gm_or_submod.recompile()


@register_fx_node_fix("openvino")
def _fix_drop_assert_ops(gm, node):
    """Erase ``aten._assert_tensor_metadata`` / ``aten._assert_scalar`` nodes.

    ``torch.export`` inserts these as dead-code (num_users=0) runtime assertions, but OV's
    frontend translates them into ``torch::None`` constants whose downstream consumers can't
    drop them — causing ``OpConversionFailure``. They have no semantic effect on the model.
    """
    if node.target not in (torch.ops.aten._assert_tensor_metadata.default, torch.ops.aten._assert_scalar.default):
        return False
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("openvino")
def _fix_symbolic_pad(gm, node):
    """Decompose ``constant_pad_nd`` with symbolic pad amounts into ``full`` + ``cat``.

    OV's translation of the inlined pad list places a symbolic amount on the wrong axis in
    some graphs (mamba2's chunked scan pads seq by ``(chunk - seq % chunk) % chunk`` and the
    pad lands on the state dim at runtime). Building the filler explicitly sidesteps the
    list-decoding entirely; constant pads keep OV's native translation.
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
                # A negative amount crops instead of padding (mamba2's conv warmup pads by
                # ``kernel - seq``); symbolic amounts can be either at runtime, so build both a
                # ``max(amount, 0)``-sized filler and a ``min(amount, 0)``-deep crop.
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

    torch allows the index to be SMALLER than the data on non-axis dims (positions beyond the
    index's extent are simply never read); OV's ``GatherElements`` requires equal shapes except
    at the axis. The index is expanded to the data's extent (gathering redundantly) and the
    OUTPUT narrowed back — rewriting the data input instead is fragile: decomposition re-fuses
    the slice into upstream expands and the mismatch reappears (efficientloftr's fine-matching
    grid gather hits this).
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

    A size taken with `.item()` off a narrow integer tensor (hunyuan_vl's vision stack reads its grid
    out of an `int32` tensor) reaches OV as an `i32` scalar — decompositions spell that read
    `aten._local_scalar_dense`. Concatenating it with the `i64` constants that make up the rest of an
    `expand`/`view` shape then fails type validation (``Argument element types are inconsistent``).
    Widening the tensor first is value-preserving.
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

    The CPU plugin miscompiles an integer ``ReduceSum`` over the last axis when a size-1 axis sits before
    it and an eltwise op follows (``(starts[:, None] <= positions[None, :, None]).sum(-1) - 1``, BLT's patch
    ids): batch rows past the first come out wrong, while float and rank-2 reductions are right. Summing
    ``[-1, last]`` and restoring the leading shape afterwards is value-preserving.
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
    """Drop ``aten.cat([empty, x], dim)`` constructed by ``DynamicLayer`` for prefill — the empty
    operand is a rank-1 ``f32[0]`` from ``aten.detach_(lift_fresh_copy(...))``, which OV's torch
    frontend can't broadcast against the non-empty 4D operand for a ``dim=-2`` cat (it rejects
    with ``Axis -2 out of the tensor rank range [-1, 0]``). Mathematically the cat is identity
    when one operand is 0-element, so replace its uses with the non-empty operand.
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
        # ``numel() == 0`` on a compound SymInt expression trips ``GuardOnDataDependentSymNode``
        # (MinimaxM3VL — the concat operand has ``3*u0*u1*u2 + ...`` numel). Default to
        # ``False`` when we can't tell — treating the cat as non-empty keeps it in the graph,
        # which is always correct (the empty-cat optimisation just doesn't fire).
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

    OV constant-folds the ``Tile`` an expand of a lifted constant lowers to by computing
    per-axis repeats ``output_dim / input_dim`` — a zero-sized dim makes that an integer
    ``0 / 0``, which SIGFPEs the whole process (chmv2's dinov3 backbone expands its
    ``[1, 0, C]`` ``register_tokens`` when ``num_register_tokens=0``). The expanded tensor
    has no elements, so a zero-filled ``full`` is equivalent — and it translates to a
    Broadcast from a scalar, which has no zero input dims and folds safely.
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

    OV lowers ``aten.view`` to a ``Reshape`` and infers the ``-1`` dimension from the input's
    element count. When another target dim is a runtime ``sym_size`` expression, OV's shape
    inference can't reconcile the dynamic dim with the inferred ``-1`` and mis-resolves it —
    edgetam/sam3_tracker's mask decoder does ``x.view(pixel_values.shape[0], -1, 8, 8)`` on a
    ``[batch, 32, spatial]`` tensor and OV folds the ``-1`` to ``1`` while shifting the other
    axes, so the runtime Reshape sees ``(64, 1, 8, 8)`` instead of ``(2, 32, 8, 8)`` and the
    pattern product no longer matches the input. Substituting the ``-1`` with its concrete size
    from the node's traced output — a static int for every graph that hits this — removes the
    inference entirely and leaves OV a fully-determined pattern.
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
    # The trace's own answer for that axis, and only when it is a plain number: an axis inferred from a
    # count that varies (`view(batch, -1)` over a growing sequence) comes back as a symbol, and pinning it
    # would be a lie. Left inferred, a decode step's head count goes dynamic, and that is what stops the
    # CPU plugin fusing attention with its KV cache — a copy of the whole cache every step.
    resolved = out_val.shape[index]
    if not isinstance(resolved, int):
        return False
    new_shape = list(shape)
    new_shape[index] = resolved
    node.args = (node.args[0], new_shape) + tuple(node.args[2:])
    return True


@register_fx_node_fix("openvino")
def _fix_index_put_none_indices(gm, node):
    """Rewrite ``aten.index_put`` with ``None`` index entries into a broadcast ``where``.

    ``x[:, :, idx] = value`` traces as ``index_put(x, [None, None, idx], value)``; OV's frontend
    turns each ``None`` into a ``torch::None`` constant it can't translate (chameleon masks image
    tokens out of its logits this way). For a single 1-D index tensor on one dim (the rest ``None``)
    and a scalar value, this is equivalent to marking that dim's ``idx`` positions with a boolean
    mask and ``where``-ing the value in — built from ``arange``/``eq``/``any``/``where``, all of
    which translate cleanly. Non-scalar values or multi-index puts fall through unchanged.
    """
    if node.target not in (torch.ops.aten.index_put.default, torch.ops.aten.index_put_.default):
        return False
    if len(node.args) < 3:
        return False
    self_arg, indices, values = node.args[0], node.args[1], node.args[2]
    accumulate = node.args[3] if len(node.args) > 3 else node.kwargs.get("accumulate", False)
    if accumulate or not isinstance(indices, (list, tuple)):
        return False
    non_none = [(dim, ix) for dim, ix in enumerate(indices) if ix is not None]
    # Only the "some `None`s + exactly one 1-D index tensor" pattern; skip fully-explicit puts
    # (OV lowers those) and multi-index puts.
    if len(non_none) != 1 or len(indices) == len(non_none):
        return False
    dim, idx = non_none[0]
    self_val = self_arg.meta.get("val")
    idx_val = idx.meta.get("val") if hasattr(idx, "meta") else None
    values_val = values.meta.get("val") if hasattr(values, "meta") else None
    if self_val is None or idx_val is None or idx_val.ndim != 1:
        return False
    if values_val is None or values_val.numel() != 1:  # scalar / broadcast value only
        return False
    size = self_val.shape[dim]
    if not isinstance(size, int):
        return False
    with gm.graph.inserting_before(node):
        iota = gm.graph.call_function(torch.ops.aten.arange.default, args=(size,), kwargs={"device": self_val.device})
        iota_u = gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(iota, 1))
        idx_u = gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(idx, 0))
        eq = gm.graph.call_function(torch.ops.aten.eq.Tensor, args=(iota_u, idx_u))
        mask = gm.graph.call_function(torch.ops.aten.any.dim, args=(eq, 1))
        broadcast_shape = [1] * self_val.ndim
        broadcast_shape[dim] = size
        mask = gm.graph.call_function(torch.ops.aten.view.default, args=(mask, broadcast_shape))
        result = gm.graph.call_function(torch.ops.aten.where.self, args=(mask, values, self_arg))
        result.meta.update(node.meta)
    node.replace_all_uses_with(result)
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("openvino")
def _fix_index_put_bool_mask(gm, node):
    """Rewrite ``aten.index_put`` with a single boolean-mask index into a broadcast ``where``.

    ``x[bool_mask] = values`` (e.g. t5gemma2 swapping in an end-of-image embedding where
    ``input_ids == eoi_token``) traces as ``index_put(x, [bool_mask], values)``; OV lowers the
    boolean advanced index through a ``nonzero``-style dynamic gather its frontend can't convert
    (``SequenceMark`` OpConversionFailure). When ``bool_mask`` indexes ``x``'s leading dims and
    ``values`` broadcasts over the trailing ones, this equals ``where(mask[..., None], values, x)``
    — pure elementwise, no dynamic indexing. Flattened per-row values (which ``where`` can't
    express) are left untouched.
    """
    if node.target not in (torch.ops.aten.index_put.default, torch.ops.aten.index_put_.default):
        return False
    if len(node.args) < 3:
        return False
    self_arg, indices, values = node.args[0], node.args[1], node.args[2]
    accumulate = node.args[3] if len(node.args) > 3 else node.kwargs.get("accumulate", False)
    if accumulate or not isinstance(indices, (list, tuple)) or len(indices) != 1 or indices[0] is None:
        return False
    mask = indices[0]
    self_val = self_arg.meta.get("val")
    mask_val = mask.meta.get("val") if hasattr(mask, "meta") else None
    values_val = values.meta.get("val") if hasattr(values, "meta") else None
    if self_val is None or mask_val is None or getattr(mask_val, "dtype", None) != torch.bool:
        return False
    # The mask must cover the leading dims and the value must fit the trailing (non-mask) dims —
    # otherwise ``values`` is a flattened selected-rows tensor that a broadcast ``where`` can't
    # reproduce.
    if mask_val.ndim > self_val.ndim or values_val is None or values_val.ndim > self_val.ndim - mask_val.ndim:
        return False
    with gm.graph.inserting_before(node):
        broadcast_mask = mask
        for _ in range(self_val.ndim - mask_val.ndim):
            broadcast_mask = gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(broadcast_mask, -1))
        result = gm.graph.call_function(torch.ops.aten.where.self, args=(broadcast_mask, values, self_arg))
        result.meta.update(node.meta)
    node.replace_all_uses_with(result)
    gm.graph.erase_node(node)
    return True


# ── Torch patches ───────────────────────────────────────────────────────────
# Each `_patch_*(original)` factory is registered via `@register_patch("openvino", path)`
# and reversibly swaps a `torch` op the OV frontend can't lower with a decomposed
# equivalent. Reverted on exit by `apply_patches("openvino")`.
#
# To add a new patch: define a `_patch_*` factory and decorate it.


@register_patch("openvino", "torch.nn.functional.layer_norm")
def _patch_layer_norm(original):
    """Substitute identity ``weight=ones``/``bias=zeros`` when either is ``None``.

    OV's frontend records a ``torch::None`` constant for any unwired optional, then refuses to
    convert it (``None constant cannot be converted to OpenVINO opset``). LayerNorm without
    affine still computes ``(x - mean) / sqrt(var + eps)``; passing identity tensors keeps the
    math unchanged and gives OV concrete operands. Affects Chameleon (no-affine RMSNorm path)
    and any model that calls ``F.layer_norm(..., weight=None, bias=None)``.
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
    """Carry `antialias=True` resampling into the graph as explicit weights.

    OV's ``Interpolate`` silently ignores ``antialias``, returning bit-for-bit the *non*-antialiased
    result, so a model that resamples with it (siglip2's position embeddings) exports subtly wrong.
    Antialiased resampling is linear and separable: each output sample is a normalised triangle-filter
    average of the inputs under its footprint, widened to the downsampling ratio. The weights depend only
    on the two extents, so they are built here and applied as two matmuls that OV reproduces exactly --
    a real ``interpolate`` call would instead leave an `aten._upsample_bilinear2d_aa` node OV cannot
    convert. They are built from tensor ops throughout, so an extent read out of a tensor (siglip2 takes
    its target from `spatial_shapes`) stays symbolic rather than guarding on an unbacked size.
    """

    def axis_weights(size_in, size_out, dtype, device):
        in_index = torch.arange(size_in, dtype=torch.float32, device=device)
        out_index = torch.arange(size_out, dtype=torch.float32, device=device)
        # The ratio is carried into the graph as a tensor: OV miscompiles a `Divide` of two reduced
        # extents when it feeds only internal nodes (it comes out inverted), and reading the extents as
        # Python numbers would instead guard on an unbacked size.
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
    """Pre-expand K/V to Q's head count before calling SDPA.

    Also clamps the additive mask so fully-masked rows stay finite, and zeroes those rows to match
    the fused kernels torch runs them through.

        OV's ``opset13::ScaledDotProductAttention`` op rejects GQA shapes (e.g. Q=[B,4,T,D],
        K/V=[B,2,T,D]) with ``Key input shape not compatible with other inputs``. Repeating K/V via
        ``repeat_interleave`` on the head axis keeps the math identical and gives OV matching shapes.
    """

    def patch(query, key, value, attn_mask=None, *args, **kwargs):
        # OV's SDPA diverges from aten on a *boolean* mask and returns NaN for fully masked rows
        # (legit under left padding + causal), poisoning the batch entry
        # (https://github.com/openvinotoolkit/openvino/issues/31630). Pass an additive mask with a
        # finite dtype minimum instead, like optimum-intel does.
        unattended = None
        if attn_mask is not None:
            masked_value = torch.finfo(query.dtype).min
            if attn_mask.dtype == torch.bool:
                unattended = ~attn_mask.any(dim=-1, keepdim=True)
                attn_mask = torch.where(
                    attn_mask,
                    torch.zeros((), dtype=query.dtype, device=attn_mask.device),
                    torch.full((), masked_value, dtype=query.dtype, device=attn_mask.device),
                )
            else:
                attn_mask = attn_mask.clamp_min(masked_value)
                unattended = attn_mask.amax(dim=-1, keepdim=True) <= masked_value
        q_heads, k_heads = query.shape[-3], key.shape[-3]
        if q_heads != k_heads and q_heads % k_heads == 0:
            reps = q_heads // k_heads
            key = key.repeat_interleave(reps, dim=-3)
            value = value.repeat_interleave(reps, dim=-3)
        # OV's SDPA op takes no `scale` from the traced graph (the aten node's scalar `scale` isn't
        # surfaced as a frontend input), so it always applies its own default ``head_dim**-0.5``. Models
        # with a non-default scale (T5-family fold ``1/sqrt(d)`` into init and pass ``scale=1.0``) would be
        # silently mis-scaled. Fold the intended scale into the query so OV's default gives the right total.
        scale = kwargs.pop("scale", None)
        if scale is not None:
            default_scale = query.shape[-1] ** -0.5
            if scale != default_scale:
                query = query * (scale / default_scale)
        attn_output = original(query, key, value, attn_mask, *args, **kwargs)
        # A row that masks every key has no defined value: OV returns the uniform average the mask
        # describes, torch's fused kernels write zeros. Zero them so the export answers like eager.
        if unattended is not None:
            attn_output = torch.where(
                unattended, torch.zeros((), dtype=attn_output.dtype, device=attn_output.device), attn_output
            )
        return attn_output

    return patch


@register_patch(
    "openvino",
    "transformers.models.falcon_mamba.modeling_falcon_mamba.mamba_selective_scan",
    "transformers.models.jamba.modeling_jamba.mamba_selective_scan",
    "transformers.models.mamba.modeling_mamba.mamba_selective_scan",
    "transformers.models.zamba.modeling_zamba.mamba_selective_scan",
)
def _patch_mamba_selective_scan(original):
    """Keep an SSM scan sequential, whatever device the model sits on.

    Neither of `associative_scan`'s combine modes converts. The `generic` mode a cpu tensor selects runs
    the combine under `vmap`, whose batch dims reach `run_decompositions` as tensors that "escaped from
    inside a function being vmapped"; the `pointwise` mode a cuda tensor selects traces, but OV's frontend
    cannot build the scan's subgraph (a `getitem` past the end of its one-element list, and a broadcast the
    body's shapes do not merge on). ONNX takes the `pointwise` mode, which is why its patch keeps it on cuda.
    """

    def patch(hidden_states, *args, **kwargs):
        kwargs["use_associative_scan"] = False
        return original(hidden_states, *args, **kwargs)

    return patch


@register_patch("openvino", "torch.matmul", "torch.Tensor.matmul")
def _patch_matmul(original):
    """Flatten a two-axis batch before `MatMul`, which folds the leading axes into one regardless.

    OV folds every leading axis into a single batch and then requires both operands to agree on it, so
    m-rope's `[sections, batch, dim, 1] @ [sections, batch, 1, positions]` reaches it as `3` against `3*2`
    and is refused. Folding the pair by hand says the same product with one batch axis both sides share.
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
    """Rewrite qwen3-vl deepstack injection to avoid boolean-mask indexing.

    ``_deepstack_process`` does ``hidden_states[visual_pos_masks] += visual_embeds``, whose selected
    length is data-dependent — OV can't trace the dynamic shape. Rebuild the per-position visual
    features with ``cumsum`` + ``index_select`` (a Gather OV keeps dynamic) and add them through a
    ``float(mask)`` multiply, which is numerically identical.
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
    """Build the downsampled attention mask with a broadcast comparison instead of a scatter.

    The wav2vec2-family ``_get_feature_vector_attention_mask`` sets a one-hot at ``output_lengths - 1``
    via integer advanced-index assignment, then flip/cumsum/flip to fill the earlier positions — OV
    can't convert the ``aten.index_put`` (integer indices become an unconvertible ``SequenceMark``).
    ``arange(seq) < output_lengths`` is identical (every position before each row's length is attended)
    and export-clean.
    """

    def patch(self, feature_vector_length, attention_mask, add_adapter=None):
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)
        positions = torch.arange(feature_vector_length, device=attention_mask.device)
        return positions.unsqueeze(0) < output_lengths.unsqueeze(1)

    return patch


@register_patch("openvino", "torch.empty_permuted")
def _patch_empty_permuted(original):
    """Replace ``torch.empty_permuted(size, physical_layout, ...)`` with plain ``torch.empty(size, ...)``.

    OV's frontend has no ``aten.empty_permuted`` lowering. The op exists only to hint a memory
    layout (stride) — the values are uninitialised either way, and downstream reads see the same
    logical content. ``torch.empty`` is enough.
    """

    def patch(size, physical_layout, **kwargs):
        return torch.empty(size, **kwargs)

    return patch


@register_patch("openvino", "torch.polar")
def _patch_polar(original):
    """Build ``polar(abs, angle)`` as ``complex(abs*cos(angle), abs*sin(angle))``.

    OV has no ``aten.polar`` lowering. Euler's formula gives the same result through ops the
    frontend already supports.
    """

    def patch(abs, angle):
        return torch.complex(abs * angle.cos(), abs * angle.sin())

    return patch


def _rotate_half_pairs(pairs: torch.Tensor) -> torch.Tensor:
    """``rotate_half`` for interleaved re/im pairs: swap each pair and negate the imaginary part."""
    real, imag = pairs[..., 0], pairs[..., 1]
    return torch.stack((-imag, real), dim=-1)


def _apply_rotary_pos_emb_pairs(x: torch.Tensor, freqs_pairs: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` by ``freqs_pairs``, both viewed as ``[..., d/2, 2]`` re/im pairs.

    The complex multiply ``(a+bi)(c+di)`` these models write is the same ``x * cos + rotate(x) * sin``
    that [`~models.llama.modeling_llama.apply_rotary_pos_emb`] applies, with the pair-wise rotation
    standing in for ``rotate_half`` because the pairs are interleaved rather than split in halves.
    """
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    cos, sin = freqs_pairs[..., 0:1], freqs_pairs[..., 1:2]
    return (pairs * cos + _rotate_half_pairs(pairs) * sin).flatten(3).type_as(x)


@register_patch("openvino", "transformers.models.deepseek_v2.modeling_deepseek_v2.apply_rotary_emb")
def _patch_deepseek_rotary_emb(original):
    """Rewrite complex-arithmetic RoPE with the equivalent real re/im-pair math.

    The traced ``view_as_complex(x) * freqs_cis`` mixes OV's native ``ComplexTypeMark``
    representation with the ``[..., 2]`` real-pair one our ``aten.complex`` extension emits
    (via the ``torch.polar`` patch) — the mul can't reconcile the two. Keeping the whole
    rotation in real arithmetic confines traced complex ops to ``complex``/``view_as_real``,
    which the extensions lower consistently.
    """

    def patch(xq, xk, freqs_cis):
        freqs_pairs = torch.view_as_real(freqs_cis).unsqueeze(1).to(xq.device)
        return _apply_rotary_pos_emb_pairs(xq, freqs_pairs), _apply_rotary_pos_emb_pairs(xk, freqs_pairs)

    return patch


@register_patch("openvino", "transformers.models.llama4.modeling_llama4.apply_rotary_emb")
def _patch_llama4_rotary_emb(original):
    """Same real-pair rewrite as ``_patch_deepseek_rotary_emb`` for llama4's text RoPE."""

    def patch(xq, xk, freqs_cis):
        freqs_pairs = torch.view_as_real(freqs_cis)[:, :, None, :, :]
        return _apply_rotary_pos_emb_pairs(xq, freqs_pairs), _apply_rotary_pos_emb_pairs(xk, freqs_pairs)

    return patch


@register_patch("openvino", "transformers.models.llama4.modeling_llama4.vision_apply_rotary_emb")
def _patch_llama4_vision_rotary_emb(original):
    """Same real-pair rewrite as ``_patch_deepseek_rotary_emb`` for llama4's vision RoPE."""

    def patch(query, key, freqs_ci):
        freqs_pairs = torch.view_as_real(freqs_ci)
        # Mirror ``reshape_for_broadcast``: keep dims 1 (seq) and -1 (d/2), plus the re/im pair.
        shape = [d if i == 1 else 1 for i, d in enumerate(query.shape[:-1])] + [freqs_pairs.shape[-2], 2]
        freqs_pairs = freqs_pairs.view(*shape).to(query.device)
        return _apply_rotary_pos_emb_pairs(query, freqs_pairs), _apply_rotary_pos_emb_pairs(key, freqs_pairs)

    return patch


@register_patch(
    "openvino",
    "transformers.models.phi3.modeling_phi3.Phi3RotaryEmbedding.forward",
    "transformers.models.phimoe.modeling_phimoe.PhimoeRotaryEmbedding.forward",
)
def _patch_longrope_rotary_emb(original):
    """Trace both LongRoPE frequency sets and select with ``torch.where`` on the sequence length.

    Eager LongRoPE picks the long- or short-context ``inv_freq`` with a Python ``if seq_len >
    original_max``, and the ``@dynamic_rope_update`` wrapper installs it by mutating a buffer. On a
    dynamic ``position_ids`` both are data-dependent (``GuardOnDataDependentSymNode``). This
    buffer-free forward computes ``inv_freq``/``mscale`` from both factors and selects with
    ``torch.where``, so the exported graph carries both paths and picks at runtime.
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
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * mscale
            sin = emb.sin() * mscale
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    return patch


@register_patch("openvino", "torch.nn.functional.avg_pool2d")
def _patch_avg_pool2d(original):
    """Clamp oversize pooling kernels to the input's spatial size.

    torch's ``ceil_mode`` pooling permits a kernel larger than the input, producing a 1×1
    output — EfficientNet's pooler (``AvgPool2d(config.hidden_dim)``) relies on it. OV's
    ``AvgPool`` rejects kernels larger than the padded input.
    """

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
    """Decompose ``Tensor.unfold(dim, size, step)`` into ``index_select`` + reshape.

    OV's ``aten.unfold`` translator builds a permutation one rank too long for 3D inputs
    (PatchTST's patchification), producing an invalid Transpose.
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
    """Strip randomness — zeros, shaped like the argument or the requested size.

    Stochastic ops have no place in an exported graph, and inference never samples.
    """

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
    """Decompose the 2-D form of ``embedding_bag`` into a gather and a reduction.

    OV's CPU plugin cannot compile `aten._embedding_bag` at all — the graph converts, then
    `compile_model` raises ``to_shape was called on a dynamic shape``. With one bag per row (a 2-D
    `input` and no `offsets`), the op is just an embedding lookup reduced along the bag axis, which
    OV handles. Anything else — explicit offsets, `include_last_offset`, `padding_idx`, `max_norm`
    — falls back to the original and fails the same way it did before.
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
    """Promote integral inputs to `int64` the way torch does before summing.

    OV's ``CumSum`` keeps the input element type, so a bool mask accumulates *as bool*: the running sum
    saturates at ``True`` and ``cumsum([1, 1, 1, 0, 0])`` comes back ``[1, 1, 1, 1, 1]`` instead of
    ``[1, 2, 3, 3, 3]``. OPT builds its `position_ids` that way, so every token ends up at position 0.
    Narrower integer types have the same overflow exposure, so they are widened too.
    """

    # The axis is passed straight through (positionally, as `dim=`, or as the `axis=` alias longt5
    # uses) — only `dtype` is intercepted.
    def patch(input, *args, dtype=None, **kwargs):
        if dtype is None and input.dtype in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32):
            dtype = torch.int64
        return original(input, *args, dtype=dtype, **kwargs)

    return patch


@register_patch("openvino", "torch.cummax", "torch.Tensor.cummax")
def _patch_cummax(original):
    """OV has no ``aten.cummax`` lowering — reuse the ONNX triangular-mask decomposition."""
    from .exporter_onnx import _patch_cummax_or_cummin

    return _patch_cummax_or_cummin(original, mode="max")


@register_patch("openvino", "torch.cummin", "torch.Tensor.cummin")
def _patch_cummin(original):
    """OV has no ``aten.cummin`` lowering — reuse the ONNX triangular-mask decomposition."""
    from .exporter_onnx import _patch_cummax_or_cummin

    return _patch_cummax_or_cummin(original, mode="min")


@register_patch("openvino", "torch.bincount", "torch.Tensor.bincount")
def _patch_bincount(original):
    """Replace ``torch.bincount`` with ``zeros + scatter_add_`` of size ``minlength`` (or input max+1
    when unknown).

    OV's PyTorch frontend has no ``aten.bincount`` lowering — same shape of fix as
    ``_patch_histc``. The static output shape ``minlength`` keeps shape inference happy.
    """

    from torch.fx.experimental.symbolic_shapes import guard_or_true

    def patch(input, weights=None, minlength=0):
        flat = input.reshape(-1)
        # ``flat.numel() > 0`` and ``int(flat.max().item())`` on a data-dependent SymInt trip
        # ``GuardOnDataDependentSymNode`` (splinter's question-token binning). ``guard_or_true``
        # optimistically assumes non-empty — an empty ``bincount`` collapses to a zero-length
        # output anyway (harmless) — and ``torch._check_is_size`` marks the ``max`` result as
        # size-like so the downstream ``bins + 1 > 0`` check in AOT autograd doesn't refire.
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
    """Compute ``irfft`` entirely in real arithmetic — split the one-sided spectrum into
    real/imag planes, mirror them to the full conjugate-symmetric spectrum, and contract
    against real cos/sin DFT bases.

    OV's ``DFT`` op rejects ``is_onesided=1``/``inverse=1`` together, and a complex-valued
    decomposition (mirror + ``ifft``) routes complex tensors through OV's builtin ``cat`` /
    ``permute`` / ``bmm`` translators, which mix OV's native ``ComplexTypeMark`` representation
    with the ``[..., 2]`` real-pair one our ``aten.complex`` extension emits (same clash as
    ``_patch_apply_rotary_emb``). Keeping the whole transform real confines traced complex ops
    to ``complex``/``view_as_real``, which the extensions handle.
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
    """Replace ``rfft`` with ``fft`` + slice to the one-sided half. OV's ``DFT(is_onesided=1)``
    has no inverse-pair (see ``_patch_irfft``); using two-sided + slice gives the same result
    for the forward direction. Affects audio models (wav2vec*, seamless_m4t, pop2piano)."""

    def patch(input, n=None, dim=-1, norm=None):
        full = torch.fft.fft(input, n=n, dim=dim, norm=norm)
        n_full = full.shape[dim]
        slc = [slice(None)] * full.ndim
        slc[dim] = slice(0, n_full // 2 + 1)
        return full[tuple(slc)]

    return patch


def _dft(input, n, dim, *, inverse):
    """1-D DFT as a twiddle matmul — OV's frontend translates no ``aten._fft_c2c``. Quadratic, but
    adequate for the audio-encoder-sized transforms that reach this path."""
    if n is None:
        n = input.shape[dim]
    k = torch.arange(n, device=input.device, dtype=torch.float32)
    angles = (2.0 if inverse else -2.0) * torch.pi * k.view(-1, 1) * k / n
    twiddle = torch.complex(angles.cos(), angles.sin())
    x = input if torch.is_complex(input) else input.to(torch.complex64)
    out = x.movedim(dim, -1) @ twiddle.T
    return (out / n if inverse else out).movedim(-1, dim)


@register_patch("openvino", "torch.fft.fft")
def _patch_fft(original):
    """``torch.fft.fft`` lowers to an ``aten._fft_c2c`` OV's frontend can't translate."""

    def patch(input, n=None, dim=-1, norm=None):
        return _dft(input, n, dim, inverse=False)

    return patch


@register_patch("openvino", "torch.fft.ifft")
def _patch_ifft(original):
    """Inverse of ``_patch_fft`` — conjugate twiddle, divided by ``n``."""

    def patch(input, n=None, dim=-1, norm=None):
        return _dft(input, n, dim, inverse=True)

    return patch


@register_patch("openvino", "torch.fft.fftn")
def _patch_fftn(original):
    """Multi-dim FFT decomposed as successive 1-D ``torch.fft.fft`` calls along each ``dim``.

    OV has no ``aten._fft_c2c`` lowering for N-D inputs; the iterative 1-D form composes with
    our ``_patch_fft`` so each axis is translated cleanly. Affects FNet.
    """

    def patch(input, s=None, dim=None, norm=None):
        dims = list(range(input.ndim)) if dim is None else list(dim)
        sizes = [None] * len(dims) if s is None else list(s)
        out = input
        for d, n in zip(dims, sizes):
            out = torch.fft.fft(out, n=n, dim=d, norm=norm)
        return out

    return patch


@register_patch("openvino", "torch.fft.ifftn")
def _patch_ifftn(original):
    """Multi-dim inverse FFT — same decomposition as ``_patch_fftn`` via ``torch.fft.ifft``."""

    def patch(input, s=None, dim=None, norm=None):
        dims = list(range(input.ndim)) if dim is None else list(dim)
        sizes = [None] * len(dims) if s is None else list(s)
        out = input
        for d, n in zip(dims, sizes):
            out = torch.fft.ifft(out, n=n, dim=d, norm=norm)
        return out

    return patch


@register_patch("openvino", "torch.fft.rfftn")
def _patch_rfftn(original):
    """Real N-D FFT — last dim uses ``rfft`` (one-sided), remaining dims use ``fft``."""

    def patch(input, s=None, dim=None, norm=None):
        dims = list(range(input.ndim)) if dim is None else list(dim)
        sizes = [None] * len(dims) if s is None else list(s)
        out = input
        for d, n in zip(dims[:-1], sizes[:-1]):
            out = torch.fft.fft(out, n=n, dim=d, norm=norm)
        return torch.fft.rfft(out, n=sizes[-1], dim=dims[-1], norm=norm)

    return patch


@register_patch("openvino", "torch.fft.irfftn")
def _patch_irfftn(original):
    """Real N-D inverse FFT — last dim uses ``irfft``, remaining dims use ``ifft``."""

    def patch(input, s=None, dim=None, norm=None):
        dims = list(range(input.ndim)) if dim is None else list(dim)
        sizes = [None] * len(dims) if s is None else list(s)
        out = torch.fft.irfft(input, n=sizes[-1], dim=dims[-1], norm=norm)
        for d, n in zip(dims[:-1], sizes[:-1]):
            out = torch.fft.ifft(out, n=n, dim=d, norm=norm)
        return out.real if torch.is_complex(out) else out

    return patch


@register_patch("openvino", "torch.Tensor.scatter_reduce_", "torch.Tensor.scatter_reduce")
def _patch_scatter_reduce(original):
    """Decompose ``scatter_reduce_(dim, index, src, reduce)`` into ``scatter_*`` variants OV
    can lower. ``sum``/``amax``/``amin`` map to ``scatter_add_``/``scatter_reduce(amax)`` /
    ``scatter_reduce(amin)`` already, but the ``two`` overload OV doesn't recognise has the
    same algorithmic content — replace with the plain ``scatter_add_`` for ``sum`` (the only
    reduce mode actually used in the failing model, BLT).
    """

    def patch(self, dim, index, src, *, reduce="sum", include_self=True):
        if reduce == "sum":
            if not include_self:
                self.zero_()
            return self.scatter_add_(dim, index, src)
        return original(self, dim, index, src, reduce=reduce, include_self=include_self)

    return patch


# ── OpenVINO conversion extensions ──────────────────────────────────────────
# Custom OV-side translations registered in ``_OV_CONVERSION_EXTENSIONS`` and passed to
# ``openvino.convert_model(extension=...)``. Mirrors the role of ONNX's
# ``_ONNX_TRANSLATION_TABLE``: use this when an op has no equivalent torch-level decomposition.
# Each ``_convert_*(context)`` receives a ``NodeContext`` (``context.get_input(i)`` for inputs)
# and returns a list of output ports built with ``openvino.opset14`` ops.
#
# To add a new translation: implement ``_convert_*`` and append a ``ConversionExtension`` to
# ``_OV_CONVERSION_EXTENSIONS``.


def _match_kv_heads(query, tensor):
    """Grouped K/V widened to the query's head count, for a `[tokens, heads, dim]` packed tensor.

    OV's SDPA merges the head axis of all three inputs, so a grouped model reaches it with two head counts
    and is refused outright. Widened here rather than inside the loop body, where the token axis is a slice
    and the arithmetic would be dynamic — out here only the token count is, and the head counts are the
    architecture's.
    """
    query_shape, tensor_shape = query.get_partial_shape(), tensor.get_partial_shape()
    if query_shape.rank.get_length() != 3 or not (query_shape[1].is_static and tensor_shape[1].is_static):
        return tensor
    heads, kv_heads = query_shape[1].get_length(), tensor_shape[1].get_length()
    if heads == kv_heads or kv_heads == 0 or heads % kv_heads:
        return tensor
    if not tensor_shape[2].is_static:
        return tensor
    # Each group's head repeated in place (`repeat_interleave`), which is what the packed layout expects:
    # a size-1 axis beside the heads, tiled, then folded back in.
    widened = ov_ops.unsqueeze(tensor, ov_ops.constant(np.array([2], dtype=np.int64)))
    tiled = ov_ops.tile(widened, ov_ops.constant(np.array([1, 1, heads // kv_heads, 1], dtype=np.int64)))
    target = np.array([0, heads, tensor_shape[2].get_length()], dtype=np.int64)
    return ov_ops.reshape(tiled, ov_ops.constant(target), special_zero=True).output(0)


def _convert_varlen_attn(context):
    """Convert ``torch_attn::_varlen_attn`` — packed variable-length attention — into an OV ``Loop``.

    The chunked vision/audio export patch packs several sequences into one flat `[L, heads, dim]` tensor and
    marks their boundaries with `cu_seqlens`, so a token attends only within its own segment. OV has no rule
    for the op, and lowering it at the torch level would force every backend to take the same expansion;
    converting it here keeps the traced graph the packed one it was.

    One iteration per segment, each a dense SDPA over that segment's `n_i` tokens, written into its rows of a
    loop-carried output — `sum(n_i^2)` work and no `L x L` mask, which is the shape the flash kernel this op
    names would do. The segment bounds come from `cu_seqlens` inside the body, so the slices vary per
    iteration; that is why the output is carried (`set_merged_input`) rather than a scan output, whose
    slices must all be the same size.
    """
    query, key, value = (context.get_input(index) for index in range(3))
    key, value = _match_kv_heads(query, key), _match_kv_heads(query, value)
    cu_seqlens = ov_ops.convert(context.get_input(3), "i64")
    element_type = query.get_element_type()
    axis0 = ov_ops.constant(np.array([0], dtype=np.int64))
    one = ov_ops.constant(np.array([1], dtype=np.int64))

    # Laid out the way OV's SDPA wants once, out here, rather than per segment inside the body: the op is
    # batch-first, and the trace hands over `[tokens, heads, dim]`.
    heads_first = ov_ops.constant(np.array([1, 0, 2], dtype=np.int64))
    axis2 = ov_ops.constant(np.array([2], dtype=np.int64))
    query, key, value = (
        ov_ops.unsqueeze(ov_ops.transpose(tensor, heads_first), axis0).output(0) for tensor in (query, key, value)
    )

    # Body: (iteration, condition, q, k, v, cu, carried output) -> (written output, condition)
    iteration = ov_ops.parameter([], Type.i64)
    condition = ov_ops.parameter([], Type.boolean)
    body_q, body_k, body_v = (ov_ops.parameter(PartialShape([-1, -1, -1, -1]), element_type) for _ in range(3))
    body_cu = ov_ops.parameter(PartialShape([-1]), Type.i64)
    carried = ov_ops.parameter(PartialShape([-1, -1, -1, -1]), element_type)

    index = ov_ops.reshape(iteration, one, False)
    start = ov_ops.gather(body_cu, index, axis0)
    stop = ov_ops.gather(body_cu, ov_ops.add(index, one), axis0)

    def _segment(tensor):
        """This iteration's tokens, still `[1, heads, n_i, dim]`."""
        return ov_ops.slice(tensor, start, stop, one, axis2)

    segment = ov_ops.scaled_dot_product_attention(_segment(body_q), _segment(body_k), _segment(body_v), causal=False)
    # Written into its own tokens of a whole-length buffer, not appended to a growing one: a loop-carried
    # value keeps one shape for every iteration, so accumulating by concatenation silently carries only the
    # last segment — which is why a single-segment call (one image, one clip) always looked right.
    rows = ov_ops.range(
        ov_ops.squeeze(start, axis0), ov_ops.squeeze(stop, axis0), ov_ops.constant(np.int64(1)), output_type="i64"
    )
    grown = ov_ops.scatter_update(carried, rows, segment, ov_ops.constant(np.int64(2)))
    body = Model(
        [ov_ops.result(grown), ov_ops.result(condition)],
        [iteration, condition, body_q, body_k, body_v, body_cu, carried],
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
    # The buffer every iteration writes into: the query's own shape, so the tokens land where they came from.
    blank = ov_ops.broadcast(ov_ops.constant(0.0, dtype=element_type), ov_ops.shape_of(query, output_type="i64"))
    loop.set_merged_input(carried, blank.output(0), grown.output(0))
    loop.validate_and_infer_types()
    # Back to the packed `[tokens, heads, dim]` the graph around it speaks.
    written = ov_ops.squeeze(loop.get_iter_value(grown.output(0), -1), axis0)
    return [ov_ops.transpose(written, heads_first).output(0)]


def _convert_grouped_mm(context):
    """Convert ``aten._grouped_mm`` / ``transformers.grouped_mm_fallback`` to OV ops.

    ``grouped_mm(mat_a: (M, K), mat_b: (G, K, N), offs: (G,)) -> (M, N)`` computes
    ``out[offs[g-1]:offs[g]] = mat_a[offs[g-1]:offs[g]] @ mat_b[g]`` per expert ``g``.
    ``G`` (number of experts) must be static at translation time, so we unroll the loop and
    emit ``G`` independent ``Slice + Gather + MatMul`` triples followed by a final ``Concat``.
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
    """Convert ``aten.empty_permuted`` to a zero-initialised constant of the requested shape.

    ``empty_permuted`` is uninitialised — only the shape matters for downstream ops. OV has no
    direct equivalent; emit a zero ``Broadcast`` of the right shape and dtype.
    """
    size = context.get_input(0)
    # Default to f32; in the MoE expert path the result feeds straight into integer index ops or
    # gets overwritten before any read, so dtype doesn't propagate to outputs.
    zero = ov_ops.constant(np.float32(0.0))
    return [ov_ops.broadcast(zero, size).output(0)]


def _convert_index_add(context):
    """Convert ``aten.index_add(self, dim, index, source, alpha=1)`` — OV's default translator
    expects 5 inputs and fails when ``alpha`` is defaulted (torch omits it from the FX call).
    Emit ``ScatterElementsUpdate`` with ``sum`` reduction: expand ``index`` from a 1-D shape
    ``(N,)`` to match ``source`` along all axes so per-position add works. Used by t5gemma /
    t5gemma2 / speecht5 relative-attention-bias accumulation."""
    data = context.get_input(0)
    dim = int(context.get_values_from_const_input(1))
    index = context.get_input(2)
    source = context.get_input(3)
    # ``index_add`` is ``self[index] += alpha * source``; fold a non-default ``alpha`` (FX input 4)
    # into ``source`` before the scatter-add.
    if context.get_input_size() > 4 and context.get_input(4).get_node().get_type_name() == "Constant":
        alpha = context.get_values_from_const_input(4)
        if alpha != 1:
            source = ov_ops.multiply(
                source, ov_ops.convert(ov_ops.constant(np.array(alpha)), source.get_element_type())
            )
    # Broadcast 1-D index to source's rank/shape along ``dim`` so ScatterElementsUpdate
    # can consume element-wise ``source`` values.
    src_shape = ov_ops.shape_of(source, output_type="i64")
    # Reshape ``index`` to a shape that's ``1`` in every dim except ``dim`` — broadcast handles
    # the rest. Then broadcast to source's shape explicitly to feed ScatterElementsUpdate.
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
    """``view_as_real(complex)`` reinterprets a complex tensor as ``[..., 2]`` real. Our
    ``_convert_complex`` already represents complex tensors that way, so this is identity."""
    return [context.get_input(0)]


def _convert_fft_c2c(context):
    """Convert ``aten._fft_c2c(self, dim, normalization, forward)`` to OV's ``DFT``/``IDFT``.

    OV's ``dft``/``idft`` expect a trailing ``[..., 2]`` real/imag pair. Our ``_convert_complex``
    produces that layout already. For models that call ``_fft_c2c`` on a real-valued tensor
    (FNet, where ``torch.fft.fftn(real)`` implicitly promotes to complex), we stack a zero
    imaginary component on the last dim first. We detect the input rank via partial shape and
    only inject the stack when there's no trailing ``[..., 2]`` already.
    """
    data = context.get_input(0)
    axes = context.get_input(1)
    forward = bool(context.get_values_from_const_input(3))
    # If the input doesn't already end in a 2-element axis, treat it as real and pad imag=0.
    pshape = data.get_partial_shape()
    needs_pair = pshape.rank.is_static and (
        not pshape[pshape.rank.get_length() - 1].is_static or pshape[pshape.rank.get_length() - 1].get_length() != 2
    )
    if needs_pair:
        zeros = ov_ops.broadcast(ov_ops.constant(np.float32(0.0)), ov_ops.shape_of(data))
        data = ov_ops.concat(
            [ov_ops.unsqueeze(data, ov_ops.constant(-1)), ov_ops.unsqueeze(zeros, ov_ops.constant(-1))],
            axis=-1,
        )
    op = ov_ops.dft if forward else ov_ops.idft
    return [op(data, ov_ops.convert(axes, "i64")).output(0)]


def _convert_conj(context):
    """Convert ``aten._conj(complex)`` — complex conjugate. With our ``[..., 2]`` real/imag
    representation, this negates the imaginary part. We split into real/imag, negate imag,
    and concat back. Used by manual FFT decompositions."""
    data = context.get_input(0)
    # last dim is 2 — split along axis -1 into real/imag, then concat [real, -imag]
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
    """Convert ``aten.bitwise_not`` — OV's default translator internally calls ``torch.sym_float``
    on the input's dynamic dims to compute output shape metadata, and that Python-level call
    remains as an unconverted node in the resulting graph. Emit ``LogicalNot`` on a boolean
    view of the input; ``bitwise_not`` on bool would reject with ``is_integral()`` check.
    Affects deformable_detr, mask2former."""
    data = context.get_input(0)
    return [ov_ops.logical_not(ov_ops.convert(data, "boolean")).output(0)]


def _convert_layer_norm(context):
    """Convert ``aten.layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable)`` to
    ``MVN + (weight * x + bias)``. OV's default translator decomposes to ``native_layer_norm``
    which returns a 3-tuple ``(out, mean, rstd)``; the unused ``mean`` / ``rstd`` outputs are
    emitted as ``torch::None`` constants that fail conversion (chameleon). Emitting MVN
    directly gives a single-output op with no dangling None."""
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
    """Convert ``aten._to_copy(self, dtype=..., ...)`` to an OV ``Convert``.

    OV's default translator throws (``Attribute dtype can't be converted to defined types``)
    when the target dtype is ``complex64`` — no native OV complex type. Our ``_convert_complex``
    uses a ``[..., 2]`` real representation, so the complex cast is a no-op we swallow. For all
    real dtypes we emit a real ``Convert`` — dropping the cast entirely regresses downstream
    ops like ``aten.bitwise_and.Tensor`` that need the mask to actually be ``bool`` (cpmant,
    chameleon)."""
    data = context.get_input(0)
    if not context.has_attribute("dtype"):
        return [data]
    try:
        dtype = context.get_attribute("dtype")
    except Exception:
        # Complex dtypes throw ``Attribute dtype can't be converted to defined types``. With
        # the ``[..., 2]`` real representation, the cast is a no-op.
        return [data]
    if dtype is None:
        return [data]
    return [ov_ops.convert(data, dtype).output(0)]


def _convert_bmm(context):
    """Translate ``aten.bmm``, shielding softmax-fed ones from OV's SDPA fusion.

    The frontend-normalization fusion matches ``bmm -> softmax -> bmm`` and mis-shapes the
    result when batch and heads are flattened into one dim (SpeechT5/MVP/SeamlessM4T's
    relative-position eager attention): the fused op emits ``[b, b*h, q, k]`` instead of
    ``[b, h, q, k]``. For a bmm consuming a ``Softmax`` output, a ``Reshape(x, ShapeOf(x))``
    no-op is appended — runtime-dependent, so normalization can't fold it away before the
    fusion pass runs, and MOC's nop-elimination cleans it up afterwards. Every other bmm
    translates to a plain ``MatMul``.
    """
    a, b = context.get_input(0), context.get_input(1)
    product = ov_ops.matmul(a, b, transpose_a=False, transpose_b=False)
    if a.get_node().get_type_name() != "Softmax":
        return [product.output(0)]
    identity = ov_ops.reshape(product, ov_ops.shape_of(product, output_type="i64"), special_zero=False)
    return [identity.output(0)]


def _convert_sdpa(context):
    """Convert ``aten.scaled_dot_product_attention`` — wrapping OV's op with a mask-dtype fix.

    OV's ``opset13::ScaledDotProductAttention`` rejects int-typed masks. Under CUDA export
    ``aten.expand`` promotes bool masks to ``i64`` during OV translation, so we insert a
    ``Convert(→ boolean)`` on the mask input before instantiating the op.

    The aten node's scalar ``scale`` is not surfaced as a frontend input here, so OV always applies its
    own ``head_dim**-0.5`` default; a non-default scale is instead folded into the query up in
    ``_patch_sdpa``. Q/K/V pass through unchanged."""
    q, k, v = context.get_input(0), context.get_input(1), context.get_input(2)
    # A ``None`` FX arg reaches the extension as an unconverted ``PtFrameworkNode``.
    mask = None
    if context.get_input_size() > 3:
        candidate = context.get_input(3)
        if candidate.get_node().get_type_name() != "PtFrameworkNode":
            # A bool mask that `aten.expand` promoted to ``i64`` under CUDA export is cast back to
            # boolean; a float *additive* mask (``0`` attend / ``-inf`` masked) passes through unchanged
            # — OV's SDPA adds it to the scores, whereas casting it to bool would invert/destroy it.
            mask = candidate if candidate.get_element_type().is_real() else ov_ops.convert(candidate, "boolean")
    is_causal = False
    if context.get_input_size() > 5:
        # ``is_causal`` is a positional FX input (arg 5), not a node attribute.
        if context.get_input(5).get_node().get_type_name() == "Constant":
            is_causal = bool(context.get_values_from_const_input(5))
    kwargs = {"causal": is_causal}
    if mask is not None:
        kwargs["attention_mask"] = mask
    # OV's own default, stated rather than left implicit: the CPU plugin folds attention and its KV cache
    # into one op only for the five-input form, and a four-input SDPA reads its cache through a Concat it
    # then has to materialise — the whole cache, on every decode step. The value is what the op would apply
    # anyway (`_patch_sdpa` has already folded any non-default scale into the query).
    head_dim = q.get_partial_shape()[-1]
    if head_dim.is_static:
        # In the query's own type: OV's SDPA merges the element types of all its inputs, so an `f32` scale
        # beside `bf16` queries is a conversion failure rather than a promotion.
        scale = np.array(head_dim.get_length() ** -0.5).astype(q.get_element_type().to_dtype())
        kwargs["scale"] = ov_ops.constant(scale, q.get_element_type())
    return [ov_ops.scaled_dot_product_attention(q, k, v, **kwargs).output(0)]


def _convert_complex(context):
    """Convert ``aten.complex(real, imag)`` by stacking as the last dim — OV represents complex
    tensors as ``[..., 2]`` real tensors via ``ComplexTypeMark``. Affects models that build
    complex tensors explicitly (RoPE polar form, manual FFT decompositions)."""
    real = context.get_input(0)
    imag = context.get_input(1)
    stacked = ov_ops.concat(
        [ov_ops.unsqueeze(real, ov_ops.constant(-1)), ov_ops.unsqueeze(imag, ov_ops.constant(-1))],
        axis=-1,
    )
    return [stacked.output(0)]


# ── SymInt builtin translations ─────────────────────────────────────────────
# torch.export records Python-level math on SymInts (``a % b``, ``a // b``, ``min(a, b)``)
# as ``call_function`` nodes whose target is the Python builtin or ``torch.sym_*`` callable.
# These survive into the EP because torch never lowers them — there's no aten op that
# produces a SymInt for ``mod``/``floordiv``/etc. OV's PyTorch frontend has no translation
# for them either, so we register one per builtin keyed on its ``str(target)`` literal.
# Each translator emits an OV opset17 elementwise op; the result is a 0-d integer tensor
# that downstream shape ops (view, reshape, expand) concat into shape lists natively.


def _convert_sym_binop(op):
    """Factory: build a 2-arg OV-op translator for SymInt binary builtins (add, mul, mod, …).

    Mixed int/float operands (e.g. ``symint - 0.5`` in deformable-attention grid math) are
    promoted to the float side — OV element-wise ops require matching types. ``mod`` must map
    to ``floor_mod``: Python's ``%`` is floored (``-7 % 3 == 2``) while OV's ``Mod`` truncates,
    which breaks ``-seq % block``-style padding arithmetic (LongT5).
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
    """Factory: build a 1-arg OV-op translator for SymInt unary builtins (floor, ceil, sym_float).

    ``cast_to_i64`` casts the output back to ``i64`` — Python's ``floor(x)`` / ``ceil(x)`` on a
    SymFloat return an int, but OV's ``floor`` / ``ceiling`` are dtype-preserving, so a float
    input yields a float output. Downstream shape ops (SequenceMark → Concat) need i64;
    without the cast, mixed-dtype Concat fails ``element::Type::merge`` (focalnet)."""

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
    """``a // b`` over SymInts → ``floor(a / b)``, cast to i64. Used by patch/window-size
    computations (focalnet, donut_swin). The i64 cast keeps the result shape-op-friendly —
    downstream ``SequenceMark → Concat`` requires a uniform int dtype.

    Truncating division breaks the ceil-div idiom ``-(-x // n)`` on a symbolic ``x`` — minimax_m3_vl's
    ``num_key_blocks`` comes out one too small and sends a ``scatter`` out of bounds."""
    a, b = _float_operands(context)
    return [ov_ops.convert(ov_ops.floor(ov_ops.divide(a, b)), "i64").output(0)]


def _convert_sym_truediv(context):
    """``a / b`` over SymInts → **float** division, matching Python's ``truediv``.

    Python's ``/`` always returns a float. granite_speech computes its merged batch dim as
    ``batch * ceil(seq / chunk)``; truncated, the ``ceil`` sees ``0`` and the reshape gets a zero
    batch dim."""
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
            ConversionExtension("aten._fft_c2c.default", _convert_fft_c2c),
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
            # ``torch.sym_float`` has an address-based ``str()`` (not a stable ``<built-in ...>``
            # form), so we register by its runtime str. Emits a real→f32 Convert.
            ConversionExtension(str(torch.sym_float), _convert_sym_unop(lambda x: ov_ops.convert(x, "f32"))),
        ]
    )
