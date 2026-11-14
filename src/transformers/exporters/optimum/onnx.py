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
"""Loading optimum-onnx's ONNX graphs: a causal LM taking and returning its cache (`model.onnx`, or a merged decoder), an
encoder-decoder (an encoder and a merged decoder) and an encoder, the cache renamed as the runtime names it. A decoder
split in two graphs, the prompt's and the steps', each holding the weights, is refused."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Unpack

from ...utils.hub import DownloadKwargs, cached_file
from ...utils.import_utils import is_torch_available
from ..metadata import ExportMetadata
from .common import _architecture, _find, _repo_files


if is_torch_available():
    import torch
    from torch.utils._pytree import tree_structure, treespec_dumps

if TYPE_CHECKING:
    import onnx_ir

    from ..base import ExportedComponent


# The graph files read, in order: one graph taking and returning the cache, or a merged decoder (which also takes a
# `use_cache_branch` flag), at the root or in `onnx/` (transformers.js' layout).
ONNX_FILES = tuple(
    f"{folder}{name}" for name in ("model.onnx", "decoder_model_merged.onnx") for folder in ("", "onnx/")
)
# An encoder-decoder's encoder, beside its merged decoder
ONNX_ENCODER_FILES = ("encoder_model.onnx", "onnx/encoder_model.onnx")
# A decoder split in two graphs, the prompt's and the steps', each holding the weights: refused.
ONNX_SPLIT_FILES = ("decoder_with_past_model.onnx", "onnx/decoder_with_past_model.onnx")

# A cache input or output, and for an encoder-decoder's, the half it belongs to; named as the runtime's cache flattens
_ONNX_CACHE = re.compile(r"(past_key_values|present)\.(\d+)(?:\.(decoder|encoder))?\.(key|value)")
_ONNX_CACHE_HALVES = {None: "cache.layers", "decoder": "cache.self_attention_cache.layers"}
_ONNX_CACHE_HALVES["encoder"] = "cache.cross_attention_cache.layers"
# An encoder-decoder's decoder inputs, under the names the runtime's decode graph takes
_ONNX_DECODER_INPUTS = {
    "input_ids": "decoder_input_ids",
    "encoder_hidden_states": "encoder_outputs.last_hidden_state",
    "encoder_attention_mask": "attention_mask",
}


def _read_onnx(pretrained_model_name_or_path, file: str, **kwargs: Unpack[DownloadKwargs]) -> onnx_ir.Model:
    """The graph, its weights fetched beside it (optimum writes them to `<file>_data`, split over `<file>_data_<n>`
    past 2GB) and read into memory: the graph is renamed, so ONNX Runtime opens it serialized rather than from this
    file."""
    import onnx_ir

    path = Path(cached_file(pretrained_model_name_or_path, file, **kwargs))
    data_prefix = f"{Path(file).name}_data"
    siblings = [
        name
        for name in _repo_files(pretrained_model_name_or_path, **kwargs)
        if Path(name).parent == Path(file).parent and Path(name).name.startswith(data_prefix)
    ]
    for sibling in siblings:
        cached_file(pretrained_model_name_or_path, sibling, **kwargs)
    program = onnx_ir.load(str(path))
    # The Hub cache links the weight files to blobs outside the snapshot, which `onnx_ir` only reads once resolved.
    for graph in program.graphs():
        for value in graph.initializers.values():
            tensor = value.const_value
            if isinstance(tensor, onnx_ir.ExternalTensor):
                resolved = onnx_ir.ExternalTensor(
                    Path(tensor.path).resolve(),
                    tensor.offset,
                    tensor.length,
                    tensor.dtype,
                    shape=tensor.shape,
                    name=tensor.name,
                )
                value.const_value = onnx_ir.Tensor(resolved.numpy(), name=tensor.name)
    return program


def _onnx_traced(value: onnx_ir.Value) -> dict[str, Any]:
    # A dynamic axis as the length of 2 a decode capture records.
    dims = [dim if isinstance(dim, int) else 2 for dim in value.shape]
    return {"rank": len(dims), "shape": dims, "dtype": str(value.dtype.numpy())}


def _rename_onnx_cache(program: onnx_ir.Model) -> dict[int, tuple[int, int, int]]:
    """Name the cache inputs and outputs as an export of the runtime does (`input.cache.layers.<i>.keys`, an
    encoder-decoder's `input.cache.self_attention_cache.layers.<i>.keys` and `cross_attention_cache`), its outputs in
    the order the runtime's cache flattens (an encoder-decoder's self-attention layers before its cross-attention
    ones), and return each self-attention layer's `(num_kv_heads, key_head_dim, value_head_dim)`."""
    graph = program.graph
    geometry: dict[int, list[int]] = {}
    order = {}
    for values, prefix in ((graph.inputs, "input"), (graph.outputs, "output")):
        for value in values:
            if (match := _ONNX_CACHE.fullmatch(value.name)) is None:
                continue
            _, index, half, kind = match.groups()
            index = int(index)
            value.name = f"{prefix}.{_ONNX_CACHE_HALVES[half]}.{index}.{kind}s"
            order[value.name] = (half == "encoder", index, kind == "value")
            if prefix == "input" and half != "encoder":
                heads, dim = value.shape[1], value.shape[3]
                geometry.setdefault(index, [heads, dim, dim])[1 if kind == "key" else 2] = dim
    outputs = list(graph.outputs)
    graph.outputs.clear()
    graph.outputs.extend(sorted(outputs, key=lambda value: (value.name in order, order.get(value.name, ()))))
    return {index: tuple(shape) for index, shape in sorted(geometry.items())}


def _drop_use_cache_branch(program: onnx_ir.Model) -> None:
    """A merged decoder picks its branch from a `use_cache_branch` input: compute it in the graph instead, from whether
    the cache holds anything."""
    import onnx_ir

    graph = program.graph
    flag = next((value for value in graph.inputs if value.name == "use_cache_branch"), None)
    if flag is None:
        return
    past = next(value for value in graph.inputs if value.name.endswith("layers.0.keys") and "cross" not in value.name)
    # Shape + Gather rather than `Shape(start, end)`, which older opsets lack.
    shape = onnx_ir.node("Shape", [past])
    axis = onnx_ir.node("Constant", [], attributes={"value": onnx_ir.tensor(torch.tensor([2], dtype=torch.int64))})
    length = onnx_ir.node("Gather", [shape.outputs[0], axis.outputs[0]])
    zero = onnx_ir.node("Constant", [], attributes={"value": onnx_ir.tensor(torch.zeros(1, dtype=torch.int64))})
    has_past = onnx_ir.node("Greater", [length.outputs[0], zero.outputs[0]])
    graph.insert_before(graph.node(0), [shape, axis, length, zero, has_past])
    onnx_ir.convenience.replace_all_uses_with(flag, has_past.outputs[0])
    graph.inputs.remove(flag)


def _cache_geometry(geometry: dict[int, tuple[int, int, int]]) -> dict[str, dict]:
    return {"kv": {str(index): list(shape) for index, shape in geometry.items()}, "indexers": {}, "linear_states": {}}


def _load_onnx_seq2seq(
    pretrained_model_name_or_path, config, encoder_file: str, **kwargs
) -> tuple[dict[str, ExportedComponent], bool]:
    """An encoder-decoder: its encoder, and its merged decoder, which computes the cross-attention cache on the first
    step (the runtime's cache starts with it empty) and takes it back after."""
    from ...cache_utils import DynamicCache, EncoderDecoderCache
    from ...modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
    from ..base import ExportedComponent
    from ..cache import materialize_cache_layers
    from ..exporter_dynamo import register_pytree_node

    decoder_file = encoder_file.replace("encoder_model.onnx", "decoder_model_merged.onnx")
    if _find(pretrained_model_name_or_path, (decoder_file,), **kwargs) is None:
        raise ValueError(
            f"{pretrained_model_name_or_path} is an optimum-onnx encoder-decoder whose decoder is split in two graphs, "
            "the prompt's and the steps', each holding the weights, without the merged one: re-export it with "
            "`OnnxExporter().export_for_generation`, whose one decode graph serves both."
        )
    hidden = BaseModelOutput(last_hidden_state=torch.zeros(1))
    encoder = _read_onnx(pretrained_model_name_or_path, encoder_file, **kwargs)
    encoder_inputs = [value.name for value in encoder.graph.inputs]
    encoder_metadata = ExportMetadata(
        architecture=_architecture(config),
        export_dtype="float32",
        export_device="cpu",
        kwargs={value.name: _onnx_traced(value) for value in encoder.graph.inputs},
        in_spec=treespec_dumps(tree_structure(((), dict.fromkeys(encoder_inputs, 0)))),
        out_spec=treespec_dumps(tree_structure(hidden)),
        export_config={"export_format": "onnx", "dynamic": True},
    )

    decoder = _read_onnx(pretrained_model_name_or_path, decoder_file, **kwargs)
    geometry = _rename_onnx_cache(decoder)
    for value in decoder.graph.inputs:
        value.name = _ONNX_DECODER_INPUTS.get(value.name, value.name)
    _drop_use_cache_branch(decoder)
    past = next(value for value in decoder.graph.inputs if value.name.startswith("input.cache."))
    dtype = getattr(torch, str(past.dtype.numpy()))
    # The cache the runtime feeds, as it materializes it (the cross-attention half empty until the first step fills
    # it): its pytree spec is the graph's.
    cache = EncoderDecoderCache(DynamicCache(config=config), DynamicCache(config=config))
    materialize_cache_layers(cache, 1, config, dtype, torch.device("cpu"), kv_geometry=geometry)
    for cache_class in (EncoderDecoderCache, DynamicCache):
        register_pytree_node(cache_class)
    traced = {
        name: _onnx_traced(value)
        for value in decoder.graph.inputs
        if (name := value.name) in ("attention_mask", "decoder_input_ids")
    }
    traced["encoder_outputs"] = {"container": "BaseModelOutput"}
    traced["cache"] = {"container": "EncoderDecoderCache"}
    decoder_inputs = {"attention_mask": 0, "decoder_input_ids": 0, "encoder_outputs": hidden, "cache": cache}
    decoder_metadata = ExportMetadata(
        architecture=_architecture(config),
        export_dtype=str(dtype).removeprefix("torch."),
        export_device="cpu",
        kwargs=traced,
        in_spec=treespec_dumps(tree_structure(((), decoder_inputs))),
        out_spec=treespec_dumps(tree_structure(Seq2SeqLMOutput(logits=torch.zeros(1), cache=cache))),
        export_config={"export_format": "onnx", "dynamic": True},
        cache_geometry=_cache_geometry(geometry),
    )
    return {
        "encoder_model": ExportedComponent(encoder, encoder_metadata),
        "decoder_model": ExportedComponent(decoder, decoder_metadata),
    }, True


def _load_onnx(pretrained_model_name_or_path, config, **kwargs) -> tuple[dict[str, ExportedComponent], bool]:
    """A causal LM taking and returning its cache (inputs `input_ids`, `attention_mask`, maybe `position_ids`,
    `past_key_values.*`; outputs `logits`, `present.*`), an encoder-decoder, or an encoder."""
    from ...cache_utils import DynamicCache
    from ...modeling_outputs import CausalLMOutputWithPast
    from ..base import ExportedComponent
    from ..cache import materialize_cache_layers
    from ..exporter_dynamo import register_pytree_node

    if (encoder_file := _find(pretrained_model_name_or_path, ONNX_ENCODER_FILES, **kwargs)) is not None:
        return _load_onnx_seq2seq(pretrained_model_name_or_path, config, encoder_file, **kwargs)
    file = _find(pretrained_model_name_or_path, ONNX_FILES, **kwargs)
    if file is None:
        raise ValueError(
            f"{pretrained_model_name_or_path} is an optimum-onnx decoder split in two graphs, the prompt's and the "
            "steps', each holding the weights, without the merged one: re-export it with "
            "`OnnxExporter().export_for_generation`, whose one graph serves both."
        )
    program = _read_onnx(pretrained_model_name_or_path, file, **kwargs)
    geometry = _rename_onnx_cache(program)
    _drop_use_cache_branch(program)
    graph = program.graph
    inputs = {value.name: value for value in graph.inputs if not value.name.startswith("input.cache.")}
    export_config = {"export_format": "onnx", "dynamic": True}

    if not geometry:
        outputs = dict.fromkeys((value.name for value in graph.outputs), torch.zeros(1))
        metadata = ExportMetadata(
            architecture=_architecture(config),
            export_dtype="float32",
            export_device="cpu",
            kwargs={name: _onnx_traced(value) for name, value in inputs.items()},
            in_spec=treespec_dumps(tree_structure(((), dict.fromkeys(inputs, 0)))),
            out_spec=treespec_dumps(tree_structure(outputs)),
            export_config=export_config,
        )
        return {"model": ExportedComponent(program, metadata)}, False

    if graph.outputs[0].name != "logits":
        raise ValueError(
            f"{pretrained_model_name_or_path} takes a cache but doesn't return logits first (outputs "
            f"{[value.name for value in graph.outputs]}): only a causal LM loads from optimum-onnx. Re-export it with "
            "`OnnxExporter`."
        )
    past = next(value for value in graph.inputs if value.name == "input.cache.layers.0.keys")
    dtype = getattr(torch, str(past.dtype.numpy()))
    # The cache the runtime feeds, as it materializes it: its pytree spec is the graph's.
    cache = DynamicCache(config=config)
    materialize_cache_layers(cache, 1, config, dtype, torch.device("cpu"), kv_geometry=geometry)
    register_pytree_node(type(cache))
    traced = {name: _onnx_traced(value) for name, value in inputs.items()}
    traced["cache"] = {"container": type(cache).__name__}
    metadata = ExportMetadata(
        architecture=_architecture(config),
        export_dtype=str(dtype).removeprefix("torch."),
        export_device="cpu",
        kwargs=traced,
        in_spec=treespec_dumps(tree_structure(((), {**dict.fromkeys(inputs, 0), "cache": cache}))),
        out_spec=treespec_dumps(tree_structure(CausalLMOutputWithPast(logits=torch.zeros(1), cache=cache))),
        export_config=export_config,
        cache_geometry=_cache_geometry(geometry),
    )
    return {"model": ExportedComponent(program, metadata)}, True


class _OnnxGenerationMixin:
    """optimum-onnx's merged decoder returns placeholders for the cross-attention cache once it has filled it: the
    cache it was given is kept over them."""

    def _call(self, **kwargs) -> Any:
        outputs = super()._call(**kwargs)
        given, cache = kwargs.get("cache"), getattr(outputs, "cache", None)
        if hasattr(given, "cross_attention_cache") and hasattr(cache, "cross_attention_cache"):
            for layer, given_layer in zip(cache.cross_attention_cache.layers, given.cross_attention_cache.layers):
                if layer.keys.shape[0] == 0:
                    layer.keys, layer.values = given_layer.keys, given_layer.values
        return outputs
