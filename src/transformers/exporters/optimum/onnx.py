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
encoder-decoder (an encoder and a merged decoder) and an encoder, the cache renamed as the runtime names it; and the
multimodal exports on onnx-community: text embeddings, an encoder per modality and a merged
decoder on the embeddings. A decoder split in two graphs, the prompt's and the steps', each holding the weights, is
refused."""

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
# `use_cache_branch` flag), at the root or in `onnx/` (onnx-community's layout).
ONNX_FILES = tuple(
    f"{folder}{name}" for name in ("model.onnx", "decoder_model_merged.onnx") for folder in ("", "onnx/")
)
# An encoder-decoder's encoder, beside its merged decoder
ONNX_ENCODER_FILES = ("encoder_model.onnx", "onnx/encoder_model.onnx")
# A multimodal export's graphs beside its merged decoder (onnx-community's): its text embeddings, and one encoder per
# modality, by the component it serves
ONNX_TEXT_EMBEDDINGS = "embed_tokens.onnx"
_ONNX_MODALITY_ENCODERS = {
    "vision_encoder.onnx": "vision_embeddings_model",
    "audio_encoder.onnx": "audio_embeddings_model",
}
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
    # A decoder exported with its attentions too (onnx-community's Whisper, for timestamps) returns only its logits and
    # cache here
    if order and any(value.name == "logits" for value in outputs):
        outputs = [value for value in outputs if value.name == "logits" or value.name in order]
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


def _full_attention_cache(config) -> None:
    """Make the cache `generate` builds from `config` match the graph's, which keeps every layer's keys and values in
    full (its sliding-window layers mask the window in the graph): sliding and chunked layers become full ones. The
    runtime runs no modeling code, so the config shapes nothing else."""
    text_config = config.get_text_config(decoder=True)
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is not None:
        text_config.layer_types = [
            "full_attention" if layer_type in ("sliding_attention", "chunked_attention") else layer_type
            for layer_type in layer_types
        ]
    for attribute in ("sliding_window", "attention_chunk_size"):
        if getattr(text_config, attribute, None) is not None:
            setattr(text_config, attribute, None)


def _onnx_decoder_component(program: onnx_ir.Model, config, geometry: dict) -> ExportedComponent:
    """A decoder taking and returning its cache (renamed by `_rename_onnx_cache`), as a component: its other inputs as
    the graph names them (`input_ids` or `inputs_embeds`, `attention_mask`, maybe `position_ids`)."""
    from ...cache_utils import DynamicCache
    from ...modeling_outputs import CausalLMOutputWithPast
    from ..base import ExportedComponent
    from ..cache import materialize_cache_layers
    from ..exporter_dynamo import register_pytree_node

    graph = program.graph
    inputs = {value.name: value for value in graph.inputs if not value.name.startswith("input.cache.")}
    past = next(value for value in graph.inputs if value.name == "input.cache.layers.0.keys")
    dtype = getattr(torch, str(past.dtype.numpy()))
    _full_attention_cache(config)
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
        export_config={"export_format": "onnx", "dynamic": True},
        cache_geometry=_cache_geometry(geometry),
    )
    return ExportedComponent(program, metadata)


def _onnx_graph_component(program: onnx_ir.Model, config, by_name: bool = False) -> ExportedComponent:
    """A graph without a cache (an embedder, a modality encoder, an encoder) as a component: its inputs as the graph
    names them, its one output as a tensor, or several (or any, with `by_name`) as a dict by their names."""
    from ..base import ExportedComponent

    graph = program.graph
    inputs = {value.name: value for value in graph.inputs}
    outputs = [value.name for value in graph.outputs]
    returns = torch.zeros(1) if len(outputs) == 1 and not by_name else dict.fromkeys(outputs, torch.zeros(1))
    metadata = ExportMetadata(
        architecture=_architecture(config),
        export_dtype=str(graph.outputs[0].dtype.numpy()),
        export_device="cpu",
        kwargs={name: _onnx_traced(value) for name, value in inputs.items()},
        in_spec=treespec_dumps(tree_structure(((), dict.fromkeys(inputs, 0)))),
        out_spec=treespec_dumps(tree_structure(returns)),
        export_config={"export_format": "onnx", "dynamic": True},
    )
    return ExportedComponent(program, metadata)


def _fold_logits_to_keep(program: onnx_ir.Model) -> None:
    """A decoder taking how many trailing positions to return logits for (`num_logits_to_keep`, Gemma 3's): 0, every
    position, as the runtime reads them."""
    import onnx_ir

    graph = program.graph
    keep = next((value for value in graph.inputs if value.name == "num_logits_to_keep"), None)
    if keep is None:
        return
    zero = onnx_ir.node("Constant", [], attributes={"value": onnx_ir.tensor(torch.tensor(0, dtype=torch.int64))})
    graph.insert_before(graph.node(0), [zero])
    onnx_ir.convenience.replace_all_uses_with(keep, zero.outputs[0])
    graph.inputs.remove(keep)


def _bool_inputs_as_int64(program: onnx_ir.Model) -> None:
    """A bool input (SmolVLM's `pixel_attention_mask`) taken as the int64 mask a processor returns, cast in the graph."""
    import onnx_ir

    graph = program.graph
    for value in list(graph.inputs):
        if value.dtype != onnx_ir.DataType.BOOL:
            continue
        cast = onnx_ir.node("Cast", [value], attributes={"to": onnx_ir.DataType.BOOL})
        onnx_ir.convenience.replace_all_uses_with(value, cast.outputs[0])
        # `replace_all_uses_with` also rewired the cast's own input
        cast.replace_input_with(0, value)
        value.dtype = onnx_ir.DataType.INT64
        graph.insert_before(graph.node(0), [cast])


def _load_onnx_multimodal(
    pretrained_model_name_or_path, config, decoder_file: str, **kwargs
) -> tuple[dict[str, ExportedComponent], bool]:
    """An optimum-onnx multimodal export (onnx-community's): `embed_tokens` (`input_ids` to `inputs_embeds`, and Gemma 3n's
    `per_layer_inputs`), `vision_encoder` / `audio_encoder` on their modality's data (the image grid fed as
    `grid_thw`), and a merged decoder on the embeddings.

    Raises:
        `ValueError`: for a decoder carrying a state besides its attention cache (LFM2-VL's convolutions).
    """
    folder = decoder_file.removesuffix("decoder_model_merged.onnx")
    decoder = _read_onnx(pretrained_model_name_or_path, decoder_file, **kwargs)
    geometry = _rename_onnx_cache(decoder)
    _drop_use_cache_branch(decoder)
    _fold_logits_to_keep(decoder)
    if states := [value.name for value in decoder.graph.inputs if value.name.startswith("past_")]:
        raise ValueError(
            f"{pretrained_model_name_or_path}'s decoder carries a state besides its attention cache ({states[0]}, ...), "
            "which this runtime doesn't feed an ONNX graph: re-export it with `OnnxExporter().export_for_generation`."
        )
    components = {"language_model": _onnx_decoder_component(decoder, config, geometry)}
    embedder = _read_onnx(pretrained_model_name_or_path, f"{folder}{ONNX_TEXT_EMBEDDINGS}", **kwargs)
    components["text_embeddings_model"] = _onnx_graph_component(embedder, config)
    for file, name in _ONNX_MODALITY_ENCODERS.items():
        if _find(pretrained_model_name_or_path, (f"{folder}{file}",), **kwargs) is None:
            continue
        encoder = _read_onnx(pretrained_model_name_or_path, f"{folder}{file}", **kwargs)
        for value in encoder.graph.inputs:
            # The image grid, which the runtime feeds a modality graph as `grid_thw`
            if value.name == "image_grid_thw":
                value.name = "grid_thw"
        _bool_inputs_as_int64(encoder)
        components[name] = _onnx_graph_component(encoder, config)
    return components, True


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
    # Only the hidden states: an encoder exported with its attentions too (onnx-community's Whisper) returns them after
    encoder_outputs = [value for value in encoder.graph.outputs if value.name == "last_hidden_state"]
    if encoder_outputs:
        encoder.graph.outputs.clear()
        encoder.graph.outputs.extend(encoder_outputs)
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


def _empty_cache_in_graph(program: onnx_ir.Model) -> None:
    """A decoder exported for embeddings (Qwen3-Embedding's: `last_hidden_state` first, a cache in and out) as an
    encoder: its cache inputs built empty in the graph (`[batch, heads, 0, head_dim]`), its cache outputs dropped."""
    import onnx_ir

    graph = program.graph
    input_ids = next(value for value in graph.inputs if value.name == "input_ids")
    shape = onnx_ir.node("Shape", [input_ids])
    first = onnx_ir.node("Constant", [], attributes={"value": onnx_ir.tensor(torch.tensor([0], dtype=torch.int64))})
    batch = onnx_ir.node("Gather", [shape.outputs[0], first.outputs[0]])
    nodes = [shape, first, batch]
    for value in [value for value in graph.inputs if value.name.startswith("input.cache.")]:
        rest = torch.tensor([value.shape[1], 0, value.shape[3]], dtype=torch.int64)
        dims = onnx_ir.node("Constant", [], attributes={"value": onnx_ir.tensor(rest)})
        full = onnx_ir.node("Concat", [batch.outputs[0], dims.outputs[0]], attributes={"axis": 0})
        zero = onnx_ir.tensor(torch.zeros(1, dtype=getattr(torch, str(value.dtype.numpy()))))
        empty = onnx_ir.node("ConstantOfShape", [full.outputs[0]], attributes={"value": zero})
        nodes += [dims, full, empty]
        onnx_ir.convenience.replace_all_uses_with(value, empty.outputs[0])
        graph.inputs.remove(value)
    graph.insert_before(graph.node(0), nodes)
    outputs = [value for value in graph.outputs if not value.name.startswith("output.cache.")]
    graph.outputs.clear()
    graph.outputs.extend(outputs)


def _load_onnx(pretrained_model_name_or_path, config, **kwargs) -> tuple[dict[str, ExportedComponent], bool]:
    """A causal LM taking and returning its cache (inputs `input_ids`, `attention_mask`, maybe `position_ids`,
    `past_key_values.*`; outputs `logits`, `present.*`), an encoder-decoder, a multimodal export, or an
    encoder."""
    if (encoder_file := _find(pretrained_model_name_or_path, ONNX_ENCODER_FILES, **kwargs)) is not None:
        return _load_onnx_seq2seq(pretrained_model_name_or_path, config, encoder_file, **kwargs)
    file = _find(pretrained_model_name_or_path, ONNX_FILES, **kwargs)
    if file is None:
        raise ValueError(
            f"{pretrained_model_name_or_path} is an optimum-onnx decoder split in two graphs, the prompt's and the "
            "steps', each holding the weights, without the merged one: re-export it with "
            "`OnnxExporter().export_for_generation`, whose one graph serves both."
        )
    if file.endswith("decoder_model_merged.onnx"):
        folder = file.removesuffix("decoder_model_merged.onnx")
        if _find(pretrained_model_name_or_path, (f"{folder}{ONNX_TEXT_EMBEDDINGS}",), **kwargs) is not None:
            return _load_onnx_multimodal(pretrained_model_name_or_path, config, file, **kwargs)
    program = _read_onnx(pretrained_model_name_or_path, file, **kwargs)
    geometry = _rename_onnx_cache(program)
    _drop_use_cache_branch(program)
    graph = program.graph
    if not geometry:
        return {"model": _onnx_graph_component(program, config, by_name=True)}, False
    if graph.outputs[0].name == "last_hidden_state":
        _empty_cache_in_graph(program)
        return {"model": _onnx_graph_component(program, config, by_name=True)}, False
    if graph.outputs[0].name != "logits":
        raise ValueError(
            f"{pretrained_model_name_or_path} takes a cache but doesn't return logits first (outputs "
            f"{[value.name for value in graph.outputs]}): only a causal LM loads from optimum-onnx. Re-export it with "
            "`OnnxExporter`."
        )
    return {"model": _onnx_decoder_component(program, config, geometry)}, True


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
