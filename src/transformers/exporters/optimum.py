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
"""Loading a model an optimum library exported: optimum-intel's OpenVINO IRs, optimum-onnx's ONNX graphs and
optimum-executorch's programs. None writes export metadata, so it is rebuilt from the graphs and the model's config, with
a warning saying so. What a graph takes that `generate` doesn't give it is supplied here, not in the runtime.

- A causal LM runs through `generate`: optimum-intel's stateful one, optimum-onnx's taking and returning its cache
  (`model.onnx`, or a merged decoder), and optimum-executorch's keeping its cache in the program.
- An encoder-decoder runs through `generate` too: optimum-intel's (Whisper's, T5's), optimum-onnx's (an encoder and a
  merged decoder) and optimum-executorch's (`encoder` and `text_decoder` methods).
- An optimum-executorch multimodal model runs as the multimodal split (`token_embedding`, `vision_encoder` or
  `audio_encoder`, then `text_decoder`).
- An optimum-intel or optimum-onnx encoder (feature extraction, classification) is called with the graph's inputs and
  returns a dict by its output names.

Refused, as their graphs are laid out differently from this runtime's: optimum-intel's multimodal exports (vision
graphs split per model), optimum-onnx's decoders split in two graphs (each holding the weights), and optimum-executorch's
other programs (encoders: no published export to check their inputs against).
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Unpack

from ..utils import logging
from ..utils.hub import DownloadKwargs, cached_file, hf_api
from ..utils.import_utils import is_torch_available
from .metadata import ExportMetadata


if is_torch_available():
    import torch
    from torch.utils._pytree import tree_structure, treespec_dumps

if TYPE_CHECKING:
    import onnx_ir
    import openvino

    from .base import ExportedComponent, ExportedModel


logger = logging.get_logger(__name__)


def _find(pretrained_model_name_or_path, files, **kwargs: Unpack[DownloadKwargs]) -> str | None:
    """The first of `files` the export holds."""
    for file in files:
        if cached_file(pretrained_model_name_or_path, file, _raise_exceptions_for_missing_entries=False, **kwargs):
            return file
    return None


def _architecture(config) -> str | None:
    return (config.architectures or [None])[0]


# ── optimum-intel (OpenVINO) ────────────────────────────────────────────────

OPENVINO_MODEL_FILE = "openvino_model.xml"
# A multimodal export's: refused, its vision graphs split differently from the components here.
OPENVINO_LANGUAGE_MODEL_FILE = "openvino_language_model.xml"
# An encoder-decoder's two graphs, and the decoder's inputs under the names our decode graph takes (its own cross
# cache, written into state, as `export_for_generation` lays out an OpenVINO encoder-decoder).
OPENVINO_SEQ2SEQ_FILES = {"encoder_model": "openvino_encoder_model.xml", "decoder_model": "openvino_decoder_model.xml"}
_OPENVINO_SEQ2SEQ_DECODER_INPUTS = {
    "input_ids": "decoder_input_ids",
    "encoder_hidden_states": "encoder_outputs.last_hidden_state",
}
_OPENVINO_DTYPES = {"i64": "int64", "i32": "int32", "f32": "float32", "f16": "float16", "bf16": "bfloat16"}


def _read_openvino(pretrained_model_name_or_path, file: str, **kwargs) -> openvino.Model:
    import openvino

    xml_file = cached_file(pretrained_model_name_or_path, file, **kwargs)
    cached_file(pretrained_model_name_or_path, file.replace(".xml", ".bin"), **kwargs)
    return openvino.Core().read_model(xml_file)


def _openvino_traced(port) -> dict[str, Any]:
    # A dynamic axis as the length of 2 a decode capture records.
    dims = [dimension.get_length() if dimension.is_static else 2 for dimension in port.get_partial_shape()]
    return {"rank": len(dims), "shape": dims, "dtype": _OPENVINO_DTYPES[port.get_element_type().get_type_name()]}


def _openvino_metadata(config, kwargs: dict[str, Any], outputs: Any, stateful: bool) -> ExportMetadata:
    """The metadata an export would have recorded for an IR: its kwargs, given as their example values (a tensor
    port's from the IR), and `outputs`, the structure it returns."""
    traced = {name: value if isinstance(value, dict) else _openvino_traced(value) for name, value in kwargs.items()}
    return ExportMetadata(
        architecture=_architecture(config),
        export_dtype="float32",
        export_device="cpu",
        kwargs=traced,
        in_spec=treespec_dumps(tree_structure(((), dict.fromkeys(kwargs, 0)))),
        out_spec=treespec_dumps(tree_structure(outputs)),
        export_config={"export_format": "openvino", "dynamic": True, "stateful": stateful},
    )


def _load_openvino(pretrained_model_name_or_path, config, **kwargs) -> tuple[dict[str, ExportedComponent], bool]:
    """A stateful causal LM (inputs `input_ids`, `attention_mask`, `position_ids`, `beam_idx`; output `logits`; the
    cache as state), an encoder-decoder (an encoder graph, and a decoder that writes its own cross cache into state), or
    an encoder."""
    from ..modeling_outputs import BaseModelOutput, CausalLMOutputWithPast, Seq2SeqLMOutput
    from .base import ExportedComponent

    if _find(pretrained_model_name_or_path, (OPENVINO_LANGUAGE_MODEL_FILE,), **kwargs) is not None:
        raise ValueError(
            f"{pretrained_model_name_or_path} is an optimum-intel multimodal export, whose vision graphs are split "
            "differently from this runtime's: re-export it with `OpenVINOExporter().export_for_generation`."
        )
    if _find(pretrained_model_name_or_path, (OPENVINO_SEQ2SEQ_FILES["decoder_model"],), **kwargs) is not None:
        encoder = _read_openvino(pretrained_model_name_or_path, OPENVINO_SEQ2SEQ_FILES["encoder_model"], **kwargs)
        decoder = _read_openvino(pretrained_model_name_or_path, OPENVINO_SEQ2SEQ_FILES["decoder_model"], **kwargs)
        for port in decoder.inputs:
            if (name := _OPENVINO_SEQ2SEQ_DECODER_INPUTS.get(port.get_any_name())) is not None:
                port.get_tensor().set_names({name})
        hidden = BaseModelOutput(last_hidden_state=torch.zeros(1))
        encoder_kwargs = {port.get_any_name(): port for port in encoder.inputs}
        decoder_kwargs = {
            "decoder_input_ids": decoder.input("decoder_input_ids"),
            "encoder_outputs": {"container": "BaseModelOutput"},
            "cache": {"container": "EncoderDecoderCache"},
        }
        encoder_metadata = _openvino_metadata(config, encoder_kwargs, hidden, stateful=False)
        decoder_metadata = _openvino_metadata(config, decoder_kwargs, Seq2SeqLMOutput(logits=torch.zeros(1)), True)
        # The output class of the encoder, which the runtime reads off the decode graph's input spec.
        decoder_metadata.in_spec = treespec_dumps(
            tree_structure(((), {"decoder_input_ids": 0, "encoder_outputs": hidden, "cache": 0}))
        )
        components = {
            "encoder_model": ExportedComponent(encoder, encoder_metadata),
            "decoder_model": ExportedComponent(decoder, decoder_metadata),
        }
        return components, True

    program = _read_openvino(pretrained_model_name_or_path, OPENVINO_MODEL_FILE, **kwargs)
    inputs = {port.get_any_name(): port for port in program.inputs if port.get_any_name() != "beam_idx"}
    output_names = [port.get_any_name() for port in program.outputs]
    if len(inputs) == len(program.inputs):
        outputs = dict.fromkeys(output_names, torch.zeros(1))
        return {"model": ExportedComponent(program, _openvino_metadata(config, inputs, outputs, False))}, False
    if output_names[:1] != ["logits"]:
        raise ValueError(
            f"{pretrained_model_name_or_path} keeps a state but doesn't return logits (outputs {output_names}): only "
            "a stateful causal LM loads from optimum-intel. Re-export it with `OpenVINOExporter`."
        )
    inputs["cache"] = {"container": "DynamicCache"}
    metadata = _openvino_metadata(config, inputs, CausalLMOutputWithPast(logits=torch.zeros(1)), stateful=True)
    return {"model": ExportedComponent(program, metadata)}, True


# ── optimum-onnx (ONNX) ──────────────────────────────────────────────────────────

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
    if Path(pretrained_model_name_or_path).is_dir():
        siblings = [str(Path(file).parent / p.name) for p in path.parent.iterdir() if p.name.startswith(data_prefix)]
    else:
        repo_files = hf_api().list_repo_files(
            pretrained_model_name_or_path, revision=kwargs.get("revision"), token=kwargs.get("token")
        )
        siblings = [
            name
            for name in repo_files
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
    from ..cache_utils import DynamicCache, EncoderDecoderCache
    from ..modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
    from .base import ExportedComponent
    from .cache import materialize_cache_layers
    from .exporter_dynamo import register_pytree_node

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
    from ..cache_utils import DynamicCache
    from ..modeling_outputs import CausalLMOutputWithPast
    from .base import ExportedComponent
    from .cache import materialize_cache_layers
    from .exporter_dynamo import register_pytree_node

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


# ── optimum-executorch ──────────────────────────────────────────────────────

EXECUTORCH_MODEL_FILE = "model.pte"
# Its program names no inputs, but each task's methods take them in a fixed order: a causal LM's `forward` (which
# reports its cache with `use_kv_cache`), an encoder-decoder's `encoder` and `text_decoder`, and a multimodal model's
# `token_embedding`, `<modality>_encoder` and `text_decoder` on the embeddings.
_EXECUTORCH_DTYPES = {3: "int32", 4: "int64", 5: "float16", 6: "float32", 15: "bfloat16"}
# A multimodal program's encoders, by the component and input they serve
_EXECUTORCH_ENCODERS = {
    "vision_encoder": ("vision_embeddings_model", "pixel_values"),
    "audio_encoder": ("audio_embeddings_model", "input_features"),
}


class _ProgramMethod:
    """A program whose `forward` is another of its methods: the runner runs `forward`."""

    def __init__(self, program, method: str):
        self.program = program
        self.method = method

    def load_method(self, name: str):
        return self.program.load_method(self.method if name == "forward" else name)


def _executorch_traced(method_meta, index: int) -> dict[str, Any]:
    tensor = method_meta.input_tensor_meta(index)
    sizes = list(tensor.sizes())
    return {"rank": len(sizes), "shape": sizes, "dtype": _EXECUTORCH_DTYPES[int(tensor.dtype())]}


def _executorch_component(config, program, method: str, names: tuple, inputs: dict, outputs: Any) -> ExportedComponent:
    """`method` as a component: its inputs, positional, under `names`; `inputs` the structure they take (an encoder's
    outputs nested); its cache written at `cache_position` and read up to it, so a new sequence needs no reset."""
    from .base import ExportedComponent

    meta = program.metadata(method)
    traced = {name: _executorch_traced(meta, index) for index, name in enumerate(names) if name != "encoder_outputs"}
    if "cache_position" in names:
        # `generate` makes a cache only for a forward taking one, and slices each step's ids off it
        traced["cache"] = {"container": "EncoderDecoderCache" if "encoder_outputs" in names else "DynamicCache"}
    if "encoder_outputs" in names:
        traced["encoder_outputs"] = {"container": "BaseModelOutput"}
    max_cache_len = program.load_method("get_max_seq_len").execute([])[0]
    metadata = ExportMetadata(
        architecture=_architecture(config),
        export_dtype="float32",
        export_device="cpu",
        kwargs=traced,
        in_spec=treespec_dumps(tree_structure(((), inputs))),
        out_spec=treespec_dumps(tree_structure(outputs)),
        export_config={"export_format": "executorch"},
        cache_geometry={"max_cache_len": int(max_cache_len)},
    )
    return ExportedComponent(_ProgramMethod(program, method), metadata)


def _load_executorch(pretrained_model_name_or_path, config, **kwargs) -> tuple[dict[str, ExportedComponent], bool]:
    """A causal LM keeping its cache in the program's buffers (`forward(input_ids, cache_position)`, the logits), an
    encoder-decoder (`encoder(input_features)`, then `text_decoder(decoder_input_ids, encoder_hidden_states,
    cache_position)`), or a multimodal model (`token_embedding(input_ids)`, `vision_encoder(pixel_values)` or
    `audio_encoder(input_features)`, then `text_decoder(inputs_embeds, cache_position)`)."""
    from ..modeling_outputs import BaseModelOutput, CausalLMOutputWithPast, Seq2SeqLMOutput
    from .exporter_executorch import ExecutorchExporter

    # Its programs use quantized and LLM kernels (embeddings, SDPA with a cache), which importing registers.
    for module in ("executorch.kernels.quantized", "executorch.extension.llm.custom_ops.custom_ops"):
        try:
            importlib.import_module(module)
        except ImportError:
            pass
    program = ExecutorchExporter.load_program(
        cached_file(pretrained_model_name_or_path, EXECUTORCH_MODEL_FILE, **kwargs)
    )
    methods = set(program.method_names)
    logits = CausalLMOutputWithPast(logits=torch.zeros(1))

    def component(method, names, inputs=None, outputs=logits):
        inputs = inputs if inputs is not None else dict.fromkeys(names, 0)
        return _executorch_component(config, program, method, names, inputs, outputs)

    if {"token_embedding", "text_decoder"} <= methods:
        components = {
            "text_embeddings_model": component("token_embedding", ("input_ids",), outputs=torch.zeros(1)),
            "language_model": component("text_decoder", ("inputs_embeds", "cache_position")),
        }
        for method, (name, input_name) in _EXECUTORCH_ENCODERS.items():
            if method in methods:
                components[name] = component(method, (input_name,), outputs=torch.zeros(1))
        return components, True
    if {"encoder", "text_decoder"} <= methods:
        hidden = BaseModelOutput(last_hidden_state=torch.zeros(1))
        decoder_names = ("decoder_input_ids", "encoder_outputs", "cache_position")
        decoder_inputs = {"decoder_input_ids": 0, "encoder_outputs": hidden, "cache_position": 0}
        return {
            "encoder_model": component("encoder", ("input_features",), outputs=hidden),
            "decoder_model": component(
                "text_decoder", decoder_names, decoder_inputs, Seq2SeqLMOutput(logits=torch.zeros(1))
            ),
        }, True
    if "use_kv_cache" in methods and "forward" in methods and program.metadata("forward").num_inputs() == 2:
        return {"model": component("forward", ("input_ids", "cache_position"))}, True
    raise ValueError(
        f"{pretrained_model_name_or_path} is an optimum-executorch export whose task this runtime doesn't read yet "
        f"(methods {sorted(methods)}): re-export it with `ExecutorchExporter`."
    )


# ── Generation ──────────────────────────────────────────────────────────────


class _OptimumGenerationMixin:
    """What an optimum export's decode graph takes that `generate` doesn't give it: an all-ones padding mask, which
    `generate` drops (optimum-intel's and optimum-onnx's causal LMs take one), and `cache_position`, which it no longer
    passes (optimum-executorch's graphs take one). And the cross-attention cache an optimum-onnx merged decoder returns
    placeholders for once it has one."""

    def _call(self, **kwargs) -> Any:
        outputs = super()._call(**kwargs)
        given, cache = kwargs.get("cache"), getattr(outputs, "cache", None)
        if hasattr(given, "cross_attention_cache") and hasattr(cache, "cross_attention_cache"):
            for layer, given_layer in zip(cache.cross_attention_cache.layers, given.cross_attention_cache.layers):
                if layer.keys.shape[0] == 0:
                    layer.keys, layer.values = given_layer.keys, given_layer.values
        return outputs

    def _validate_model_kwargs(self, model_kwargs: dict):
        """Also accept a processor's token types (Gemma 3's), which a graph with its attention built in doesn't take."""
        traced = self._decode_metadata().kwargs
        if "token_type_ids" not in traced:
            model_kwargs = {key: value for key, value in model_kwargs.items() if key != "token_type_ids"}
        super()._validate_model_kwargs(model_kwargs)

    def prepare_inputs_for_generation(self, *args, attention_mask=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, attention_mask=attention_mask, **kwargs)
        traced = self._decode_metadata().kwargs
        # The running sequence, of which the step is the tail
        sequence = args[0] if args else kwargs["input_ids"]
        step = model_inputs.get("decoder_input_ids" if self.config.is_encoder_decoder else "input_ids")
        mask = traced.get("attention_mask", {})
        if not self.config.is_encoder_decoder and mask.get("rank") == 2 and model_inputs.get("attention_mask") is None:
            model_inputs["attention_mask"] = torch.ones(
                sequence.shape, dtype=getattr(torch, mask["dtype"]), device=sequence.device
            )
        if "cache_position" in traced and step is not None:
            model_inputs["cache_position"] = torch.arange(
                sequence.shape[1] - step.shape[1], sequence.shape[1], device=step.device
            )
        return model_inputs


def _optimum_model_class(model_class: type) -> type:
    """`model_class` as an optimum export runs: with `_OptimumGenerationMixin` for one `generate` drives."""
    from .generation import ExportedModelForCausalLM

    if not issubclass(model_class, ExportedModelForCausalLM):
        return model_class
    return type(model_class.__name__, (_OptimumGenerationMixin, model_class), {"__module__": __name__})


# ── Loading ─────────────────────────────────────────────────────────────────

# Per library: the files that tell its exports apart, its format, its loader and the exporter to re-export with.
_LIBRARIES = {
    "optimum-intel": (
        (OPENVINO_MODEL_FILE, OPENVINO_SEQ2SEQ_FILES["decoder_model"], OPENVINO_LANGUAGE_MODEL_FILE),
        "openvino",
        _load_openvino,
        "OpenVINOExporter",
    ),
    "optimum-onnx": (ONNX_FILES + ONNX_ENCODER_FILES + ONNX_SPLIT_FILES, "onnx", _load_onnx, "OnnxExporter"),
    "optimum-executorch": ((EXECUTORCH_MODEL_FILE,), "executorch", _load_executorch, "ExecutorchExporter"),
}


def optimum_library(pretrained_model_name_or_path, **kwargs: Unpack[DownloadKwargs]) -> str | None:
    """Which optimum library exported `pretrained_model_name_or_path`, if one did."""
    for library, (files, *_) in _LIBRARIES.items():
        if _find(pretrained_model_name_or_path, files, **kwargs) is not None:
            return library
    return None


def load_optimum_export(
    pretrained_model_name_or_path, cls: type, device=None, runner_kwargs=None, **kwargs: Unpack[DownloadKwargs]
) -> ExportedModel:
    """An optimum export as an [`ExportedModel`] of `cls`, on `device`: its metadata rebuilt from its graphs and the
    model's config, with a warning that it is.

    Raises:
        `OSError`: if no optimum library wrote it either.
        `ValueError`: for an export whose graphs are laid out differently from this runtime's (see the module's
            docstring).
    """
    from ..generation import GenerationConfig
    from ..models.auto import AutoConfig
    from .base import EXPORT_MANIFEST_FILE, _exported_model_class

    library = optimum_library(pretrained_model_name_or_path, **kwargs)
    if library is None:
        raise OSError(
            f"{pretrained_model_name_or_path} has no {EXPORT_MANIFEST_FILE}: it isn't an export saved with "
            "`ExportedModel.save_pretrained`, nor one an optimum library wrote."
        )
    _, export_format, load, exporter = _LIBRARIES[library]
    logger.warning(
        f"{pretrained_model_name_or_path} is an {library} export, which writes no export metadata: it is rebuilt from "
        "its graphs, so calls take the graphs' inputs, an encoder's outputs come back as a dict rather than the model's "
        f"output class, and it can't be saved again. Re-export the model with `{exporter}` (`export`, or "
        "`export_for_generation` for a generative model) and `save_pretrained` it for the full runtime."
    )
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    try:
        generation_config = GenerationConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    except OSError:
        generation_config = None
    components, generation = load(pretrained_model_name_or_path, config, **kwargs)
    model_class = _optimum_model_class(_exported_model_class(components, generation=generation))
    if not issubclass(model_class, cls):
        raise TypeError(
            f"{pretrained_model_name_or_path} runs as a {model_class.__name__}, not a {cls.__name__}: load it with "
            f"`{model_class.__name__}.from_pretrained` or `ExportedModel.from_pretrained`."
        )
    model = model_class(
        components, export_format, config=config, generation_config=generation_config, generation=generation
    )
    return model.to(device, **(runner_kwargs or {}))
