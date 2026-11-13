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
"""Loading a model an optimum library exported: optimum-intel's OpenVINO IRs, optimum-onnx's ONNX graphs and optimum-executorch's
programs. None writes export metadata, so it is rebuilt from the graphs and the model's config:

- A causal LM runs through `generate`: optimum-intel's stateful one, optimum-onnx's taking and returning its cache
  (`model.onnx`, or a merged decoder), and optimum-executorch's keeping its cache in the program.
- An optimum-intel encoder-decoder (Whisper's, T5's) runs through `generate` too.
- An optimum-intel or optimum-onnx encoder (feature extraction, classification) is called with the graph's inputs and
  returns a dict by its output names.

Refused, as their graphs are laid out differently from this runtime's: multimodal exports (vision graphs split per
model), optimum-onnx's encoder-decoders (separate decoder and cross caches) and decoders split in two graphs (each
holding the weights), and optimum-executorch's other programs (no input names to bind).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Unpack

from ..utils import logging
from ..utils.hub import DownloadKwargs, cached_file, hf_api
from ..utils.import_utils import is_onnxscript_available, is_openvino_available, is_torch_available
from .metadata import ExportMetadata


if is_torch_available():
    import torch
    from torch.utils._pytree import tree_structure, treespec_dumps

if is_openvino_available():
    import openvino

if is_onnxscript_available():
    import onnx_ir

if TYPE_CHECKING:
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
# An encoder-decoder's encoder: refused, its decoder's caches split differently from the runtime's.
ONNX_ENCODER_FILES = ("encoder_model.onnx", "onnx/encoder_model.onnx")
# A decoder split in two graphs, the prompt's and the steps', each holding the weights: refused.
ONNX_SPLIT_FILES = ("decoder_with_past_model.onnx", "onnx/decoder_with_past_model.onnx")

_ONNX_PAST = re.compile(r"past_key_values\.(\d+)\.(key|value)")
_ONNX_PRESENT = re.compile(r"present\.(\d+)\.(key|value)")


def _read_onnx(pretrained_model_name_or_path, file: str, **kwargs: Unpack[DownloadKwargs]) -> onnx_ir.Model:
    """The graph, its weights fetched beside it (optimum writes them to `<file>_data`, split over `<file>_data_<n>`
    past 2GB) and read into memory: the graph is renamed, so ONNX Runtime opens it serialized rather than from this
    file."""
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
    """Name the cache inputs and outputs as an export of the runtime does (`input.cache.layers.<i>.keys`, ...), and
    return each layer's `(num_kv_heads, key_head_dim, value_head_dim)`."""
    geometry: dict[int, list[int]] = {}
    for value in program.graph.inputs:
        if match := _ONNX_PAST.fullmatch(value.name):
            index, kind = int(match.group(1)), match.group(2)
            value.name = f"input.cache.layers.{index}.{kind}s"
            heads, dim = value.shape[1], value.shape[3]
            geometry.setdefault(index, [heads, dim, dim])[1 if kind == "key" else 2] = dim
    for value in program.graph.outputs:
        if match := _ONNX_PRESENT.fullmatch(value.name):
            value.name = f"output.cache.layers.{match.group(1)}.{match.group(2)}s"
    return {index: tuple(shape) for index, shape in sorted(geometry.items())}


def _drop_use_cache_branch(program: onnx_ir.Model) -> None:
    """A merged decoder picks its branch from a `use_cache_branch` input: compute it in the graph instead, from whether
    the cache holds anything."""
    graph = program.graph
    flag = next((value for value in graph.inputs if value.name == "use_cache_branch"), None)
    if flag is None:
        return
    past = next(value for value in graph.inputs if value.name == "input.cache.layers.0.keys")
    # Shape + Gather rather than `Shape(start, end)`, which older opsets lack.
    shape = onnx_ir.node("Shape", [past])
    axis = onnx_ir.node("Constant", [], attributes={"value": onnx_ir.tensor(torch.tensor([2], dtype=torch.int64))})
    length = onnx_ir.node("Gather", [shape.outputs[0], axis.outputs[0]])
    zero = onnx_ir.node("Constant", [], attributes={"value": onnx_ir.tensor(torch.zeros(1, dtype=torch.int64))})
    has_past = onnx_ir.node("Greater", [length.outputs[0], zero.outputs[0]])
    graph.insert_before(graph.node(0), [shape, axis, length, zero, has_past])
    onnx_ir.convenience.replace_all_uses_with(flag, has_past.outputs[0])
    graph.inputs.remove(flag)


def _load_onnx(pretrained_model_name_or_path, config, **kwargs) -> tuple[dict[str, ExportedComponent], bool]:
    """A causal LM taking and returning its cache (inputs `input_ids`, `attention_mask`, maybe `position_ids`,
    `past_key_values.*`; outputs `logits`, `present.*`), or an encoder."""
    from ..cache_utils import DynamicCache
    from ..modeling_outputs import CausalLMOutputWithPast
    from .base import ExportedComponent
    from .cache import materialize_cache_layers
    from .exporter_dynamo import register_pytree_node

    if _find(pretrained_model_name_or_path, ONNX_ENCODER_FILES, **kwargs) is not None:
        raise ValueError(
            f"{pretrained_model_name_or_path} is an optimum-onnx encoder-decoder, whose decoder's caches are split "
            "differently from this runtime's: re-export it with `OnnxExporter().export_for_generation`."
        )
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
        cache_geometry={
            "kv": {str(index): list(shape) for index, shape in geometry.items()},
            "indexers": {},
            "linear_states": {},
        },
    )
    return {"model": ExportedComponent(program, metadata)}, True


# ── optimum-executorch ──────────────────────────────────────────────────────

EXECUTORCH_MODEL_FILE = "model.pte"
# A causal LM's `forward`, whose program names no inputs; it reports its cache with `use_kv_cache`.
_EXECUTORCH_CAUSAL_LM_INPUTS = ("input_ids", "cache_position")


def _load_executorch(pretrained_model_name_or_path, config, **kwargs) -> tuple[dict[str, ExportedComponent], bool]:
    """A causal LM keeping its cache in the program's buffers: `forward(input_ids, cache_position)` returns the logits,
    a token at a time."""
    from ..modeling_outputs import CausalLMOutputWithPast
    from .base import ExportedComponent
    from .exporter_executorch import ExecutorchExporter

    program = ExecutorchExporter.load_program(
        cached_file(pretrained_model_name_or_path, EXECUTORCH_MODEL_FILE, **kwargs)
    )
    forward = program.metadata("forward")
    if "use_kv_cache" not in program.method_names or forward.num_inputs() != len(_EXECUTORCH_CAUSAL_LM_INPUTS):
        raise ValueError(
            f"{pretrained_model_name_or_path} is an optimum-executorch export other than a causal LM (methods "
            f"{sorted(program.method_names)}), whose program names no inputs to bind: re-export it with "
            "`ExecutorchExporter`."
        )
    traced = {}
    for index, name in enumerate(_EXECUTORCH_CAUSAL_LM_INPUTS):
        sizes = list(forward.input_tensor_meta(index).sizes())
        traced[name] = {"rank": len(sizes), "shape": sizes, "dtype": "int64"}
    traced["cache"] = {"container": "DynamicCache"}
    max_cache_len = program.load_method("get_max_seq_len").execute([])[0]
    metadata = ExportMetadata(
        architecture=_architecture(config),
        export_dtype="float32",
        export_device="cpu",
        kwargs=traced,
        in_spec=treespec_dumps(tree_structure(((), dict.fromkeys(_EXECUTORCH_CAUSAL_LM_INPUTS, 0)))),
        out_spec=treespec_dumps(tree_structure(CausalLMOutputWithPast(logits=torch.zeros(1)))),
        # Its cache is written at `cache_position` and read up to it: a new sequence needs no reset.
        export_config={"export_format": "executorch"},
        cache_geometry={"max_cache_len": int(max_cache_len)},
    )
    return {"model": ExportedComponent(program, metadata)}, True


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


def load_optimum_export(pretrained_model_name_or_path, **kwargs: Unpack[DownloadKwargs]) -> ExportedModel:
    """An optimum export as an [`ExportedModel`], its metadata rebuilt from its graphs and the model's config.

    Raises:
        `ValueError`: for an export whose graphs are laid out differently from this runtime's (see the module's
            docstring).
    """
    from ..generation import GenerationConfig
    from ..models.auto import AutoConfig
    from .base import _exported_model_class

    library = optimum_library(pretrained_model_name_or_path, **kwargs)
    _, export_format, load, exporter = _LIBRARIES[library]
    logger.warning(
        f"{pretrained_model_name_or_path} is an {library} export: its metadata is rebuilt from its graphs, so calls "
        "take the graphs' inputs, and an encoder's outputs come back as a dict rather than the model's output class. "
        f"Re-export the model with `{exporter}` (`export`, or `export_for_generation` for a generative model) and "
        "`save_pretrained` it for the full runtime."
    )
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    try:
        generation_config = GenerationConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    except OSError:
        generation_config = None
    components, generation = load(pretrained_model_name_or_path, config, **kwargs)
    model_class = _exported_model_class(components, generation=generation)
    return model_class(
        components, export_format, config=config, generation_config=generation_config, generation=generation
    )
