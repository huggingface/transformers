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
"""Loading a model that optimum-intel exported to OpenVINO as one graph (`openvino_model.xml`): a stateful causal LM,
for the generation runtime, or an encoder. Its export metadata, which optimum-intel doesn't write, is rebuilt from the
IR's ports and the model's config."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Unpack

from ..utils import logging
from ..utils.hub import DownloadKwargs, cached_file
from ..utils.import_utils import is_openvino_available, is_torch_available
from .metadata import ExportMetadata


if is_torch_available():
    import torch
    from torch.utils._pytree import tree_structure, treespec_dumps

if is_openvino_available():
    import openvino

if TYPE_CHECKING:
    from .base import ExportArtifacts


logger = logging.get_logger(__name__)

OPTIMUM_INTEL_MODEL_FILE = "openvino_model.xml"
# A multimodal export's: refused, its vision graphs split differently from the components here.
OPTIMUM_INTEL_LANGUAGE_MODEL_FILE = "openvino_language_model.xml"

_OPENVINO_DTYPES = {"i64": "int64", "i32": "int32", "f32": "float32", "f16": "float16", "bf16": "bfloat16"}


# An encoder-decoder's two graphs, and the decoder's inputs under the names our decode graph takes (its own cross
# cache, written into state, as `export_for_generation` lays out an OpenVINO encoder-decoder).
OPTIMUM_INTEL_SEQ2SEQ_FILES = {
    "encoder_model": "openvino_encoder_model.xml",
    "decoder_model": "openvino_decoder_model.xml",
}
_SEQ2SEQ_DECODER_INPUTS = {
    "input_ids": "decoder_input_ids",
    "encoder_hidden_states": "encoder_outputs.last_hidden_state",
}


def is_optimum_intel(pretrained_model_name_or_path, **kwargs: Unpack[DownloadKwargs]) -> bool:
    """Whether `pretrained_model_name_or_path` holds an optimum-intel export this module reads."""
    files = [OPTIMUM_INTEL_MODEL_FILE, OPTIMUM_INTEL_SEQ2SEQ_FILES["decoder_model"], OPTIMUM_INTEL_LANGUAGE_MODEL_FILE]
    return any(
        cached_file(pretrained_model_name_or_path, file, _raise_exceptions_for_missing_entries=False, **kwargs)
        is not None
        for file in files
    )


def _read(pretrained_model_name_or_path, file: str, **kwargs) -> openvino.Model:
    xml_file = cached_file(pretrained_model_name_or_path, file, **kwargs)
    cached_file(pretrained_model_name_or_path, file.replace(".xml", ".bin"), **kwargs)
    return openvino.Core().read_model(xml_file)


def _traced(port) -> dict[str, Any]:
    # A dynamic axis as the length of 2 a decode capture records.
    dims = [dimension.get_length() if dimension.is_static else 2 for dimension in port.get_partial_shape()]
    return {"rank": len(dims), "shape": dims, "dtype": _OPENVINO_DTYPES[port.get_element_type().get_type_name()]}


def _metadata(program, config, kwargs: dict[str, Any], outputs: Any, stateful: bool) -> ExportMetadata:
    """The metadata an export would have recorded for `program`: its kwargs, given as their example values (a tensor
    port's from the IR), and `outputs`, the structure it returns."""
    traced = {name: value if isinstance(value, dict) else _traced(value) for name, value in kwargs.items()}
    return ExportMetadata(
        architecture=(config.architectures or [None])[0],
        dtype="float32",
        device="cpu",
        kwargs=traced,
        input_names=[port.get_any_name() for port in program.inputs],
        output_names=[port.get_any_name() for port in program.outputs][
            : len(torch.utils._pytree.tree_leaves(outputs))
        ],
        in_spec=treespec_dumps(tree_structure(((), dict.fromkeys(kwargs, 0)))),
        out_spec=treespec_dumps(tree_structure(outputs)),
        export_config={"export_format": "openvino", "dynamic": True, "stateful": stateful},
    )


def load_optimum_intel(pretrained_model_name_or_path, **kwargs: Unpack[DownloadKwargs]) -> ExportArtifacts:
    """An optimum-intel export as [`ExportArtifacts`]: a stateful causal LM (inputs `input_ids`, `attention_mask`,
    `position_ids`, `beam_idx`; output `logits`; the cache as state) or encoder-decoder (Whisper's, T5's: an encoder
    graph, and a decoder that writes its own cross cache into state) that the generation runtime drives, or an
    encoder (feature extraction, classification), whose outputs come back as a dict by the IR's output names.

    Raises:
        `ValueError`: for a multimodal export (several graphs around a language model), whose vision graphs don't
            match the components here.
    """
    from ..configuration_utils import PreTrainedConfig
    from ..generation import GenerationConfig
    from ..modeling_outputs import BaseModelOutput, CausalLMOutputWithPast, Seq2SeqLMOutput
    from ..models.auto import AutoConfig
    from .base import ExportArtifacts, ExportedComponent

    logger.warning(
        f"{pretrained_model_name_or_path} is an optimum-intel export, loaded as a legacy format: its metadata is "
        "rebuilt from the OpenVINO IR: calls take the IR's inputs, and an encoder's outputs come back as a dict "
        "rather than the model's output class. Re-export the model with `OpenVINOExporter` (`export`, or "
        "`export_for_generation` for a generative model) and `save_pretrained` it for the full runtime."
    )
    config: PreTrainedConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    try:
        generation_config = GenerationConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    except OSError:
        generation_config = None
    files = {
        file: cached_file(pretrained_model_name_or_path, file, _raise_exceptions_for_missing_entries=False, **kwargs)
        for file in (
            OPTIMUM_INTEL_MODEL_FILE,
            *OPTIMUM_INTEL_SEQ2SEQ_FILES.values(),
            OPTIMUM_INTEL_LANGUAGE_MODEL_FILE,
        )
    }
    if files[OPTIMUM_INTEL_LANGUAGE_MODEL_FILE] is not None:
        raise ValueError(
            f"{pretrained_model_name_or_path} is an optimum-intel multimodal export, whose vision graphs don't match "
            "the components here: re-export it with `OpenVINOExporter().export_for_generation`."
        )

    def artifacts(components: dict, generation: bool) -> ExportArtifacts:
        return ExportArtifacts(
            components, "openvino", config=config, generation_config=generation_config, generation=generation
        )

    if files[OPTIMUM_INTEL_SEQ2SEQ_FILES["decoder_model"]] is not None:
        encoder = _read(pretrained_model_name_or_path, OPTIMUM_INTEL_SEQ2SEQ_FILES["encoder_model"], **kwargs)
        decoder = _read(pretrained_model_name_or_path, OPTIMUM_INTEL_SEQ2SEQ_FILES["decoder_model"], **kwargs)
        for port in decoder.inputs:
            if (name := _SEQ2SEQ_DECODER_INPUTS.get(port.get_any_name())) is not None:
                port.get_tensor().set_names({name})
        hidden = BaseModelOutput(last_hidden_state=torch.zeros(1))
        encoder_kwargs = {port.get_any_name(): port for port in encoder.inputs}
        decoder_kwargs = {
            "decoder_input_ids": decoder.input("decoder_input_ids"),
            "encoder_outputs": {"container": "BaseModelOutput"},
            "past_key_values": {"container": "EncoderDecoderCache"},
        }
        encoder_metadata = _metadata(encoder, config, encoder_kwargs, hidden, stateful=False)
        decoder_metadata = _metadata(decoder, config, decoder_kwargs, Seq2SeqLMOutput(logits=torch.zeros(1)), True)
        # The output class of the encoder, which the runtime reads off the decode graph's input spec.
        decoder_metadata.in_spec = treespec_dumps(
            tree_structure(((), {"decoder_input_ids": 0, "encoder_outputs": hidden, "past_key_values": 0}))
        )
        return artifacts(
            {
                "encoder_model": ExportedComponent(encoder, encoder_metadata),
                "decoder_model": ExportedComponent(decoder, decoder_metadata),
            },
            generation=True,
        )

    program = _read(pretrained_model_name_or_path, OPTIMUM_INTEL_MODEL_FILE, **kwargs)
    inputs = {port.get_any_name(): port for port in program.inputs if port.get_any_name() != "beam_idx"}
    output_names = [port.get_any_name() for port in program.outputs]
    if len(inputs) == len(program.inputs):
        outputs = dict.fromkeys(output_names, torch.zeros(1))
        return artifacts(
            {"model": ExportedComponent(program, _metadata(program, config, inputs, outputs, False))}, False
        )
    if output_names[:1] != ["logits"]:
        raise ValueError(
            f"{pretrained_model_name_or_path} keeps a state but doesn't return logits (outputs {output_names}): only "
            "a stateful causal LM loads from optimum-intel. Re-export it with `OpenVINOExporter`."
        )
    inputs["past_key_values"] = {"container": "DynamicCache"}
    metadata = _metadata(program, config, inputs, CausalLMOutputWithPast(logits=torch.zeros(1)), stateful=True)
    return artifacts({"model": ExportedComponent(program, metadata)}, generation=True)
