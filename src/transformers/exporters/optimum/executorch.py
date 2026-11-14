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
"""Loading optimum-executorch's programs: a causal LM keeping its cache in the program, an encoder-decoder (`encoder` and
`text_decoder` methods) and a multimodal model (`token_embedding`, `<modality>_encoder`, then `text_decoder`). Its
programs name no inputs, so each task's methods are told apart by name and fed in a fixed order."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from ...utils.hub import cached_file
from ...utils.import_utils import is_torch_available
from ..metadata import ExportMetadata
from .common import _architecture


if is_torch_available():
    import torch
    from torch.utils._pytree import tree_structure, treespec_dumps

if TYPE_CHECKING:
    from ..base import ExportedComponent


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
    from ..base import ExportedComponent

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
    from ...modeling_outputs import BaseModelOutput, CausalLMOutputWithPast, Seq2SeqLMOutput
    from ..exporter_executorch import ExecutorchExporter

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
