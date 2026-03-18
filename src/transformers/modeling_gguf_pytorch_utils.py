# Copyright 2024 The ggml.ai team and The HuggingFace Inc. team. and pygguf author (github.com/99991)
# https://github.com/99991/pygguf
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

from .core_model_loading import (
    Concatenate,
    GGUFQuantizedTensor,
    Transpose,
)
from .gguf_conversion_ops import (
    BloomReshapeQKVBias,
    BloomReshapeQKVWeight,
    GGUFDequantizer,
    LogNegate,
    ReversePermuteAttnK,
    ReversePermuteAttnQ,
    SubtractOne,
    Unsqueeze,
)
from .integrations import (
    GGUF_CONFIG_DEFAULTS_MAPPING,
    GGUF_CONFIG_MAPPING,
    GGUF_TOKENIZER_MAPPING,
    _gguf_parse_value,
)
from .utils import is_torch_available
from .utils.import_utils import is_gguf_available
from .utils.logging import get_logger


logger = get_logger(__name__)


GGUF_TO_TRANSFORMERS_MAPPING = {
    "ignore": {
        "GGUF": {
            "version": "version",
            "tensor_count": "tensor_count",
            "kv_count": "kv_count",
        },
        "general": {"file_type": "file_type", "quantization_version": "quantization_version"},
    },
    "config": GGUF_CONFIG_MAPPING,
    "tokenizer": {"tokenizer": GGUF_TOKENIZER_MAPPING["tokenizer"]},
    "tokenizer_config": {"tokenizer": GGUF_TOKENIZER_MAPPING["tokenizer_config"]},
}

GGUF_SUPPORTED_ARCHITECTURES = list(GGUF_TO_TRANSFORMERS_MAPPING["config"].keys())


# ---------------------------------------------------------------------------
# Static per-architecture GGUF→HF converter tables
# ---------------------------------------------------------------------------
# Each entry is a list of GGUFDequantizer instances.
# `*` in source patterns becomes `(\d+)` (layer-index capture group).
# `*` in target patterns becomes `\1`  (backreference, resolved at rename time).

# Shared building blocks
_ROPE_ATTN_CONVERTERS = [
    GGUFDequantizer("blk.*.attn_q.weight", "model.layers.*.self_attn.q_proj.weight", [ReversePermuteAttnQ()]),
    GGUFDequantizer("blk.*.attn_k.weight", "model.layers.*.self_attn.k_proj.weight", [ReversePermuteAttnK()]),
]

_NORM_SUBTRACT_ONE_CONVERTERS = [
    GGUFDequantizer("blk.*.attn_norm.weight", "model.layers.*.input_layernorm.weight", [SubtractOne()]),
    GGUFDequantizer("blk.*.ffn_norm.weight", "model.layers.*.post_attention_layernorm.weight", [SubtractOne()]),
]

_NORM_CONVERTERS = [
    GGUFDequantizer("blk.*.attn_norm.weight", "model.layers.*.input_layernorm.weight"),
    GGUFDequantizer("blk.*.ffn_norm.weight", "model.layers.*.post_attention_layernorm.weight"),
]

_LLAMA_SHARED = [
    GGUFDequantizer("blk.*.attn_v.weight", "model.layers.*.self_attn.v_proj.weight"),
    GGUFDequantizer("blk.*.attn_output.weight", "model.layers.*.self_attn.o_proj.weight"),
    GGUFDequantizer("blk.*.ffn_gate.weight", "model.layers.*.mlp.gate_proj.weight"),
    GGUFDequantizer("blk.*.ffn_up.weight", "model.layers.*.mlp.up_proj.weight"),
    GGUFDequantizer("blk.*.ffn_down.weight", "model.layers.*.mlp.down_proj.weight"),
    GGUFDequantizer("token_embd.weight", "model.embed_tokens.weight"),
    GGUFDequantizer("output_norm.weight", "model.norm.weight"),
    GGUFDequantizer("output.weight", "lm_head.weight"),
]

_LLAMA_CONVERTERS = _ROPE_ATTN_CONVERTERS + _NORM_CONVERTERS + _LLAMA_SHARED
_NEMOTRON_CONVERTERS = _ROPE_ATTN_CONVERTERS + _NORM_SUBTRACT_ONE_CONVERTERS + _LLAMA_SHARED
_GEMMA_CONVERTERS = _ROPE_ATTN_CONVERTERS + _NORM_SUBTRACT_ONE_CONVERTERS + _LLAMA_SHARED

_QWEN2_MOE_CONVERTERS = [
    *_ROPE_ATTN_CONVERTERS,
    *_NORM_CONVERTERS,
    GGUFDequantizer("blk.*.attn_v.", "model.layers.*.self_attn.v_proj."),
    GGUFDequantizer("blk.*.attn_k.bias", "model.layers.*.self_attn.k_proj.bias"),
    GGUFDequantizer("blk.*.attn_q.bias", "model.layers.*.self_attn.q_proj.bias"),

    GGUFDequantizer("blk.*.attn_output.weight", "model.layers.*.self_attn.o_proj.weight"),
    GGUFDequantizer("blk.*.ffn_down_shexp.weight", "model.layers.*.mlp.shared_expert.down_proj.weight"),
    GGUFDequantizer("blk.*.ffn_gate_shexp.weight", "model.layers.*.mlp.shared_expert.gate_proj.weight"),
    GGUFDequantizer("blk.*.ffn_up_shexp.weight", "model.layers.*.mlp.shared_expert.up_proj.weight"),
    GGUFDequantizer("blk.*.ffn_gate_inp.weight", "model.layers.*.mlp.gate.weight"),
    GGUFDequantizer("blk.*.ffn_gate_inp_shexp", "model.layers.*.mlp.shared_expert_gate", [Unsqueeze(0)]),
    GGUFDequantizer(
        ["blk.*.ffn_gate_exps.weight", "blk.*.ffn_up_exps.weight"],
        "model.layers.*.mlp.experts.gate_up_proj",
        [Concatenate(dim=1)],
    ),
    GGUFDequantizer("blk.*.ffn_down_exps.weight", "model.layers.*.mlp.experts.down_proj"),
    GGUFDequantizer("token_embd.weight", "model.embed_tokens.weight"),
    GGUFDequantizer("output_norm.weight", "model.norm.weight"),
    GGUFDequantizer("output.weight", "lm_head.weight"),
]

_GGUF_ARCH_CONVERTERS: dict[str, list[GGUFDequantizer]] = {
    # RoPE llama-family
    "llama": _LLAMA_CONVERTERS,
    "mistral": _LLAMA_CONVERTERS,
    "phi3": _LLAMA_CONVERTERS,
    "cohere": _LLAMA_CONVERTERS,
    "qwen2": _LLAMA_CONVERTERS,
    "qwen3": _LLAMA_CONVERTERS,
    # Norm subtract-one variants
    "nemotron": _NEMOTRON_CONVERTERS,
    "gemma2": _GEMMA_CONVERTERS,
    "gemma3_text": _GEMMA_CONVERTERS,
    # Bloom
    "bloom": [
        GGUFDequantizer("blk.*.attn_norm.weight", "transformer.h.*.ln_attn.weight"),
        GGUFDequantizer("blk.*.attn_norm.bias", "transformer.h.*.ln_attn.bias"),
        GGUFDequantizer("blk.*.attn_qkv.weight", "transformer.h.*.self_attention.query_key_value.weight", [BloomReshapeQKVWeight()]),
        GGUFDequantizer("blk.*.attn_qkv.bias", "transformer.h.*.self_attention.query_key_value.bias", [BloomReshapeQKVBias()]),
        GGUFDequantizer("blk.*.attn_output.weight", "transformer.h.*.self_attention.dense.weight"),
        GGUFDequantizer("blk.*.attn_output.bias", "transformer.h.*.self_attention.dense.bias"),
        GGUFDequantizer("blk.*.ffn_norm.weight", "transformer.h.*.ln_mlp.weight"),
        GGUFDequantizer("blk.*.ffn_norm.bias", "transformer.h.*.ln_mlp.bias"),
        GGUFDequantizer("blk.*.ffn_up.weight", "transformer.h.*.mlp.dense_h_to_4h.weight"),
        GGUFDequantizer("blk.*.ffn_up.bias", "transformer.h.*.mlp.dense_h_to_4h.bias"),
        GGUFDequantizer("blk.*.ffn_down.weight", "transformer.h.*.mlp.dense_4h_to_h.weight"),
        GGUFDequantizer("blk.*.ffn_down.bias", "transformer.h.*.mlp.dense_4h_to_h.bias"),
        GGUFDequantizer("token_embd.weight", "transformer.word_embeddings.weight"),
        GGUFDequantizer("token_embd.bias", "transformer.word_embeddings.bias"),
        GGUFDequantizer("token_embd_norm.weight", "transformer.word_embeddings_layernorm.weight"),
        GGUFDequantizer("token_embd_norm.bias", "transformer.word_embeddings_layernorm.bias"),
        GGUFDequantizer("output_norm.weight", "transformer.ln_f.weight"),
        GGUFDequantizer("output_norm.bias", "transformer.ln_f.bias"),
    ],
    # GPT2
    "gpt2": [
        GGUFDequantizer("blk.*.attn_qkv.weight", "transformer.h.*.attn.c_attn.weight", [Transpose()]),
        GGUFDequantizer("blk.*.attn_qkv.bias", "transformer.h.*.attn.c_attn.bias"),
        GGUFDequantizer("blk.*.attn_output.weight", "transformer.h.*.attn.c_proj.weight", [Transpose()]),
        GGUFDequantizer("blk.*.attn_output.bias", "transformer.h.*.attn.c_proj.bias"),
        GGUFDequantizer("blk.*.ffn_up.weight", "transformer.h.*.mlp.c_fc.weight", [Transpose()]),
        GGUFDequantizer("blk.*.ffn_up.bias", "transformer.h.*.mlp.c_fc.bias"),
        GGUFDequantizer("blk.*.ffn_down.weight", "transformer.h.*.mlp.c_proj.weight", [Transpose()]),
        GGUFDequantizer("blk.*.ffn_down.bias", "transformer.h.*.mlp.c_proj.bias"),
        GGUFDequantizer("blk.*.attn_norm.weight", "transformer.h.*.ln_1.weight"),
        GGUFDequantizer("blk.*.attn_norm.bias", "transformer.h.*.ln_1.bias"),
        GGUFDequantizer("blk.*.ffn_norm.weight", "transformer.h.*.ln_2.weight"),
        GGUFDequantizer("blk.*.ffn_norm.bias", "transformer.h.*.ln_2.bias"),
        GGUFDequantizer("token_embd.weight", "transformer.wte.weight"),
        GGUFDequantizer("position_embd.weight", "transformer.wpe.weight"),
        GGUFDequantizer("output_norm.weight", "transformer.ln_f.weight"),
        GGUFDequantizer("output_norm.bias", "transformer.ln_f.bias"),
        GGUFDequantizer("output.weight", "lm_head.weight"),
    ],
    # Mamba
    "mamba": [
        GGUFDequantizer("blk.*.ssm_in.weight", "backbone.layers.*.mixer.in_proj.weight"),
        GGUFDequantizer("blk.*.ssm_conv1d.weight", "backbone.layers.*.mixer.conv1d.weight", [Unsqueeze(1)]),
        GGUFDequantizer("blk.*.ssm_conv1d.bias", "backbone.layers.*.mixer.conv1d.bias"),
        GGUFDequantizer("blk.*.ssm_x.weight", "backbone.layers.*.mixer.x_proj.weight"),
        GGUFDequantizer("blk.*.ssm_dt.weight", "backbone.layers.*.mixer.dt_proj.weight"),
        GGUFDequantizer("blk.*.ssm_dt.bias", "backbone.layers.*.mixer.dt_proj.bias"),
        GGUFDequantizer("blk.*.ssm_a", "backbone.layers.*.mixer.A_log", [LogNegate()]),
        GGUFDequantizer("blk.*.ssm_d", "backbone.layers.*.mixer.D"),
        GGUFDequantizer("blk.*.ssm_out.weight", "backbone.layers.*.mixer.out_proj.weight"),
        GGUFDequantizer("blk.*.attn_norm.weight", "backbone.layers.*.norm.weight"),
        GGUFDequantizer("token_embd.weight", "backbone.embedding.weight"),
        GGUFDequantizer("output_norm.weight", "backbone.norm_f.weight"),
        GGUFDequantizer("output.weight", "lm_head.weight"),
    ],
    # LFM2
    "lfm2": [
        *_ROPE_ATTN_CONVERTERS,
        *_NORM_CONVERTERS,
        *_LLAMA_SHARED,
        GGUFDequantizer("blk.*.shortconv.conv.weight", "model.layers.*.conv.weight", [Unsqueeze(1)]),
    ],
    # Qwen2MoE / Qwen3MoE
    "qwen2_moe": _QWEN2_MOE_CONVERTERS,
    "qwen3_moe": _QWEN2_MOE_CONVERTERS,
}


def get_gguf_converters(model_type: str) -> list[GGUFDequantizer]:
    """Return the static list of GGUFDequantizer converters for a given HF model type."""
    return list(_GGUF_ARCH_CONVERTERS.get(model_type, []))


def read_field(reader, field):
    if field not in reader.fields:
        return []
    value = reader.fields[field]
    return [_gguf_parse_value(value.parts[_data_index], value.types) for _data_index in value.data]


def load_gguf_checkpoint(gguf_checkpoint_path, return_tensors=False):
    """
    Load a GGUF file and return a dictionary of parsed parameters containing tensors, the parsed
    tokenizer and config attributes.

    Args:
        gguf_checkpoint_path (`str`):
            The path the to GGUF file to load
        return_tensors (`bool`, defaults to `False`):
            Whether to read the tensors from the file and return them. Not doing so is faster
            and only loads the metadata in memory.
    """
    if is_gguf_available() and is_torch_available():
        from gguf import GGUFReader
    else:
        logger.error(
            "Loading a GGUF checkpoint in PyTorch, requires both PyTorch and GGUF>=0.10.0 to be installed. Please see "
            "https://pytorch.org/ and https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise ImportError("Please install torch and gguf>=0.10.0 to load a GGUF checkpoint in PyTorch.")

    reader = GGUFReader(gguf_checkpoint_path)
    fields = reader.fields
    reader_keys = list(fields.keys())

    parsed_parameters = {k: {} for k in GGUF_TO_TRANSFORMERS_MAPPING}

    architecture = read_field(reader, "general.architecture")[0]
    # NOTE: Some GGUF checkpoints may miss `general.name` field in metadata
    model_name = read_field(reader, "general.name")

    updated_architecture = None
    # in llama.cpp mistral models use the same architecture as llama. We need
    # to add this patch to ensure things work correctly on our side.
    if "llama" in architecture and "mistral" in model_name:
        updated_architecture = "mistral"
    # FIXME: Currently this implementation is only for flan-t5 architecture.
    # It needs to be developed for supporting legacy t5.
    elif "t5" in architecture or "t5encoder" in architecture:
        parsed_parameters["config"]["is_gated_act"] = True
        if model_name and "umt5" in model_name[0].lower():
            updated_architecture = "umt5"
            if "t5encoder" in architecture:
                parsed_parameters["config"]["architectures"] = ["UMT5EncoderModel"]
        else:
            if "t5encoder" in architecture:
                parsed_parameters["config"]["architectures"] = ["T5EncoderModel"]
            updated_architecture = "t5"
    else:
        updated_architecture = architecture

    if "qwen2moe" in architecture:
        updated_architecture = "qwen2_moe"
    elif "qwen3moe" in architecture:
        updated_architecture = "qwen3_moe"

    # For stablelm architecture, we need to set qkv_bias and use_parallel_residual from tensors
    # If `qkv_bias=True`, qkv_proj with bias will be present in the tensors
    # If `use_parallel_residual=False`, ffn_norm will be present in the tensors
    if "stablelm" in architecture:
        attn_bias_name = {"attn_q.bias", "attn_k.bias", "attn_v.bias"}
        ffn_norm_name = "ffn_norm"
        qkv_bias = any(bias_name in tensor.name for tensor in reader.tensors for bias_name in attn_bias_name)
        use_parallel_residual = any(ffn_norm_name in tensor.name for tensor in reader.tensors)
        parsed_parameters["config"]["use_qkv_bias"] = qkv_bias
        parsed_parameters["config"]["use_parallel_residual"] = not use_parallel_residual

    if architecture not in GGUF_SUPPORTED_ARCHITECTURES and updated_architecture not in GGUF_SUPPORTED_ARCHITECTURES:
        raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")

    # Handle tie_word_embeddings, if lm_head.weight is not present in tensors,
    # tie_word_embeddings is true otherwise false
    exceptions = ["falcon", "bloom"]
    parsed_parameters["config"]["tie_word_embeddings"] = (
        all(tensor.name != "output.weight" for tensor in reader.tensors) or architecture in exceptions
    )

    # Set GGUF-specific default values
    config_defaults = GGUF_CONFIG_DEFAULTS_MAPPING.get(
        updated_architecture, GGUF_CONFIG_DEFAULTS_MAPPING.get(architecture) or {}
    )
    for key, value in config_defaults.items():
        parsed_parameters["config"].setdefault(key, value)

    # List all key-value pairs in a columnized format
    for gguf_key, field in reader.fields.items():
        gguf_key = gguf_key.replace(architecture, updated_architecture)
        split = gguf_key.split(".")
        prefix = split[0]
        config_key = ".".join(split[1:])

        value = [_gguf_parse_value(field.parts[_data_index], field.types) for _data_index in field.data]

        if len(value) == 1:
            value = value[0]

        if isinstance(value, str) and architecture in value:
            value = value.replace(architecture, updated_architecture)

        for parameter, parameter_renames in GGUF_TO_TRANSFORMERS_MAPPING.items():
            if prefix in parameter_renames and config_key in parameter_renames[prefix]:
                renamed_config_key = parameter_renames[prefix][config_key]
                if renamed_config_key == -1:
                    continue

                if renamed_config_key is not None:
                    parsed_parameters[parameter][renamed_config_key] = value

                if gguf_key in reader_keys:
                    reader_keys.remove(gguf_key)

        if gguf_key in reader_keys:
            logger.info(f"Some keys were not parsed and added into account {gguf_key} | {value}")

    # Gemma3 GGUF checkpoint only contains weights of text backbone
    if parsed_parameters["config"]["model_type"] == "gemma3":
        parsed_parameters["config"]["model_type"] = "gemma3_text"

    if parsed_parameters["config"]["model_type"] == "lfm2":
        gguf_num_key_value_heads = parsed_parameters["config"]["num_key_value_heads"]
        # LFM2 GGUF checkpoint defines num_key_value_heads as a list of integers .e.g [0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0] but we need to set it to the max value for HF
        parsed_parameters["config"]["num_key_value_heads"] = max(gguf_num_key_value_heads)
        ## we already read the correct intermediate_size from the GGUF checkpoint so we need to set block_auto_adjust_ff_dim to False
        parsed_parameters["config"]["block_auto_adjust_ff_dim"] = False

        ## llama.cpp defines the layers that are full-attention by looking at num_key_value_heads
        ## we need to set the full_attn_idxs to the layers that are full-attention
        parsed_parameters["config"]["full_attn_idxs"] = [
            i for i, num_kv_heads in enumerate(gguf_num_key_value_heads) if num_kv_heads > 0
        ]

    # retrieve config vocab_size from tokenizer
    # Please refer to https://github.com/huggingface/transformers/issues/32526 for more details
    if "vocab_size" not in parsed_parameters["config"]:
        tokenizer_parameters = parsed_parameters["tokenizer"]
        if "tokens" in tokenizer_parameters:
            parsed_parameters["config"]["vocab_size"] = len(tokenizer_parameters["tokens"])
        else:
            logger.warning(
                "Can't find a way to retrieve missing config vocab_size from tokenizer parameters. "
                "This will use default value from model config class and cause unexpected behavior."
            )

    if return_tensors:
        config = parsed_parameters.get("config", {})
        model_type = config.get("model_type", architecture)

        parsed_parameters["tensors"] = {
            tensor.name: GGUFQuantizedTensor(tensor.data, tensor.tensor_type)
            for tensor in reader.tensors
        }
        parsed_parameters["weight_mapping"] = get_gguf_converters(model_type)

    if len(reader_keys) > 0:
        logger.info(f"Some keys of the GGUF file were not considered: {reader_keys}")

    return parsed_parameters
