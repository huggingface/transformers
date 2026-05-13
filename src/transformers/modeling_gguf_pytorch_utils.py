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



# --- Llama-family (rope-permuted Q/K, plain Llama norms) --------------------
# Constructing the rename rules pulls in torch via WeightRenaming/WeightConverter,
# so we keep that behind ``is_torch_available()`` — ``load_gguf_checkpoint`` is
# imported by ``tokenization_utils_tokenizers`` even in tokenizer-only installs.
_GGUF_ARCH_CONVERTERS: dict[str, list] = {}

if is_torch_available():
    from .core_model_loading import Concatenate, Transpose, WeightConverter, WeightRenaming
    from .gguf_conversion_ops import (
        BloomReshapeQKVBias,
        BloomReshapeQKVWeight,
        LogNegate,
        ReversePermuteAttnK,
        ReversePermuteAttnQ,
        SubtractOne,
        Unsqueeze,
    )
    from .integrations.gguf_dequant import GGUFQuantizedTensor  # noqa: F401  (referenced by load_gguf_checkpoint)

    _BLK_PREFIX = WeightRenaming(r"^blk\.", "model.layers.")
    _LLAMA_SHARED_RENAMES = [
        _BLK_PREFIX,
        WeightRenaming(r"^token_embd\.weight", "model.embed_tokens.weight"),
        WeightRenaming(r"^output_norm\.weight", "model.norm.weight"),
        WeightRenaming(r"^output\.weight", "lm_head.weight"),
        WeightRenaming(r"\.attn_v\.weight", ".self_attn.v_proj.weight"),
        WeightRenaming(r"\.attn_output\.weight", ".self_attn.o_proj.weight"),
        WeightRenaming(r"\.ffn_(gate|up|down)\.weight", r".mlp.\1_proj.weight"),
        # Attn biases exist on some Llama-family archs (Qwen2/3, StableLM, …) and not others;
        # the rules are no-ops when the GGUF doesn't ship those tensors.
        WeightRenaming(r"\.attn_(q|k|v)\.bias", r".self_attn.\1_proj.bias"),
    ]
    _ROPE_ATTN_CONVERTERS = [
        WeightConverter(
            source_patterns=r"\.attn_q\.weight",
            target_patterns=".self_attn.q_proj.weight",
            operations=[ReversePermuteAttnQ()],
        ),
        WeightConverter(
            source_patterns=r"\.attn_k\.weight",
            target_patterns=".self_attn.k_proj.weight",
            operations=[ReversePermuteAttnK()],
        ),
    ]
    _NORM_RENAMES = [
        WeightRenaming(r"\.attn_norm\.weight", ".input_layernorm.weight"),
        WeightRenaming(r"\.ffn_norm\.weight", ".post_attention_layernorm.weight"),
    ]
    _NORM_SUBTRACT_ONE_CONVERTERS = [
        WeightConverter(
            source_patterns=r"\.attn_norm\.weight",
            target_patterns=".input_layernorm.weight",
            operations=[SubtractOne()],
        ),
        WeightConverter(
            source_patterns=r"\.ffn_norm\.weight",
            target_patterns=".post_attention_layernorm.weight",
            operations=[SubtractOne()],
        ),
    ]

    _LLAMA_CONVERTERS = _LLAMA_SHARED_RENAMES + _NORM_RENAMES + _ROPE_ATTN_CONVERTERS
    _NEMOTRON_CONVERTERS = _LLAMA_SHARED_RENAMES + _NORM_SUBTRACT_ONE_CONVERTERS + _ROPE_ATTN_CONVERTERS
    _GEMMA_CONVERTERS = _NEMOTRON_CONVERTERS  # same structure as Nemotron

    _T5_CONVERTERS = [
        WeightRenaming(r"^enc\.blk\.", "encoder.block."),
        WeightRenaming(r"^dec\.blk\.", "decoder.block."),
        WeightRenaming(r"^enc\.output_norm\.weight", "encoder.final_layer_norm.weight"),
        WeightRenaming(r"^dec\.output_norm\.weight", "decoder.final_layer_norm.weight"),
        WeightRenaming(r"^token_embd\.weight", "shared.weight"),
        WeightRenaming(r"^output\.weight", "lm_head.weight"),
        # Self-attention (layer.0) — shared by encoder + decoder
        WeightRenaming(r"\.attn_(q|k|v|o)\.weight", r".layer.0.SelfAttention.\1.weight"),
        WeightRenaming(r"\.attn_rel_b\.weight", ".layer.0.SelfAttention.relative_attention_bias.weight"),
        WeightRenaming(r"\.attn_norm\.weight", ".layer.0.layer_norm.weight"),
        # Cross-attention (layer.1) — decoder only
        WeightRenaming(r"\.cross_attn_(q|k|v|o)\.weight", r".layer.1.EncDecAttention.\1.weight"),
        WeightRenaming(r"\.cross_attn_norm\.weight", ".layer.1.layer_norm.weight"),
        # FFN — encoder uses layer.1, decoder uses layer.2. Inject the layer index
        # via two structural renames so the per-tensor renames below are shared.
        WeightRenaming(r"^encoder\.block\.(\d+)\.ffn_", r"encoder.block.\1.layer.1.ffn_"),
        WeightRenaming(r"^decoder\.block\.(\d+)\.ffn_", r"decoder.block.\1.layer.2.ffn_"),
        WeightRenaming(r"\.ffn_norm\.weight", ".layer_norm.weight"),
        WeightRenaming(r"\.ffn_gate\.weight", ".DenseReluDense.wi_0.weight"),
        WeightRenaming(r"\.ffn_up\.weight", ".DenseReluDense.wi_1.weight"),
        WeightRenaming(r"\.ffn_down\.weight", ".DenseReluDense.wo.weight"),
    ]

    # --- StableLM (Llama-like + optional q_norm/k_norm and attn biases) --------
    _STABLELM_CONVERTERS = _LLAMA_CONVERTERS + [
        WeightRenaming(r"\.attn_(q|k)_norm\.weight", r".self_attn.\1_layernorm.weight"),
        WeightRenaming(r"\.attn_(q|k|v)\.bias", r".self_attn.\1_proj.bias"),
    ]

    # --- Starcoder2 (Llama-style attn + single-layer c_fc/c_proj MLP, biases) --
    _STARCODER2_CONVERTERS = [
        _BLK_PREFIX,
        WeightRenaming(r"^token_embd\.weight", "model.embed_tokens.weight"),
        WeightRenaming(r"^output_norm\.(weight|bias)", r"model.norm.\1"),
        WeightRenaming(r"^output\.weight", "lm_head.weight"),
        WeightRenaming(r"\.attn_v\.(weight|bias)", r".self_attn.v_proj.\1"),
        WeightRenaming(r"\.attn_output\.(weight|bias)", r".self_attn.o_proj.\1"),
        WeightRenaming(r"\.attn_(q|k)\.bias", r".self_attn.\1_proj.bias"),
        WeightRenaming(r"\.ffn_up\.(weight|bias)", r".mlp.c_fc.\1"),
        WeightRenaming(r"\.ffn_down\.(weight|bias)", r".mlp.c_proj.\1"),
        WeightRenaming(r"\.attn_norm\.(weight|bias)", r".input_layernorm.\1"),
        WeightRenaming(r"\.ffn_norm\.(weight|bias)", r".post_attention_layernorm.\1"),
        *_ROPE_ATTN_CONVERTERS,
    ]

    # --- Falcon (merged QKV, dense_h_to_4h MLP, 7B vs 40B norm variants) -------
    _FALCON_CONVERTERS = [
        WeightRenaming(r"^blk\.", "transformer.h."),
        WeightRenaming(r"^token_embd\.weight", "transformer.word_embeddings.weight"),
        WeightRenaming(r"^output_norm\.(weight|bias)", r"transformer.ln_f.\1"),
        WeightRenaming(r"^output\.weight", "lm_head.weight"),
        WeightRenaming(r"\.attn_qkv\.weight", ".self_attention.query_key_value.weight"),
        WeightRenaming(r"\.attn_output\.weight", ".self_attention.dense.weight"),
        WeightRenaming(r"\.ffn_up\.weight", ".mlp.dense_h_to_4h.weight"),
        WeightRenaming(r"\.ffn_down\.weight", ".mlp.dense_4h_to_h.weight"),
        # 7B style: parallel attn+mlp shares one input_layernorm
        WeightRenaming(r"\.attn_norm\.(weight|bias)", r".input_layernorm.\1"),
        # 40B style: separate ln_attn / ln_mlp
        WeightRenaming(r"\.attn_norm_2\.(weight|bias)", r".ln_mlp.\1"),
    ]

    # --- GPT-OSS (MoE: router + post_attention_norm + merged gate_up_proj) -----
    _GPT_OSS_CONVERTERS = [
        _BLK_PREFIX,
        WeightRenaming(r"^token_embd\.weight", "model.embed_tokens.weight"),
        WeightRenaming(r"^output_norm\.weight", "model.norm.weight"),
        WeightRenaming(r"^output\.weight", "lm_head.weight"),
        WeightRenaming(r"\.attn_v\.weight", ".self_attn.v_proj.weight"),
        WeightRenaming(r"\.attn_output\.weight", ".self_attn.o_proj.weight"),
        WeightRenaming(r"\.attn_norm\.weight", ".input_layernorm.weight"),
        WeightRenaming(r"\.post_attention_norm\.weight", ".post_attention_layernorm.weight"),
        WeightRenaming(r"\.ffn_norm\.weight", ".post_attention_layernorm.weight"),
        WeightRenaming(r"\.ffn_gate_inp\.weight", ".mlp.router.weight"),
        WeightRenaming(r"\.ffn_down_exps\.weight", ".mlp.experts.down_proj"),
        *_ROPE_ATTN_CONVERTERS,
        WeightConverter(
            source_patterns=[r"\.ffn_gate_exps\.weight", r"\.ffn_up_exps\.weight"],
            target_patterns=".mlp.experts.gate_up_proj",
            operations=[Concatenate(dim=1)],
        ),
    ]

    # --- MiniMax-M2 (Qwen3-MoE-like + e_score_correction_bias) -----------------
    _MINIMAX_M2_CONVERTERS = [
        _BLK_PREFIX,
        WeightRenaming(r"^token_embd\.weight", "model.embed_tokens.weight"),
        WeightRenaming(r"^output_norm\.weight", "model.norm.weight"),
        WeightRenaming(r"^output\.weight", "lm_head.weight"),
        WeightRenaming(r"\.attn_v\.weight", ".self_attn.v_proj.weight"),
        WeightRenaming(r"\.attn_output\.weight", ".self_attn.o_proj.weight"),
        WeightRenaming(r"\.attn_(q|k)_norm\.weight", r".self_attn.\1_norm.weight"),
        WeightRenaming(r"\.attn_norm\.weight", ".input_layernorm.weight"),
        WeightRenaming(r"\.ffn_norm\.weight", ".post_attention_layernorm.weight"),
        WeightRenaming(r"\.ffn_gate_inp\.weight", ".mlp.gate.weight"),
        WeightRenaming(r"\.exp_probs_b\.bias", ".mlp.e_score_correction_bias"),
        WeightRenaming(r"\.ffn_down_exps\.weight", ".mlp.experts.down_proj"),
        *_ROPE_ATTN_CONVERTERS,
        WeightConverter(
            source_patterns=[r"\.ffn_gate_exps\.weight", r"\.ffn_up_exps\.weight"],
            target_patterns=".mlp.experts.gate_up_proj",
            operations=[Concatenate(dim=1)],
        ),
    ]

    # --- Qwen2-MoE / Qwen3-MoE (shared experts + merged gate_up_proj) ----------
    _QWEN2_MOE_CONVERTERS = [
        _BLK_PREFIX,
        WeightRenaming(r"^token_embd\.weight", "model.embed_tokens.weight"),
        WeightRenaming(r"^output_norm\.weight", "model.norm.weight"),
        WeightRenaming(r"^output\.weight", "lm_head.weight"),
        WeightRenaming(r"\.attn_v\.weight", ".self_attn.v_proj.weight"),
        WeightRenaming(r"\.attn_output\.weight", ".self_attn.o_proj.weight"),
        WeightRenaming(r"\.attn_(q|k|v)\.bias", r".self_attn.\1_proj.bias"),
        WeightRenaming(r"\.attn_norm\.weight", ".input_layernorm.weight"),
        WeightRenaming(r"\.ffn_norm\.weight", ".post_attention_layernorm.weight"),
        WeightRenaming(r"\.ffn_(gate|up|down)_shexp\.weight", r".mlp.shared_expert.\1_proj.weight"),
        WeightRenaming(r"\.ffn_gate_inp\.weight", ".mlp.gate.weight"),
        WeightRenaming(r"\.ffn_down_exps\.weight", ".mlp.experts.down_proj"),
        *_ROPE_ATTN_CONVERTERS,
        WeightConverter(
            source_patterns=r"\.ffn_gate_inp_shexp",
            target_patterns=".mlp.shared_expert_gate",
            operations=[Unsqueeze(0)],
        ),
        WeightConverter(
            source_patterns=[r"\.ffn_gate_exps\.weight", r"\.ffn_up_exps\.weight"],
            target_patterns=".mlp.experts.gate_up_proj",
            operations=[Concatenate(dim=1)],
        ),
    ]

    # --- Bloom -----------------------------------------------------------------
    _BLOOM_CONVERTERS = [
        _BLK_PREFIX,
        WeightRenaming(r"^token_embd\.(weight|bias)", r"transformer.word_embeddings.\1"),
        WeightRenaming(r"^token_embd_norm\.(weight|bias)", r"transformer.word_embeddings_layernorm.\1"),
        WeightRenaming(r"^output_norm\.(weight|bias)", r"transformer.ln_f.\1"),
        WeightRenaming(r"^model\.layers\.", "transformer.h."),
        WeightRenaming(r"\.attn_norm\.(weight|bias)", r".ln_attn.\1"),
        WeightRenaming(r"\.attn_output\.(weight|bias)", r".self_attention.dense.\1"),
        WeightRenaming(r"\.ffn_norm\.(weight|bias)", r".ln_mlp.\1"),
        WeightRenaming(r"\.ffn_up\.(weight|bias)", r".mlp.dense_h_to_4h.\1"),
        WeightRenaming(r"\.ffn_down\.(weight|bias)", r".mlp.dense_4h_to_h.\1"),
        WeightConverter(
            source_patterns=r"\.attn_qkv\.weight",
            target_patterns=".self_attention.query_key_value.weight",
            operations=[BloomReshapeQKVWeight()],
        ),
        WeightConverter(
            source_patterns=r"\.attn_qkv\.bias",
            target_patterns=".self_attention.query_key_value.bias",
            operations=[BloomReshapeQKVBias()],
        ),
    ]

    # --- GPT-2 (Transpose on every Linear weight, bias kept as-is) -------------
    # GGUF stores GPT-2 ``c_attn``/``c_proj``/``c_fc`` weights transposed relative to HF.
    # The TARGET names differ per source (attn_qkv→c_attn, attn_output/ffn_down→c_proj,
    # ffn_up→c_fc), so the four Transpose converters can't be merged into one rule.
    _GPT2_CONVERTERS = [
        _BLK_PREFIX,
        WeightRenaming(r"^token_embd\.weight", "transformer.wte.weight"),
        WeightRenaming(r"^position_embd\.weight", "transformer.wpe.weight"),
        WeightRenaming(r"^output_norm\.(weight|bias)", r"transformer.ln_f.\1"),
        WeightRenaming(r"^output\.weight", "lm_head.weight"),
        WeightRenaming(r"^model\.layers\.", "transformer.h."),
        WeightRenaming(r"\.attn_qkv\.bias", ".attn.c_attn.bias"),
        WeightRenaming(r"\.attn_output\.bias", ".attn.c_proj.bias"),
        WeightRenaming(r"\.ffn_up\.bias", ".mlp.c_fc.bias"),
        WeightRenaming(r"\.ffn_down\.bias", ".mlp.c_proj.bias"),
        WeightRenaming(r"\.attn_norm\.(weight|bias)", r".ln_1.\1"),
        WeightRenaming(r"\.ffn_norm\.(weight|bias)", r".ln_2.\1"),
        WeightConverter(
            source_patterns=r"\.attn_qkv\.weight", target_patterns=".attn.c_attn.weight", operations=[Transpose()]
        ),
        WeightConverter(
            source_patterns=r"\.attn_output\.weight", target_patterns=".attn.c_proj.weight", operations=[Transpose()]
        ),
        WeightConverter(source_patterns=r"\.ffn_up\.weight", target_patterns=".mlp.c_fc.weight", operations=[Transpose()]),
        WeightConverter(
            source_patterns=r"\.ffn_down\.weight", target_patterns=".mlp.c_proj.weight", operations=[Transpose()]
        ),
    ]

    # --- Mamba (SSM) -----------------------------------------------------------
    _MAMBA_CONVERTERS = [
        WeightRenaming(r"^blk\.", "backbone.layers."),
        WeightRenaming(r"^token_embd\.weight", "backbone.embedding.weight"),
        WeightRenaming(r"^output_norm\.weight", "backbone.norm_f.weight"),
        WeightRenaming(r"^output\.weight", "lm_head.weight"),
        WeightRenaming(r"\.ssm_in\.weight", ".mixer.in_proj.weight"),
        WeightRenaming(r"\.ssm_conv1d\.bias", ".mixer.conv1d.bias"),
        WeightRenaming(r"\.ssm_x\.weight", ".mixer.x_proj.weight"),
        WeightRenaming(r"\.ssm_dt\.weight", ".mixer.dt_proj.weight"),
        WeightRenaming(r"\.ssm_dt\.bias", ".mixer.dt_proj.bias"),
        WeightRenaming(r"\.ssm_d$", ".mixer.D"),
        WeightRenaming(r"\.ssm_out\.weight", ".mixer.out_proj.weight"),
        WeightRenaming(r"\.attn_norm\.weight", ".norm.weight"),
        WeightConverter(
            source_patterns=r"\.ssm_conv1d\.weight",
            target_patterns=".mixer.conv1d.weight",
            operations=[Unsqueeze(1)],
        ),
        WeightConverter(source_patterns=r"\.ssm_a$", target_patterns=".mixer.A_log", operations=[LogNegate()]),
    ]

    # --- LFM2 (Llama + shortconv) ----------------------------------------------
    _LFM2_CONVERTERS = _LLAMA_CONVERTERS + [
        WeightConverter(
            source_patterns=r"\.shortconv\.conv\.weight",
            target_patterns=".conv.weight",
            operations=[Unsqueeze(1)],
        ),
    ]


    _GGUF_ARCH_CONVERTERS: dict[str, list] = {
        # RoPE Llama family
        "llama": _LLAMA_CONVERTERS,
        "mistral": _LLAMA_CONVERTERS,
        "phi3": _LLAMA_CONVERTERS,
        "cohere": _LLAMA_CONVERTERS,
        "qwen2": _LLAMA_CONVERTERS,
        "qwen3": _LLAMA_CONVERTERS,
        "deci": _LLAMA_CONVERTERS,
        # Norm-subtract-one variants
        "nemotron": _NEMOTRON_CONVERTERS,
        "gemma2": _GEMMA_CONVERTERS,
        "gemma3": _GEMMA_CONVERTERS,
        "gemma3_text": _GEMMA_CONVERTERS,
        # Misc archs
        "bloom": _BLOOM_CONVERTERS,
        "gpt2": _GPT2_CONVERTERS,
        "mamba": _MAMBA_CONVERTERS,
        "lfm2": _LFM2_CONVERTERS,
        "falcon": _FALCON_CONVERTERS,
        "stablelm": _STABLELM_CONVERTERS,
        "starcoder2": _STARCODER2_CONVERTERS,
        # MoE
        "qwen2_moe": _QWEN2_MOE_CONVERTERS,
        "qwen3_moe": _QWEN2_MOE_CONVERTERS,
        "minimax_m2": _MINIMAX_M2_CONVERTERS,
        "gpt_oss": _GPT_OSS_CONVERTERS,
        # T5 / UMT5 / T5-encoder share the same encoder–decoder mapping
        "t5": _T5_CONVERTERS,
        "t5encoder": _T5_CONVERTERS,
        "umt5": _T5_CONVERTERS,
    }


def get_gguf_converters(model_type: str) -> list:
    """Return the GGUF→HF rename rules (``WeightRenaming`` / ``WeightConverter``) for a given HF model type."""
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

    # MiniMax-M2: convert expert_gating_func integer to scoring_func string
    if parsed_parameters["config"].get("model_type") == "minimax_m2":
        _gating_func_map = {0: "none", 1: "softmax", 2: "sigmoid"}
        _scoring = parsed_parameters["config"].get("scoring_func")
        if isinstance(_scoring, int):
            parsed_parameters["config"]["scoring_func"] = _gating_func_map.get(_scoring, "softmax")

    # GPT-OSS: reconstruct rope_scaling from the architecture-prefixed metadata keys
    if updated_architecture == "gpt_oss":
        rope_type_field = reader.fields.get("gpt-oss.rope.scaling.type")
        if rope_type_field is not None:
            rope_type = rope_type_field.parts[0]
            if isinstance(rope_type, bytes):
                rope_type = rope_type.decode("utf-8")
            rope_scaling = {"rope_type": rope_type}
            for key in reader.fields:
                if not key.startswith("gpt-oss.rope.scaling.") or key.endswith(".type"):
                    continue
                suffix = key[len("gpt-oss.rope.scaling.") :]
                value = reader.fields[key].parts[0]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                if suffix in ("factor", "attention_factor", "beta_fast", "beta_slow"):
                    value = float(value)
                elif suffix in ("original_context_length", "original_max_position_embeddings"):
                    suffix = "original_max_position_embeddings"
                    value = int(value)
                rope_scaling[suffix] = value
            parsed_parameters["config"]["rope_scaling"] = rope_scaling

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

        # Wrap raw uint8 bytes in a ``torch.Tensor`` subclass that carries ``quant_type``.
        # ``GGUFDequantize`` does the actual dequant inside the WeightConverter chain,
        # on whatever device the loader has moved the bytes to.
        import torch  # local: keep top-of-file import-light when torch isn't required

        parsed_parameters["tensors"] = {
            tensor.name: GGUFQuantizedTensor(torch.from_numpy(tensor.data), quant_type=tensor.tensor_type)
            for tensor in reader.tensors
        }
        parsed_parameters["weight_mapping"] = get_gguf_converters(model_type)

    if len(reader_keys) > 0:
        logger.info(f"Some keys of the GGUF file were not considered: {reader_keys}")

    return parsed_parameters
