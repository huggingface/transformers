# coding=utf-8
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
"""
Integration with GGML / The file is copied and adapted from https://github.com/99991/pygguf
with extra methods beings exposed
"""

from array import array

import numpy as np
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE

from .. import AddedToken
from ..convert_slow_tokenizer import LlamaConverter, Qwen2Converter
from ..utils import logging
from ..utils.logging import tqdm


logger = logging.get_logger(__name__)


# Listed here: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
GGML_TYPES = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q8_0": 8,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "IQ2_XXS": 16,
}

# The Blocksizes are reported in bytes
# Check out: https://github.com/ggerganov/llama.cpp/blob/8a56075b07a8b571bf95a912ffdce4c928c2b414/gguf-py/gguf/constants.py#L801
GGML_BLOCK_SIZES = {
    "Q8_0": 2 + 32,  # Q8_0 uses a blocksize of 32 (int8 tensors) + 2 bytes allocated for the scales
    "Q4_K": 144,
    # Q4_0 uses a blocksize of 32 but the 4-bit tensors are packed into 8-bit tensors + 2 bytes for the scales
    "Q4_0": 2 + 16,
    "Q6_K": 210,
    # See: https://github.com/99991/pygguf/commit/a417edbfc029a1bc270f984a694f9128c5afa8b9
    "Q2_K": 256 // 16 + 256 // 4 + 2 + 2,
    "Q3_K": 256 // 8 + 256 // 4 + 12 + 2,
    "Q5_K": 2 + 2 + 12 + 256 // 8 + 256 // 2,
    "IQ2_XXS": 2 + 256 // 8 * 2,
}

# Listed here: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
DATA_TYPES = {
    "uint32": 4,
    "int32": 5,
    "float32": 6,
    "bool": 7,
    "string": 8,
    "array": 9,
    "uint64": 10,
}

GGUF_TENSOR_MAPPING = {
    "llama": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    },
    "mistral": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    },
    "qwen2": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    },
}


GGUF_CONFIG_MAPPING = {
    "general": {
        "architecture": "model_type",
        "name": "_model_name_or_path",
    },
    "llama": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    },
    "mistral": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    },
    "qwen2": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    },
    "tokenizer": {
        "ggml.bos_token_id": "bos_token_id",
        "ggml.eos_token_id": "eos_token_id",
        "ggml.unknown_token_id": "unk_token_id",
        "ggml.padding_token_id": "pad_token_id",
    },
}

GGUF_TOKENIZER_MAPPING = {
    "tokenizer": {
        "ggml.model": "tokenizer_type",
        "ggml.tokens": "tokens",
        "ggml.scores": "scores",
        "ggml.token_type": "token_type",
        "ggml.merges": "merges",
        "ggml.bos_token_id": "bos_token_id",
        "ggml.eos_token_id": "eos_token_id",
        "ggml.unknown_token_id": "unk_token_id",
        "ggml.padding_token_id": "pad_token_id",
        "ggml.add_space_prefix": "add_prefix_space",
    },
    "tokenizer_config": {
        "chat_template": "chat_template",
        "ggml.model": "model_type",
        "ggml.bos_token_id": "bos_token_id",
        "ggml.eos_token_id": "eos_token_id",
        "ggml.unknown_token_id": "unk_token_id",
        "ggml.padding_token_id": "pad_token_id",
    },
}


def _gguf_parse_value(_value, data_type):
    if not isinstance(data_type, list):
        data_type = [data_type]
    if len(data_type) == 1:
        data_type = data_type[0]
        array_data_type = None
    else:
        if data_type[0] != 9:
            raise ValueError("Received multiple types, therefore expected the first type to indicate an array.")
        data_type, array_data_type = data_type

    if data_type in [0, 1, 2, 3, 4, 5, 10, 11]:
        _value = int(_value[0])
    elif data_type in [6, 12]:
        _value = float(_value[0])
    elif data_type in [7]:
        _value = bool(_value[0])
    elif data_type in [8]:
        _value = array("B", list(_value)).tobytes().decode()
    elif data_type in [9]:
        _value = _gguf_parse_value(_value, array_data_type)
    return _value


def dequantize_q4_k(data, n_bytes: int):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1929
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L116
    block_size = GGML_BLOCK_SIZES["Q4_K"]
    num_blocks = n_bytes // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    # Casting to float32 because float16 is very slow on CPU
    scale_factors = data_f16[:, 0].reshape(num_blocks, 1, 1).astype(np.float32)
    scale_offsets = data_f16[:, 1].reshape(num_blocks, 1, 1).astype(np.float32)
    qs1 = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qs2 = data_u8[:, 16:].reshape(num_blocks, 4, 32)

    # Dequantize scales and offsets (6 bits and 4 + 2 bits)
    factors = scale_factors * np.concatenate(
        [qs1[:, 0:4] & 0b111111, (qs1[:, 8:] & 15) | ((qs1[:, 0:4] >> 6) << 4)], axis=1
    )
    offsets = scale_offsets * np.concatenate(
        [qs1[:, 4:8] & 0b111111, (qs1[:, 8:] >> 4) | ((qs1[:, 4:8] >> 6) << 4)], axis=1
    )

    # Interleave low and high quantized bits
    qs2 = np.stack([qs2 & 0xF, qs2 >> 4], axis=2).reshape(num_blocks, 8, 32)
    # Dequantize final weights using scales and offsets
    return factors * qs2 - offsets


def dequantize_q4_0(data, n_bytes: int):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1086
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L11
    block_size = GGML_BLOCK_SIZES["Q4_0"]
    num_blocks = n_bytes // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    # The scales are stored on the first 2 bytes and the rest corresponds to the quants
    scales = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    # scales = np.nan_to_num(scales)
    # the rest of the bytes corresponds to the quants - we discard the first two bytes
    quants = data_u8[:, 2:]

    ql = (quants[:, :] & 0xF).astype(np.int8) - 8
    qr = (quants[:, :] >> 4).astype(np.int8) - 8

    # Use hstack
    quants = np.hstack([ql, qr])

    return (scales * quants).astype(np.float32)


def dequantize_q6_k(data, n_bytes: int):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2275
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L152
    block_size = GGML_BLOCK_SIZES["Q6_K"]
    num_blocks = n_bytes // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)
    data_i8 = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, block_size)

    scales = data_f16[:, -1].reshape(num_blocks, 1).astype(np.float32)

    # TODO use uint8 and cast later?
    ql = data_u8[:, :128].astype(np.int16)
    qh = data_u8[:, 128:192].astype(np.int16)
    sc = data_i8[:, 192:208, np.newaxis].astype(np.float32)

    # Unpack bits, subtraction requires signed data type
    q1 = (ql[:, :32] & 0xF) | (((qh[:, :32] >> 0) & 3) << 4) - 32
    q2 = (ql[:, 32:64] & 0xF) | (((qh[:, :32] >> 2) & 3) << 4) - 32
    q3 = (ql[:, :32] >> 4) | (((qh[:, :32] >> 4) & 3) << 4) - 32
    q4 = (ql[:, 32:64] >> 4) | (((qh[:, :32] >> 6) & 3) << 4) - 32
    q5 = (ql[:, 64:96] & 0xF) | (((qh[:, 32:] >> 0) & 3) << 4) - 32
    q6 = (ql[:, 96:128] & 0xF) | (((qh[:, 32:] >> 2) & 3) << 4) - 32
    q7 = (ql[:, 64:96] >> 4) | (((qh[:, 32:] >> 4) & 3) << 4) - 32
    q8 = (ql[:, 96:128] >> 4) | (((qh[:, 32:] >> 6) & 3) << 4) - 32

    # Dequantize
    return scales * np.concatenate(
        [
            sc[:, 0] * q1[:, :16],
            sc[:, 1] * q1[:, 16:],
            sc[:, 2] * q2[:, :16],
            sc[:, 3] * q2[:, 16:],
            sc[:, 4] * q3[:, :16],
            sc[:, 5] * q3[:, 16:],
            sc[:, 6] * q4[:, :16],
            sc[:, 7] * q4[:, 16:],
            sc[:, 8] * q5[:, :16],
            sc[:, 9] * q5[:, 16:],
            sc[:, 10] * q6[:, :16],
            sc[:, 11] * q6[:, 16:],
            sc[:, 12] * q7[:, :16],
            sc[:, 13] * q7[:, 16:],
            sc[:, 14] * q8[:, :16],
            sc[:, 15] * q8[:, 16:],
        ],
        axis=1,
    )


def dequantize_q8_0(data, n_bytes: int):
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43
    block_size = GGML_BLOCK_SIZES["Q8_0"]
    num_blocks = n_bytes // block_size

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 16)[:, :1].astype(np.float32)
    qs = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, 2 + 32)[:, 2:]

    return scales * qs


def dequantize_q2_k(data, n_bytes: int):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1547
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L74
    num_blocks = n_bytes // GGML_BLOCK_SIZES["Q2_K"]

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, GGML_BLOCK_SIZES["Q2_K"] // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, GGML_BLOCK_SIZES["Q2_K"])

    dmin = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    d = data_f16[:, -2].reshape(num_blocks, 1, 1).astype(np.float32)
    scales = data_u8[:, :16].reshape(num_blocks, 16, 1)
    qs = data_u8[:, 16:80].reshape(num_blocks, 64)

    tmp = np.stack(
        [
            qs[:, 00:16] >> 0,
            qs[:, 16:32] >> 0,
            qs[:, 00:16] >> 2,
            qs[:, 16:32] >> 2,
            qs[:, 00:16] >> 4,
            qs[:, 16:32] >> 4,
            qs[:, 00:16] >> 6,
            qs[:, 16:32] >> 6,
            qs[:, 32:48] >> 0,
            qs[:, 48:64] >> 0,
            qs[:, 32:48] >> 2,
            qs[:, 48:64] >> 2,
            qs[:, 32:48] >> 4,
            qs[:, 48:64] >> 4,
            qs[:, 32:48] >> 6,
            qs[:, 48:64] >> 6,
        ],
        axis=1,
    )

    return d * (scales & 15) * (tmp & 3) - dmin * (scales >> 4)


def dequantize_q3_k(data, n_bytes: int):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1723C32-L1723C42
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L95
    num_blocks = n_bytes // GGML_BLOCK_SIZES["Q3_K"]

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, GGML_BLOCK_SIZES["Q3_K"] // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, GGML_BLOCK_SIZES["Q3_K"])

    d = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    bits = np.unpackbits(data_u8[:, :32].reshape(num_blocks, 32, 1), axis=-1, bitorder="little")
    bits = 4 ^ (bits << 2)
    qs = data_u8[:, 32 : 32 + 64].astype(np.int16)
    a, b, c = data_u8[:, 96 : 96 + 12].reshape(num_blocks, 3, 4).transpose(1, 0, 2)
    scales = np.zeros((num_blocks, 4, 4), dtype=np.uint8)
    scales[:, 0] = (a & 15) | ((c & 3) << 4)
    scales[:, 1] = (b & 15) | (((c >> 2) & 3) << 4)
    scales[:, 2] = (a >> 4) | (((c >> 4) & 3) << 4)
    scales[:, 3] = (b >> 4) | ((c >> 6) << 4)
    scales = scales.reshape(num_blocks, 16, 1).astype(np.int16)

    return (
        d
        * (scales - 32)
        * np.stack(
            [
                (((qs[:, 00:16] >> 0) & 3) - bits[:, :16, 0]),
                (((qs[:, 16:32] >> 0) & 3) - bits[:, 16:, 0]),
                (((qs[:, 00:16] >> 2) & 3) - bits[:, :16, 1]),
                (((qs[:, 16:32] >> 2) & 3) - bits[:, 16:, 1]),
                (((qs[:, 00:16] >> 4) & 3) - bits[:, :16, 2]),
                (((qs[:, 16:32] >> 4) & 3) - bits[:, 16:, 2]),
                (((qs[:, 00:16] >> 6) & 3) - bits[:, :16, 3]),
                (((qs[:, 16:32] >> 6) & 3) - bits[:, 16:, 3]),
                (((qs[:, 32:48] >> 0) & 3) - bits[:, :16, 4]),
                (((qs[:, 48:64] >> 0) & 3) - bits[:, 16:, 4]),
                (((qs[:, 32:48] >> 2) & 3) - bits[:, :16, 5]),
                (((qs[:, 48:64] >> 2) & 3) - bits[:, 16:, 5]),
                (((qs[:, 32:48] >> 4) & 3) - bits[:, :16, 6]),
                (((qs[:, 48:64] >> 4) & 3) - bits[:, 16:, 6]),
                (((qs[:, 32:48] >> 6) & 3) - bits[:, :16, 7]),
                (((qs[:, 48:64] >> 6) & 3) - bits[:, 16:, 7]),
            ],
            axis=1,
        )
    )


def dequantize_q5_k(data, n_bytes: int):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2129
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L138
    num_blocks = n_bytes // GGML_BLOCK_SIZES["Q5_K"]

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, GGML_BLOCK_SIZES["Q5_K"] // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, GGML_BLOCK_SIZES["Q5_K"])

    d = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    dmin = data_f16[:, 1].reshape(num_blocks, 1).astype(np.float32)
    scales = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qh = data_u8[:, 16 : 16 + 32].reshape(num_blocks, 32, 1)
    qs = data_u8[:, 48 : 48 + 128].reshape(num_blocks, 4, 32)

    bits = np.unpackbits(qh, axis=-1, bitorder="little")

    qs_hi_4 = qs >> 4
    qs_lo_4 = qs & 15

    scales_lo_6 = scales[:, :8] & 63
    scales_hi_6 = scales[:, :8] >> 6
    scales_lo_4 = scales[:, 8:] & 15
    scales_hi_4 = scales[:, 8:] >> 4

    m1 = dmin * scales_lo_6[:, 4]
    m2 = dmin * scales_lo_6[:, 5]
    m3 = dmin * scales_lo_6[:, 6]
    m4 = dmin * scales_lo_6[:, 7]
    m5 = dmin * (scales_hi_4[:, 0] | (scales_hi_6[:, 4] << 4))
    m6 = dmin * (scales_hi_4[:, 1] | (scales_hi_6[:, 5] << 4))
    m7 = dmin * (scales_hi_4[:, 2] | (scales_hi_6[:, 6] << 4))
    m8 = dmin * (scales_hi_4[:, 3] | (scales_hi_6[:, 7] << 4))

    d1 = d * scales_lo_6[:, 0]
    d2 = d * scales_lo_6[:, 1]
    d3 = d * scales_lo_6[:, 2]
    d4 = d * scales_lo_6[:, 3]
    d5 = d * (scales_lo_4[:, 0] | (scales_hi_6[:, 0] << 4))
    d6 = d * (scales_lo_4[:, 1] | (scales_hi_6[:, 1] << 4))
    d7 = d * (scales_lo_4[:, 2] | (scales_hi_6[:, 2] << 4))
    d8 = d * (scales_lo_4[:, 3] | (scales_hi_6[:, 3] << 4))

    return np.concatenate(
        [
            d1 * (qs_lo_4[:, 0] + (bits[:, :, 0] << 4)) - m1,
            d2 * (qs_hi_4[:, 0] + (bits[:, :, 1] << 4)) - m2,
            d3 * (qs_lo_4[:, 1] + (bits[:, :, 2] << 4)) - m3,
            d4 * (qs_hi_4[:, 1] + (bits[:, :, 3] << 4)) - m4,
            d5 * (qs_lo_4[:, 2] + (bits[:, :, 4] << 4)) - m5,
            d6 * (qs_hi_4[:, 2] + (bits[:, :, 5] << 4)) - m6,
            d7 * (qs_lo_4[:, 3] + (bits[:, :, 6] << 4)) - m7,
            d8 * (qs_hi_4[:, 3] + (bits[:, :, 7] << 4)) - m8,
        ],
        axis=1,
    )


def dequantize_iq2_xxs(data, n_bytes):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/3f5a4bbe59285c0f679b376f6259187d5514ff9c/src/ggml-quants.c#L3311
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/3f5a4bbe59285c0f679b376f6259187d5514ff9c/src/ggml-common.h#L314-L321
    def _dequantize_iq2xxs_column(qs_block, d):
        """
        The qs matrix could be splitted into 8 sub_blocks (4 x int16 bytes each block) for each row:
        | Block_11 | Block_12 | Block_13 | Block_14 | Block_15 | Block_16 | Block_17 | Block_18 |
        | Block_21 | Block_22 | Block_23 | Block_24 | Block_25 | Block_26 | Block_27 | Block_28 |
        ...
        | Block_n1 | Block_n2 | Block_n3 | Block_n4 | Block_n5 | Block_n6 | Block_n7 | Block_n8 |

        This function process n rows at a time (Block_11 to Block_n1, Block_12 to Block_n2 etc).
        """
        from .ggml_utils import IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS

        aux32 = np.frombuffer(qs_block, dtype=np.uint32).reshape(num_blocks, 2)
        aux8 = np.frombuffer(qs_block, dtype=np.uint8).reshape(num_blocks, 8)

        l = np.arange(4)
        db = d * (0.5 + (aux32[:, [1]] >> 28)) * 0.25

        grid = np.frombuffer(np.ascontiguousarray(IQ2XXS_GRID[aux8[:, l]]), dtype=np.uint8).reshape(num_blocks, 32)
        signs = KSIGNS_IQ2XS[(aux32[:, [1]] >> 7 * l) & 127]

        y = db * grid * np.where(signs.repeat(8, axis=1) & np.tile(KMASK_IQ2XS, 4), -1, 1)
        return y

    num_blocks = n_bytes // GGML_BLOCK_SIZES["IQ2_XXS"]

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, GGML_BLOCK_SIZES["IQ2_XXS"] // 2)
    data_i16 = np.frombuffer(data, dtype=np.int16).reshape(num_blocks, GGML_BLOCK_SIZES["IQ2_XXS"] // 2)

    d = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    qs = data_i16[:, 1:].reshape(num_blocks, 32)

    y = [_dequantize_iq2xxs_column(np.ascontiguousarray(qs[:, 4 * i : 4 * (i + 1)]), d) for i in range(8)]
    y = np.concatenate(y, axis=1)
    return y


def load_dequant_gguf_tensor(shape, ggml_type, data, n_bytes):
    if ggml_type == GGML_TYPES["F32"]:
        values = data
    elif ggml_type == GGML_TYPES["F16"]:
        values = data
    elif ggml_type == GGML_TYPES["Q8_0"]:
        values = dequantize_q8_0(data, n_bytes)
    elif ggml_type == GGML_TYPES["Q4_0"]:
        values = dequantize_q4_0(data, n_bytes)
    elif ggml_type == GGML_TYPES["Q4_K"]:
        values = dequantize_q4_k(data, n_bytes)
    elif ggml_type == GGML_TYPES["Q6_K"]:
        values = dequantize_q6_k(data, n_bytes)
    elif ggml_type == GGML_TYPES["Q2_K"]:
        values = dequantize_q2_k(data, n_bytes)
    elif ggml_type == GGML_TYPES["Q3_K"]:
        values = dequantize_q3_k(data, n_bytes)
    elif ggml_type == GGML_TYPES["Q5_K"]:
        values = dequantize_q5_k(data, n_bytes)
    elif ggml_type == GGML_TYPES["IQ2_XXS"]:
        values = dequantize_iq2_xxs(data, n_bytes)
    else:
        raise NotImplementedError(
            f"ggml_type {ggml_type} not implemented - please raise an issue on huggingface transformers: https://github.com/huggingface/transformers/issues/new/choose"
        )

    return values.reshape(shape[::-1])


class GGUFTokenizerSkeleton:
    def __init__(self, dict_):
        for k, v in dict_.items():
            setattr(self, k, v)

        if not hasattr(self, "merges"):
            if not hasattr(self, "tokens") or not hasattr(self, "scores"):
                raise ValueError(
                    "tokens and scores need to be passed for a LLaMa tokenizer without merges to be instantiated."
                )
            tokens = self.tokens
            scores = self.scores
            vocab = {t: scores[i] for i, t in enumerate(tokens)}

            logger.warning("Merges were not in checkpoint, building merges on the fly.")
            merges = []
            for merge, piece_score in tqdm(vocab.items()):
                local = []
                for index in range(1, len(merge)):
                    piece_l, piece_r = merge[:index], merge[index:]
                    if piece_l in tokens and piece_r in tokens:
                        local.append((piece_l, piece_r, piece_score))
                local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]), reverse=True)
                merges.extend(local)
            merges = sorted(merges, key=lambda val: val[2], reverse=True)
            merges = [(val[0], val[1]) for val in merges]
            self.merges = merges
        else:
            self.merges = [tuple(merge.split(" ")) for merge in self.merges]
            if not hasattr(self, "scores"):
                self.scores = [None for _ in range(len(self.tokens))]

        if not hasattr(self, "added_tokens"):
            self.added_tokens = []

        if not hasattr(self, "unk_token_id"):
            self.unk_token_id = None

        # Llama2 uses the field `unknown_token_id`
        if hasattr(self, "unknown_token_id") and self.unk_token_id is None:
            self.unk_token_id = self.unknown_token_id


class GGUFLlamaConverter(LlamaConverter):
    def __init__(self, tokenizer_dict):
        self.proto = GGUFTokenizerSkeleton(tokenizer_dict)
        self.original_tokenizer = self.proto
        self.additional_kwargs = {}
        self.is_llama_3_tokenizer = getattr(self.proto, "tokenizer_type", "llama") != "llama"

    def vocab(self, proto):
        return list(zip(proto.tokens, proto.scores))

    def merges(self, proto):
        return proto.merges

    def tokenizer(self, proto):
        vocab_scores = self.vocab(self.proto)
        merges = self.merges(self.proto)
        bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}

        unk_token = proto.tokens[proto.unk_token_id] if proto.unk_token_id is not None else None
        bos_token = proto.tokens[proto.bos_token_id] if getattr(proto, "bos_token_id", None) is not None else None
        eos_token = proto.tokens[proto.bos_token_id] if getattr(proto, "eos_token_id", None) is not None else None

        tokenizer = Tokenizer(BPE(bpe_vocab, merges, unk_token=unk_token, fuse_unk=True, byte_fallback=True))

        special_tokens = []

        if not hasattr(self.proto, "token_type"):
            if unk_token is not None:
                special_tokens.append(AddedToken(unk_token, normalized=False, special=True))

            if bos_token is not None:
                special_tokens.append(AddedToken(bos_token, normalized=False, special=True))

            if eos_token is not None:
                special_tokens.append(AddedToken(eos_token, normalized=False, special=True))
        else:
            # 3 stands for special tokens
            special_tokens_idx = np.where(np.array(self.proto.token_type) == 3)[0]

            for idx in special_tokens_idx:
                special_tokens.append(AddedToken(self.proto.tokens[idx], normalized=False, special=True))

        if len(special_tokens) != 0:
            tokenizer.add_special_tokens(special_tokens)

        if len(self.proto.added_tokens) != 0:
            tokenizer.add_tokens(
                [AddedToken(added_token, normalized=False, special=False) for added_token in self.proto.added_tokens]
            )

        self.additional_kwargs["unk_token"] = unk_token
        self.additional_kwargs["eos_token"] = bos_token
        self.additional_kwargs["bos_token"] = eos_token

        if self.is_llama_3_tokenizer:
            self.additional_kwargs["add_prefix_space"] = None
            self.additional_kwargs["clean_up_tokenization_spaces"] = True

            self.additional_kwargs["legacy"] = False
            self.original_tokenizer.legacy = False

        return tokenizer

    def decoder(self, replacement, add_prefix_space):
        sequence = [
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Replace("▁", " "),
        ]

        if self.is_llama_3_tokenizer:
            sequence += [decoders.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=True)]

        if add_prefix_space:
            sequence += [decoders.Strip(content=" ", left=1)]
        return decoders.Sequence(sequence)

    def converted(self):
        # Copied partly from converted method in SpmConverter class
        tokenizer = self.tokenizer(self.proto)

        # Tokenizer assemble
        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = True
        if hasattr(self.original_tokenizer, "add_prefix_space"):
            add_prefix_space = self.original_tokenizer.add_prefix_space

        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        post_processor = self.post_processor()
        if post_processor:
            tokenizer.post_processor = post_processor

        # HACK: patch the llama-3 tokenizer to use the correspinding pre-tokenizer
        # and normalizer
        if self.is_llama_3_tokenizer:
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=False, trim_offsets=False, use_regex=True
            )
            # This is tricky as the additional kwargs are passed after legacy is force-set in LlamaTokenizer's
            # init.
            tokenizer.normalizer = normalizers.Sequence([])

        return tokenizer


class GGUFQwen2Converter(Qwen2Converter):
    def __init__(self, tokenizer_dict):
        self.original_tokenizer = GGUFTokenizerSkeleton(tokenizer_dict)
        self.additional_kwargs = {}

    def converted(self) -> Tokenizer:
        vocab = {word: i for i, word in enumerate(self.original_tokenizer.tokens)}
        merges = self.original_tokenizer.merges
        tokenizer = super().converted(vocab, merges)

        tokenizer.add_special_tokens(
            [
                AddedToken("<|endoftext|>", normalized=False, special=True),
                AddedToken("<|im_start|>", normalized=False, special=True),
                AddedToken("<|im_end|>", normalized=False, special=True),
            ]
        )
        return tokenizer


GGUF_TO_FAST_CONVERTERS = {
    "llama": GGUFLlamaConverter,
    "qwen2": GGUFQwen2Converter,
}


def convert_gguf_tokenizer(architecture, tokenizer_dict) -> Tokenizer:
    """
    Utilities to convert a slow tokenizer instance in a fast tokenizer instance.

    Args:
        architecture (`str`): The model architecture derived from gguf file.
        transformer_tokenizer ([`~tokenization_utils_base.PreTrainedTokenizer`]):
            Instance of a slow tokenizer to convert in the backend tokenizer for
            [`~tokenization_utils_base.PreTrainedTokenizerFast`].

    Return:
        A instance of [`~tokenizers.Tokenizer`] to be used as the backend tokenizer of a
        [`~tokenization_utils_base.PreTrainedTokenizerFast`]
    """
    tokenizer_class_name = architecture
    converter = GGUF_TO_FAST_CONVERTERS[tokenizer_class_name](tokenizer_dict)
    fast_tokenizer = converter.converted()
    return fast_tokenizer, converter.additional_kwargs
