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
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE

from .. import AddedToken
from ..convert_slow_tokenizer import LlamaConverter


# Listed here: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
GGML_TYPES = {
    "F32": 0,
    "Q4_0": 2,
    "Q8_0": 8,
    "Q4_K": 12,
    "Q6_K": 14,
}

# The Blocksizes are reported in bytes
# Check out: https://github.com/ggerganov/llama.cpp/blob/8a56075b07a8b571bf95a912ffdce4c928c2b414/gguf-py/gguf/constants.py#L801
GGML_BLOCK_SIZES = {
    "Q8_0": 2 + 32,  # Q8_0 uses a blocksize of 32 (int8 tensors) + 2 bytes allocated for the scales
    "Q4_K": 144,
    "Q4_0": 2
    + 16,  # Q4_0 uses a blocksize of 32 but the 4-bit tensors are packed into 8-bit tensors + 2 bytes for the scales
    "Q6_K": 210,
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
    "tokenizer": {
        "ggml.model": "model_type",
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
            raise ValueError("Received multiple types, but therefore expect the first type to indicate an array.")
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


def dequantize_q4_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1929
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L116
    block_size = GGML_BLOCK_SIZES["Q4_K"]
    num_blocks = len(data) // block_size

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


def dequantize_q4_0(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1086
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L11
    block_size = GGML_BLOCK_SIZES["Q4_0"]
    num_blocks = len(data) // block_size

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


def dequantize_q6_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2275
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L152
    block_size = GGML_BLOCK_SIZES["Q6_K"]
    num_blocks = len(data) // block_size

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


def dequantize_q8_0(data):
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43
    block_size = GGML_BLOCK_SIZES["Q8_0"]
    num_blocks = len(data) // block_size

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 16)[:, :1].astype(np.float32)
    qs = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, 2 + 32)[:, 2:]

    return scales * qs


def load_dequant_gguf_tensor(shape, ggml_type, data):
    if ggml_type == GGML_TYPES["F32"]:
        values = data
    elif ggml_type == GGML_TYPES["Q8_0"]:
        values = dequantize_q8_0(data)
    elif ggml_type == GGML_TYPES["Q4_0"]:
        values = dequantize_q4_0(data)
    elif ggml_type == GGML_TYPES["Q4_K"]:
        values = dequantize_q4_k(data)
    elif ggml_type == GGML_TYPES["Q6_K"]:
        values = dequantize_q6_k(data)
    else:
        raise NotImplementedError(
            f"ggml_type {ggml_type} not implemented - please raise an issue on huggingface transformers: https://github.com/huggingface/transformers/issues/new/choose"
        )

    return values.reshape(shape[::-1])


class GGUFTokenizerSkeleton:
    def __init__(self, dict_):
        for k, v in dict_.items():
            setattr(self, k, v)


class GGUFLlamaConverter(LlamaConverter):
    def __init__(self, tokenizer_dict):
        self.proto = GGUFTokenizerSkeleton(tokenizer_dict)
        self.original_tokenizer = self.proto

    def vocab(self, proto):
        return list(zip(proto.tokens, proto.scores))

    def merges(self, proto):
        return [tuple(merge.split(" ")) for merge in proto.merges]

    def tokenizer(self, proto):
        vocab_scores = self.vocab(self.proto)
        merges = self.merges(self.proto)
        bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}
        tokenizer = Tokenizer(
            BPE(bpe_vocab, merges, unk_token=proto.tokens[proto.unk_token_id], fuse_unk=True, byte_fallback=True)
        )
        tokenizer.add_special_tokens(
            [
                AddedToken("<unk>", normalized=False, special=True),
                AddedToken("<s>", normalized=False, special=True),
                AddedToken("</s>", normalized=False, special=True),
            ]
        )
        return tokenizer

    def decoder(self, replacement, add_prefix_space):
        sequence = [
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Replace("▁", " "),
        ]
        print(add_prefix_space)
        add_prefix_space = False
        if add_prefix_space:
            sequence += [decoders.Strip(content=" ", left=1)]
        return decoders.Sequence(sequence)


GGUF_TO_FAST_CONVERTERS = {
    "llama": GGUFLlamaConverter,
}


def convert_gguf_tokenizer(tokenizer_dict) -> Tokenizer:
    """
    Utilities to convert a slow tokenizer instance in a fast tokenizer instance.

    Args:
        transformer_tokenizer ([`~tokenization_utils_base.PreTrainedTokenizer`]):
            Instance of a slow tokenizer to convert in the backend tokenizer for
            [`~tokenization_utils_base.PreTrainedTokenizerFast`].

    Return:
        A instance of [`~tokenizers.Tokenizer`] to be used as the backend tokenizer of a
        [`~tokenization_utils_base.PreTrainedTokenizerFast`]
    """
    tokenizer_class_name = tokenizer_dict["tokenizer_type"]
    converter_class = GGUF_TO_FAST_CONVERTERS[tokenizer_class_name]
    return converter_class(tokenizer_dict).converted()
