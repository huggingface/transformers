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
with extra methods being exposed
"""

from array import array

from ..utils import logging


logger = logging.get_logger(__name__)


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
        # NOTE: rope.dimension_count==head_dim only suitable for llama/mistral
        "rope.dimension_count": "head_dim",
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
        # NOTE: rope.dimension_count==head_dim only suitable for llama/mistral
        "rope.dimension_count": "head_dim",
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
    "qwen2_moe": {
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
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
    },
    "gpt_oss": {
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
        "expert_count": "num_local_experts",
        "expert_used_count": "num_experts_per_tok",
        "sliding_window": "sliding_window",
    },
    "lfm2": {
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
        "shortconv.l_cache": "conv_L_cache",
    },
    "qwen3": {
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
    "qwen3_moe": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.key_length": "head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
    },
    "falcon": {
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
    "phi3": {
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
    "bloom": {
        "block_count": "n_layer",
        "embedding_length": "hidden_size",
        "attention.head_count": "n_head",
        "vocab_size": "vocab_size",
        "attention.layer_norm_epsilon": "layer_norm_epsilon",
    },
    "t5": {
        "context_length": "n_positions",
        "block_count": "num_layers",
        "feed_forward_length": "d_ff",
        "embedding_length": "d_model",
        "attention.key_length": "d_kv",
        "attention.head_count": "num_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_epsilon": "layer_norm_epsilon",
        "attention.relative_buckets_count": "relative_attention_num_buckets",
        "decoder_start_token_id": "decoder_start_token_id",
        "vocab_size": "vocab_size",
    },
    "stablelm": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_epsilon": "layer_norm_eps",
        "vocab_size": "vocab_size",
    },
    "gpt2": {
        "block_count": "n_layer",
        "context_length": "n_ctx",
        "embedding_length": "n_embd",
        "feed_forward_length": "feed_forward_length",
        "attention.head_count": "n_head",
        "attention.layer_norm_epsilon": "layer_norm_epsilon",
    },
    "starcoder2": {
        "block_count": "num_hidden_layers",
        "context_length": "max_position_embeddings",
        "embedding_length": "hidden_size",
        "feed_forward_length": "intermediate_size",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_epsilon": "norm_epsilon",
    },
    "mamba": {
        "vocab_size": "vocab_size",
        "context_length": "max_position_embeddings",
        "embedding_length": "hidden_size",
        "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
        "block_count": "num_hidden_layers",
        "ssm.conv_kernel": "conv_kernel",
        "ssm.state_size": "state_size",
        "ssm.time_step_rank": "time_step_rank",
        "ssm.inner_size": "intermediate_size",
    },
    "nemotron": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "norm_eps",
        "vocab_size": "vocab_size",
    },
    "gemma2": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        # NOTE: Gemma2 has key_length==value_length==head_dim
        # See: https://github.com/ggerganov/llama.cpp/blob/2e2f8f093cd4fb6bbb87ba84f6b9684fa082f3fa/convert_hf_to_gguf.py#L3293-L3294
        "attention.key_length": "head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.sliding_window": "sliding_window",
        "vocab_size": "vocab_size",
    },
    "gemma3": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        # NOTE: Gemma3 has key_length==value_length==head_dim
        # See: https://github.com/ggml-org/llama.cpp/blob/fe5b78c89670b2f37ecb216306bed3e677b49d9f/convert_hf_to_gguf.py#L3495-L3496
        "attention.key_length": "head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.sliding_window": "sliding_window",
        "vocab_size": "vocab_size",
    },
    "gemma4": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": None,
        # Gemma4 has mixed attention: sliding (head_dim=256) and full (global_head_dim=512)
        # GGUF stores the full attention head dimension in attention.key_length
        # We want to preserve the default head_dim=256 and only set global_head_dim from GGUF
        "attention.key_length": "global_head_dim",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.sliding_window": "sliding_window",
        "vocab_size": "vocab_size",
    },
    "umt5": {
        "context_length": "n_positions",
        "block_count": "num_layers",
        "feed_forward_length": "d_ff",
        "embedding_length": "d_model",
        "attention.key_length": "d_kv",
        "attention.head_count": "num_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_epsilon": "layer_norm_epsilon",
        "attention.relative_buckets_count": "relative_attention_num_buckets",
        "decoder_start_token_id": "decoder_start_token_id",
        "vocab_size": "vocab_size",
    },
    "deci": {
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
    "minimax_m2": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": "rotary_dim",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.key_length": "head_dim",
        "attention.value_length": None,
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "expert_count": "num_local_experts",
        "expert_used_count": "num_experts_per_tok",
        "expert_feed_forward_length": None,
        "vocab_size": "vocab_size",
        "expert_gating_func": "scoring_func",
    },
}

GGUF_CONFIG_DEFAULTS_MAPPING = {
    "qwen3_moe": {
        # NOTE: Qwen3MoeConfig defaults to false but llama.cpp needs this to be true.
        # See: https://github.com/ggml-org/llama.cpp/blob/17f7f4baad8b3a716ee139da7bb56ae984e8c0fa/src/models/qwen3moe.cpp#L85-L96
        #      (the parameter right after LLM_FFN_SILU corresponds to norm_topk_prob)
        "norm_topk_prob": True,
    },
    "minimax_m2": {
        # MiniMax-M2 uses routing bias (e_score_correction_bias) for MoE expert selection,
        # but this is not stored in GGUF metadata. Set it as default so the model weights
        # (which include e_score_correction_bias tensors) are loaded correctly.
        "use_routing_bias": True,
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
    elif data_type == 7:
        _value = bool(_value[0])
    elif data_type == 8:
        _value = array("B", list(_value)).tobytes().decode()
    elif data_type == 9:
        _value = _gguf_parse_value(_value, array_data_type)
    return _value
