# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Rebuilding a model config from a GGUF file's metadata.

A GGUF repo ships no `config.json`: the metadata is the config, under llama.cpp's key names. One
function per architecture turns those into a transformers config dict, so a file loads on its own.

Everything the file states about the model's shape is set even where it matches the config class's
default, since those defaults are one checkpoint's values and would silently apply to another.
"""


def _qwen35_config(metadata: dict, tensor_names: tuple[str, ...], prefix: str = "qwen35") -> dict:
    """Qwen3.5: hybrid GatedDeltaNet + full attention, with an mrope and an MTP block."""
    key = lambda name: metadata[f"{prefix}.{name}"]  # noqa: E731
    head_dim = key("attention.key_length")
    value_heads = key("ssm.time_step_rank")
    mtp_layers = metadata.get(f"{prefix}.nextn_predict_layers", 0)

    config = {
        "model_type": "qwen3_5_text",
        # Nothing in the file states this, but callers that pick a class from `config.architectures`
        # would otherwise need a GGUF special case.
        "architectures": ["Qwen3_5ForCausalLM"],
        "max_position_embeddings": key("context_length"),
        "hidden_size": key("embedding_length"),
        "num_attention_heads": key("attention.head_count"),
        "num_key_value_heads": key("attention.head_count_kv"),
        "head_dim": head_dim,
        "rms_norm_eps": key("attention.layer_norm_rms_epsilon"),
        "full_attention_interval": key("full_attention_interval"),
        "linear_conv_kernel_dim": key("ssm.conv_kernel"),
        "linear_key_head_dim": key("ssm.state_size"),
        "linear_num_key_heads": key("ssm.group_count"),
        "linear_num_value_heads": value_heads,
        # the file counts the multi-token-prediction block as a layer; the decoder stack does not
        "num_hidden_layers": key("block_count") - mtp_layers,
        # the value dimension is stated whole, where transformers wants it per head
        "linear_value_head_dim": key("ssm.inner_size") // value_heads,
        "rope_parameters": {
            "rope_theta": key("rope.freq_base"),
            # one section per axis, padded to four; transformers keeps the ones it uses
            "mrope_section": [section for section in key("rope.dimension_sections") if section],
            "mrope_interleaved": True,
            # a rotary width in dimensions, where transformers takes a fraction of the head
            "partial_rotary_factor": key("rope.dimension_count") / head_dim,
        },
        # `read_gguf_metadata` leaves the vocabulary as its length
        "vocab_size": metadata["tokenizer.ggml.tokens"],
        # llama.cpp writes the output projection only when it is not the embedding matrix
        "tie_word_embeddings": "output.weight" not in tensor_names,
        # absent ids stay `None`, as they are on a config by default
        "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "pad_token_id": metadata.get("tokenizer.ggml.padding_token_id"),
    }
    # An all-expert file has no dense FFN to describe, so llama.cpp omits its width.
    if (feed_forward := metadata.get(f"{prefix}.feed_forward_length")) is not None:
        config["intermediate_size"] = feed_forward
    return config


def _qwen35moe_config(metadata: dict, tensor_names: tuple[str, ...]) -> dict:
    config = _qwen35_config(metadata, tensor_names, prefix="qwen35moe")
    key = lambda name: metadata[f"qwen35moe.{name}"]  # noqa: E731
    return config | {
        "model_type": "qwen3_5_moe_text",
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "moe_intermediate_size": key("expert_feed_forward_length"),
        "shared_expert_intermediate_size": key("expert_shared_feed_forward_length"),
        "num_experts": key("expert_count"),
        "num_experts_per_tok": key("expert_used_count"),
    }


def _lfm2_config(metadata: dict, tensor_names: tuple[str, ...]) -> dict:
    """LFM2: short convolutions, with full attention on some layers."""
    key = lambda name: metadata[f"lfm2.{name}"]  # noqa: E731
    # one count per layer, zero on the convolution ones: the file says which layer is which this way
    kv_heads_per_layer = key("attention.head_count_kv")

    return {
        "model_type": "lfm2",
        "architectures": ["Lfm2ForCausalLM"],
        "max_position_embeddings": key("context_length"),
        "hidden_size": key("embedding_length"),
        "intermediate_size": key("feed_forward_length"),
        # the width above is the final one, which the config would otherwise rescale
        "block_auto_adjust_ff_dim": False,
        "num_hidden_layers": key("block_count"),
        "num_attention_heads": key("attention.head_count"),
        "num_key_value_heads": max(kv_heads_per_layer),
        "layer_types": ["full_attention" if heads else "conv" for heads in kv_heads_per_layer],
        "norm_eps": key("attention.layer_norm_rms_epsilon"),
        "conv_L_cache": key("shortconv.l_cache"),
        "conv_bias": any(name.startswith("blk.") and name.endswith("shortconv.in_proj.bias") for name in tensor_names),
        "rope_parameters": {"rope_type": "default", "rope_theta": key("rope.freq_base")},
        # `read_gguf_metadata` leaves the vocabulary as its length
        "vocab_size": metadata["tokenizer.ggml.tokens"],
        # llama.cpp writes the output projection only when it is not the embedding matrix
        "tie_word_embeddings": "output.weight" not in tensor_names,
        # absent ids stay `None`, as they are on a config by default
        "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "pad_token_id": metadata.get("tokenizer.ggml.padding_token_id"),
    }


def _glm4moe_config(metadata: dict, tensor_names: tuple[str, ...]) -> dict:
    """GLM-4.5 / 4.6: sigmoid-routed MoE with shared experts, a few dense layers first, and an MTP block."""
    key = lambda name: metadata[f"glm4moe.{name}"]  # noqa: E731
    head_dim = key("attention.key_length")
    mtp_layers = metadata.get("glm4moe.nextn_predict_layers", 0)
    # The checkpoint stops on three ids, `<|endoftext|>`, `<|user|>` and `<|observation|>`, which llama.cpp
    # writes as eos, eot and eom. With eos alone, a chat would run on past the end of the assistant's turn.
    stop_ids = [metadata.get(f"tokenizer.ggml.{name}_token_id") for name in ("eos", "eot", "eom")]
    stop_ids = list(dict.fromkeys(token_id for token_id in stop_ids if token_id is not None))

    return {
        "model_type": "glm4_moe",
        # Nothing in the file states this, but callers that pick a class from `config.architectures`
        # would otherwise need a GGUF special case.
        "architectures": ["Glm4MoeForCausalLM"],
        "max_position_embeddings": key("context_length"),
        "hidden_size": key("embedding_length"),
        "intermediate_size": key("feed_forward_length"),
        "num_attention_heads": key("attention.head_count"),
        "num_key_value_heads": key("attention.head_count_kv"),
        # GLM-4.5 heads are not `hidden_size // num_attention_heads` wide, so this has to be stated
        "head_dim": head_dim,
        "rms_norm_eps": key("attention.layer_norm_rms_epsilon"),
        # the file counts the multi-token-prediction block as a layer; the decoder stack does not
        "num_hidden_layers": key("block_count") - mtp_layers,
        "num_nextn_predict_layers": mtp_layers,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": key("rope.freq_base"),
            # a rotary width in dimensions, where transformers takes a fraction of the head
            "partial_rotary_factor": key("rope.dimension_count") / head_dim,
        },
        "moe_intermediate_size": key("expert_feed_forward_length"),
        "n_routed_experts": key("expert_count"),
        "num_experts_per_tok": key("expert_used_count"),
        "n_shared_experts": key("expert_shared_count"),
        "first_k_dense_replace": key("leading_dense_block_count"),
        "routed_scaling_factor": key("expert_weights_scale"),
        "norm_topk_prob": key("expert_weights_norm"),
        # llama.cpp writes the groups only when the checkpoint's config has them; one group is no grouping
        "n_group": metadata.get("glm4moe.expert_group_count", 1),
        "topk_group": metadata.get("glm4moe.expert_group_used_count", 1),
        # neither has a key of its own; the file says so by carrying the tensors or not
        "attention_bias": "blk.0.attn_q.bias" in tensor_names,
        "use_qk_norm": "blk.0.attn_q_norm.weight" in tensor_names,
        # `read_gguf_metadata` leaves the vocabulary as its length
        "vocab_size": metadata["tokenizer.ggml.tokens"],
        # llama.cpp writes the output projection only when it is not the embedding matrix
        "tie_word_embeddings": "output.weight" not in tensor_names,
        # absent ids stay `None`, as they are on a config by default
        "eos_token_id": (stop_ids[0] if len(stop_ids) == 1 else stop_ids) or None,
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "pad_token_id": metadata.get("tokenizer.ggml.padding_token_id"),
    }


GGUF_CONFIG_ARCHS = {
    "qwen35": _qwen35_config,
    "qwen35moe": _qwen35moe_config,
    "lfm2": _lfm2_config,
    "glm4moe": _glm4moe_config,
}


def get_gguf_config(metadata: dict, tensor_names: tuple[str, ...]) -> dict:
    """The transformers config dict for a file with this metadata and these tensors.

    Raises for an architecture with no entry above; callers with a fallback check `GGUF_CONFIG_ARCHS`.
    """
    architecture = metadata["general.architecture"]
    if architecture not in GGUF_CONFIG_ARCHS:
        raise ValueError(
            f"Cannot rebuild a config from a GGUF file of architecture {architecture!r}. "
            f"Supported: {sorted(GGUF_CONFIG_ARCHS)}."
        )
    return GGUF_CONFIG_ARCHS[architecture](metadata, tensor_names)
