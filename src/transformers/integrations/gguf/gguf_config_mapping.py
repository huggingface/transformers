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

A GGUF repo ships no `config.json`: the file's metadata is the config, under llama.cpp's key names. One
function per architecture turns those keys into a transformers config dict, so a file can be loaded on
its own — `from_pretrained(repo, gguf_file=...)` — with nothing borrowed from a reference checkpoint.

Most entries are a rename. The ones that are not are where llama.cpp states the same fact differently,
and are written as the expression that converts it.

Everything the file states about the model's shape is set, even where it currently matches the config
class's default: those defaults are one checkpoint's values, so a key left out would silently take them
from a different model — `Qwen3_5TextConfig` defaults to a hidden size of 4096, which no 4B file has.
What is left out is what the config derives or defaults on its own, like the rope type.

This mirrors `gguf_conversion_mapping.py`: one entry per `general.architecture`, and an architecture is
supported only when it appears in both.
"""


def _qwen35_config(metadata: dict, tensor_names: tuple[str, ...]) -> dict:
    """Qwen3.5: hybrid GatedDeltaNet + full attention, with an mrope and an MTP block."""
    key = lambda name: metadata[f"qwen35.{name}"]  # noqa: E731
    head_dim = key("attention.key_length")
    value_heads = key("ssm.time_step_rank")
    # Optional: a file converted without the MTP block does not carry the key at all — unsloth's
    # Qwen3.5-4B-Q4_K_M is one — and then every block in the file is a decoder layer.
    mtp_layers = metadata.get("qwen35.nextn_predict_layers", 0)

    return {
        "model_type": "qwen3_5_text",
        "max_position_embeddings": key("context_length"),
        "hidden_size": key("embedding_length"),
        "intermediate_size": key("feed_forward_length"),
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
        "mtp_num_hidden_layers": mtp_layers,
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
        # `read_gguf_metadata` leaves the vocabulary as its length, which is all a config wants from it
        "vocab_size": metadata["tokenizer.ggml.tokens"],
        # llama.cpp writes the output projection only when it is not the embedding matrix
        "tie_word_embeddings": "output.weight" not in tensor_names,
        # the ids the file itself declares; absent ones stay `None`, as they are on a config by default
        "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "pad_token_id": metadata.get("tokenizer.ggml.padding_token_id"),
    }


GGUF_CONFIG_ARCHS = {
    "qwen35": _qwen35_config,
}


def get_gguf_config(metadata: dict, tensor_names: tuple[str, ...]) -> dict:
    """The transformers config dict for a file with this metadata and these tensors.

    Raises for an architecture with no entry above; callers that have a fallback check
    `GGUF_CONFIG_ARCHS` first.
    """
    architecture = metadata["general.architecture"]
    if architecture not in GGUF_CONFIG_ARCHS:
        raise ValueError(
            f"Cannot rebuild a config from a GGUF file of architecture {architecture!r}. "
            f"Supported: {sorted(GGUF_CONFIG_ARCHS)}."
        )
    return GGUF_CONFIG_ARCHS[architecture](metadata, tensor_names)
