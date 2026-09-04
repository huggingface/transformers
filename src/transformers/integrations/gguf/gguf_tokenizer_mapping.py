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
"""Building a tokenizer from a GGUF file's metadata.

A GGUF repo ships no `tokenizer.json`: the vocabulary, merges and special token ids are metadata keys.
The renaming table is llama.cpp's convention, shared with `integrations.ggml.GGUF_TOKENIZER_MAPPING`;
the only per-architecture fact is which converter reads the result.
"""

from .reader import read_gguf_metadata


# `general.architecture` -> the `model_type` whose `GGUF_TO_FAST_CONVERTERS` entry reads this vocabulary.
GGUF_TOKENIZER_ARCHS = {
    "qwen35": "qwen3_5_text",
    # The MoE writes the same vocabulary; only the layer stack differs.
    "qwen35moe": "qwen3_5_moe_text",
}

# The metadata arrays a tokenizer needs in full; everything else the reader can leave as a count.
_VOCABULARY_KEYS = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges")


def get_gguf_tokenizer(gguf_path: str) -> tuple[str, dict, dict]:
    """`(model_type, tokenizer_dict, tokenizer_config)` for the tokenizer this file describes.

    Raises for an architecture with no entry above; callers with a fallback check
    `GGUF_TOKENIZER_ARCHS` first.
    """
    from ..ggml import GGUF_TOKENIZER_MAPPING

    metadata, _ = read_gguf_metadata(gguf_path, _VOCABULARY_KEYS)
    architecture = metadata["general.architecture"]
    if architecture not in GGUF_TOKENIZER_ARCHS:
        raise ValueError(
            f"Cannot build a tokenizer from a GGUF file of architecture {architecture!r}. "
            f"Supported: {sorted(GGUF_TOKENIZER_ARCHS)}."
        )
    sections = {
        section: {
            name: metadata[f"tokenizer.{key}"] for key, name in renames.items() if f"tokenizer.{key}" in metadata
        }
        for section, renames in GGUF_TOKENIZER_MAPPING.items()
    }
    tokenizer, tokenizer_config = sections["tokenizer"], sections["tokenizer_config"]
    # A GGUF names its special tokens by id, the tokenizer wants the strings. Undeclared ones are stated
    # as `None` rather than left out, or the sentencepiece `<s>`/`</s>` fallback invents two tokens.
    for name in ("bos_token", "eos_token", "pad_token", "unk_token"):
        token_id = tokenizer_config.get(f"{name}_id")
        tokenizer_config[name] = tokenizer["tokens"][token_id] if token_id is not None else None
    return GGUF_TOKENIZER_ARCHS[architecture], tokenizer, tokenizer_config
