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

A GGUF repo ships no `tokenizer.json` either: the vocabulary, the merges and the special token ids
are metadata keys, and the tokenizer is assembled from them the way the config is in
`gguf_config_mapping.py`.

What the keys mean is llama.cpp's convention, not an architecture's, so the renaming table is shared
with the legacy reader (`integrations.ggml.GGUF_TOKENIZER_MAPPING`) and the only per-architecture
fact is which converter reads the result -- one name per entry below.

This mirrors `gguf_config_mapping.py`: an architecture whose tokenizer this can build appears here,
and the callers that have a fallback check `GGUF_TOKENIZER_ARCHS` before asking.
"""

from .reader import read_gguf_metadata


# `general.architecture` -> the `model_type` whose entry in `GGUF_TO_FAST_CONVERTERS` reads this
# file's vocabulary. Qwen3.5 writes a gpt2-style BPE, as every other Qwen does.
GGUF_TOKENIZER_ARCHS = {
    "qwen35": "qwen3_5_text",
}

# The metadata arrays a tokenizer needs in full. Everything else the reader can leave as a count --
# and these two are most of the file's metadata, so it is worth naming them rather than reading all.
_VOCABULARY_KEYS = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges")


def get_gguf_tokenizer(gguf_path: str) -> tuple[str, dict, dict]:
    """`(model_type, tokenizer_dict, tokenizer_config)` for the tokenizer this file describes.

    `tokenizer_dict` is what a `GGUF*Converter` builds a fast tokenizer from, and `tokenizer_config`
    the keyword arguments that go to the tokenizer around it.

    Raises for an architecture with no entry above; callers that have a fallback check
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
    # A GGUF names its special tokens by id; the tokenizer wants the strings, and falls back to a
    # sentencepiece `<s>`/`</s>` pair when it is given neither -- which, for a repo shipping no
    # `tokenizer_config.json`, means two invented tokens appended to the vocabulary. The ones the file
    # does not declare are stated as `None` rather than left out, to keep that fallback out of the way.
    for name in ("bos_token", "eos_token", "pad_token", "unk_token"):
        token_id = tokenizer_config.get(f"{name}_id")
        tokenizer_config[name] = tokenizer["tokens"][token_id] if token_id is not None else None
    return GGUF_TOKENIZER_ARCHS[architecture], tokenizer, tokenizer_config
