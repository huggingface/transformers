# Copyright 2026 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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

"""Shared fixtures for Mistral tekken tokenizer tests."""

import base64
import json
from pathlib import Path


NUM_SPECIAL_TOKENS = 20

FAKE_TEKKEN_SPECIAL_TOKENS = [
    {"rank": 0, "token_str": "<unk>", "is_control": True},
    {"rank": 1, "token_str": "<s>", "is_control": True},
    {"rank": 2, "token_str": "</s>", "is_control": True},
    {"rank": 3, "token_str": "[INST]", "is_control": True},
    {"rank": 4, "token_str": "[/INST]", "is_control": True},
    {"rank": 5, "token_str": "[AVAILABLE_TOOLS]", "is_control": True},
    {"rank": 6, "token_str": "[/AVAILABLE_TOOLS]", "is_control": True},
    {"rank": 7, "token_str": "[TOOL_RESULTS]", "is_control": True},
    {"rank": 8, "token_str": "[/TOOL_RESULTS]", "is_control": True},
    {"rank": 9, "token_str": "[TOOL_CALLS]", "is_control": True},
    {"rank": 10, "token_str": "[IMG]", "is_control": True},
    {"rank": 11, "token_str": "<pad>", "is_control": True},
    {"rank": 12, "token_str": "[IMG_BREAK]", "is_control": True},
    {"rank": 13, "token_str": "[IMG_END]", "is_control": True},
    {"rank": 14, "token_str": "[PREFIX]", "is_control": True},
    {"rank": 15, "token_str": "[MIDDLE]", "is_control": True},
    {"rank": 16, "token_str": "[SUFFIX]", "is_control": True},
    {"rank": 17, "token_str": "[SYSTEM_PROMPT]", "is_control": True},
    {"rank": 18, "token_str": "[/SYSTEM_PROMPT]", "is_control": True},
    {"rank": 19, "token_str": "[TOOL_CONTENT]", "is_control": True},
]

FAKE_TEKKEN_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# 256 byte-level BPE tokens + 20 special tokens = full single-byte coverage.
FULL_BYTE_VOCAB = 256 + NUM_SPECIAL_TOKENS


def build_fake_tekken_dict(
    vocab_size: int = FULL_BYTE_VOCAB,
    image_config: dict | None = None,
    num_special_tokens: int | None = None,
    mixed_token_str: bool = False,
) -> dict:
    """Build a minimal tekken.json dict for testing.

    Args:
        vocab_size: Total vocabulary size (special + BPE tokens).
        image_config: Optional image config dict added under ``"image"`` key.
        num_special_tokens: Override for ``default_num_special_tokens`` in config.
            When greater than ``len(FAKE_TEKKEN_SPECIAL_TOKENS)``, the loader will
            generate filler ``<SPECIAL_n>`` tokens, exercising the filler-token path.
            Defaults to ``NUM_SPECIAL_TOKENS`` (no fillers).
        mixed_token_str: When ``True``, printable ASCII bytes (0x20–0x7E) carry a real
            ``token_str`` equal to their decoded character; all other bytes carry ``null``.

    Returns:
        A dict representing a minimal tekken.json structure.
    """
    effective_num_special = num_special_tokens if num_special_tokens is not None else NUM_SPECIAL_TOKENS
    num_bpe = vocab_size - effective_num_special

    vocab_list: list[dict] = []
    for rank in range(num_bpe):
        raw_byte = bytes([rank % 256])
        byte_val = rank % 256
        tok_str = chr(byte_val) if (mixed_token_str and 0x20 <= byte_val <= 0x7E) else None
        vocab_list.append(
            {
                "rank": rank,
                "token_bytes": base64.b64encode(raw_byte).decode("ascii"),
                "token_str": tok_str,
            }
        )

    tekken_data: dict = {
        "vocab": vocab_list,
        "special_tokens": FAKE_TEKKEN_SPECIAL_TOKENS,
        "config": {
            "pattern": FAKE_TEKKEN_PATTERN,
            "num_vocab_tokens": num_bpe,
            "default_vocab_size": vocab_size,
            "default_num_special_tokens": effective_num_special,
            "version": "v3",
        },
        "version": 1,
        "type": "tekken",
    }

    if image_config is not None:
        tekken_data["image"] = image_config

    return tekken_data


def write_fake_tekken_json(
    directory,
    vocab_size: int = FULL_BYTE_VOCAB,
    image_config: dict | None = None,
    num_special_tokens: int | None = None,
    mixed_token_str: bool = False,
) -> Path:
    """Write a minimal tekken.json into ``directory`` and return its path.

    Args:
        directory: Directory to write tekken.json into.
        vocab_size: Total vocabulary size (special + BPE tokens).
        image_config: Optional image config dict added under ``"image"`` key.
        num_special_tokens: Override for ``default_num_special_tokens`` in config.
        mixed_token_str: When ``True``, printable ASCII bytes carry a real ``token_str``.

    Returns:
        Path to the written ``tekken.json`` file.
    """
    tekken_data = build_fake_tekken_dict(
        vocab_size=vocab_size,
        image_config=image_config,
        num_special_tokens=num_special_tokens,
        mixed_token_str=mixed_token_str,
    )
    output_path = Path(directory) / "tekken.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tekken_data, f, ensure_ascii=False)
    return output_path
