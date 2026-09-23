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
"""Building a tokenizer from a GGUF file's metadata.

A GGUF has no `tokenizer.json`. It stores the vocabulary, the merges and the special token ids as
metadata, under llama.cpp's key names. The tokenizer it describes is almost always one transformers
already has, so we read those out and hand them to that class, which brings its own normalizer,
pre-tokenizer and decoder:

    architecture, tokenizer_dict, config = get_gguf_tokenizer("model.gguf")
    backend, _ = convert_gguf_tokenizer(architecture, tokenizer_dict)

To support a new file, add an entry to `GGUF_TOKENIZER_TYPES`, or a builder if it needs more.
"""

import re
from functools import partial

from tokenizers import AddedToken, Regex, Tokenizer, normalizers, pre_tokenizers

from ...convert_slow_tokenizer import bytes_to_unicode
from ...models.gemma.tokenization_gemma import GemmaTokenizer
from ...models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from ...models.llama.tokenization_llama import LlamaTokenizer
from ...models.qwen2.tokenization_qwen2 import PRETOKENIZE_REGEX as _QWEN2_SPLIT
from ...models.qwen3_5.tokenization_qwen3_5 import PRETOKENIZE_REGEX as _QWEN35_SPLIT
from ...models.t5.tokenization_t5 import T5Tokenizer
from ...tokenization_utils_base import generate_merges
from ...utils import logging


logger = logging.get_logger(__name__)


GGUF_TOKENIZER_MAPPING = {
    "tokenizer": {
        "ggml.model": "tokenizer_type",
        "ggml.pre": "pre_tokenizer_type",
        "ggml.tokens": "tokens",
        "ggml.scores": "scores",
        "ggml.token_type": "token_type",
        "ggml.merges": "merges",
        "ggml.precompiled_charsmap": "precompiled_charsmap",
        "ggml.bos_token_id": "bos_token_id",
        "ggml.eos_token_id": "eos_token_id",
        "ggml.unknown_token_id": "unk_token_id",
        "ggml.padding_token_id": "pad_token_id",
        "ggml.add_space_prefix": "add_prefix_space",
    },
    "tokenizer_config": {
        "chat_template": "chat_template",
    },
}


_SPECIAL_TOKENS = {
    "bos_token": "bos_token_id",
    "eos_token": "eos_token_id",
    "unk_token": "unknown_token_id",
    "pad_token": "padding_token_id",
}


def get_gguf_tokenizer(gguf_path: str) -> tuple[str, dict, dict]:
    """`(architecture, tokenizer_dict, tokenizer_config)` for the tokenizer this file describes."""
    from .reader import read_gguf_metadata

    # Only these two are needed in full; the reader leaves every other array as a count.
    metadata, _ = read_gguf_metadata(gguf_path, ("tokenizer.ggml.tokens", "tokenizer.ggml.merges"))
    architecture = metadata["general.architecture"]
    grouped = {
        group: {name: metadata[f"tokenizer.{key}"] for key, name in renames.items() if f"tokenizer.{key}" in metadata}
        for group, renames in GGUF_TOKENIZER_MAPPING.items()
    }
    tokenizer, tokenizer_config = grouped["tokenizer"], grouped["tokenizer_config"]

    for name, key in _SPECIAL_TOKENS.items():
        token_id = metadata.get(f"tokenizer.ggml.{key}")
        tokenizer_config[name] = tokenizer["tokens"][token_id] if token_id is not None else None
    return architecture, tokenizer, tokenizer_config


def convert_gguf_tokenizer(architecture: str, tokenizer_dict: dict) -> tuple[Tokenizer, dict]:
    tokenizer_type = tokenizer_dict.get("tokenizer_type")
    tokenizer = select_tokenizer_builder(architecture, tokenizer_type)(tokenizer_dict)
    tokenizer = add_gguf_special_tokens(tokenizer, tokenizer_dict)
    tokenizer = set_split_regex(tokenizer, tokenizer_dict)
    return tokenizer.backend_tokenizer, {}


def sentencepiece_tokenizer(tokenizer_dict, tokenizer_class=LlamaTokenizer, vocab_scores=None):
    """The vocabulary as the file spells it, and merges from the file or recovered from it."""
    vocab = {token: index for index, token in enumerate(tokenizer_dict["tokens"])}
    return tokenizer_class(vocab=vocab, merges=get_merges(tokenizer_dict, vocab_scores))


def phi3_tokenizer(tokenizer_dict):
    """Sentencepiece, but the prefix space comes from a normalizer, not from `Metaspace`.

    There is no `Phi3Tokenizer` to hand this to. Phi-3's shape lives in the `tokenizer.json` of its
    repo, which a GGUF does not carry, so we rebuild it.

    It only shows on text starting with a space: `"  hi"` becomes `\u2581\u2581\u2581hi` here and `\u2581\u2581hi`
    with `Metaspace`. Nothing in a GGUF says which a model wants, so this is picked by architecture.
    """
    tokenizer = sentencepiece_tokenizer(tokenizer_dict)
    tokenizer.backend_tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Prepend(prepend="\u2581"), normalizers.Replace(pattern=" ", content="\u2581")]
    )
    tokenizer.backend_tokenizer.pre_tokenizer = None
    return tokenizer


def gemma_tokenizer(tokenizer_dict):
    """Sentencepiece, with two fixes Gemma files need.

    1. A token of two or more spaces is stored as real spaces ("  "), while `GemmaTokenizer` looks
       it up as "\u2581\u2581". Rename those tokens first, so the merges built from them match.
    2. The file lists no merges, so they get rebuilt from the vocabulary, cut into pairs and sorted
       best-first. Elsewhere the score each token carries is the right sort key, but Gemma's scores
       do not follow its merge order -- the order the tokens appear in the file does.
    """
    respelled = dict(
        tokenizer_dict,
        tokens=["\u2581" * len(t) if " " in t and not t.strip() else t for t in tokenizer_dict["tokens"]],
    )
    positions = {token: -index for index, token in enumerate(respelled["tokens"])}
    return sentencepiece_tokenizer(respelled, GemmaTokenizer, vocab_scores=positions)


def byte_level_tokenizer(tokenizer_dict):
    """Byte-level BPE"""
    spelling = bytes_to_unicode()
    alphabet = set(spelling.values())
    vocab = {
        token if set(token) <= alphabet else "".join(spelling[byte] for byte in token.encode("utf-8")): index
        for index, token in enumerate(tokenizer_dict["tokens"])
    }
    return GPT2Tokenizer(vocab=vocab, merges=get_merges(tokenizer_dict))


def unigram_tokenizer(tokenizer_dict):
    """Unigram: a vocabulary of `(token, score)` pairs, and no merges.

    T5 writes a blank in a prompt as `<extra_id_0>`..`<extra_id_99>`, but the file calls those
    `[PAD32000]`... We rename the first 100 back, which is how many T5 has by default.
    """
    count = 100
    placeholder = re.compile(r"^\[PAD\d+\]$")
    tokens = list(tokenizer_dict["tokens"])
    blanks = [index for index, token in enumerate(tokens) if placeholder.match(token)]
    for offset, index in enumerate(blanks[:count]):
        tokens[index] = f"<extra_id_{count - 1 - offset}>"
    return T5Tokenizer(vocab=list(zip(tokens, tokenizer_dict["scores"])))


def get_merges(tokenizer_dict, vocab_scores=None):
    """The merges the file gives, or the ones its vocabulary implies."""
    if "merges" in tokenizer_dict:
        return [tuple(merge.split(" ")) for merge in tokenizer_dict["merges"]]

    tokens = tokenizer_dict["tokens"]
    logger.info("No merges in the file; deriving them from the vocabulary.")
    if vocab_scores is None:
        vocab_scores = dict(zip(tokens, tokenizer_dict["scores"]))
    return generate_merges({token: index for index, token in enumerate(tokens)}, vocab_scores)


def add_gguf_special_tokens(tokenizer, tokenizer_dict):
    """Add the tokens the file marks as control, like `<|endoftext|>`, as special tokens.

    Without this they get cut into pieces instead of staying whole. In a byte-level vocabulary every
    real token is written in the byte alphabet, so one marked control that uses any other character
    is a llama.cpp mislabel -- adding it would push the vocabulary past the size the model expects,
    so it is skipped.
    """
    byte_level = tokenizer_dict.get("tokenizer_type") == "gpt2"
    spelled = set(bytes_to_unicode().values()) if byte_level else None
    control = [
        AddedToken(token, normalized=False, special=True)
        for token, token_type in zip(tokenizer_dict["tokens"], tokenizer_dict.get("token_type") or ())
        if token_type == 3 and (spelled is None or set(token) <= spelled)  # 3 is CONTROL
    ]
    if control:
        tokenizer.add_special_tokens({"additional_special_tokens": control}, replace_extra_special_tokens=False)
    return tokenizer


def select_tokenizer_builder(architecture: str, tokenizer_type: str | None):
    """The builder for the tokenizer a file describes, from `tokenizer.ggml.model`."""
    if tokenizer_type == "llama" and architecture in GGUF_SENTENCEPIECE_BUILDERS:
        return GGUF_SENTENCEPIECE_BUILDERS[architecture]
    if tokenizer_type in GGUF_TOKENIZER_TYPES:
        return GGUF_TOKENIZER_TYPES[tokenizer_type]
    raise ValueError(
        f"Cannot build a tokenizer from a GGUF file stating tokenizer {tokenizer_type!r}. "
        f"Supported: {sorted(GGUF_TOKENIZER_TYPES)}."
    )


_LLAMA3_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


_TEKKEN_SPLIT = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


_GPT4O_SPLIT = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

GGUF_PRE_TOKENIZER_SPLITS = {
    "llama3": _LLAMA3_SPLIT,
    "llama-v3": _LLAMA3_SPLIT,
    "llama-bpe": _LLAMA3_SPLIT,
    "falcon3": _LLAMA3_SPLIT,
    "falcon-h1": _LLAMA3_SPLIT,
    "pixtral": _LLAMA3_SPLIT,
    "midm-2.0": _LLAMA3_SPLIT,
    "lfm2": _LLAMA3_SPLIT,
    "jina-v5-nano": _LLAMA3_SPLIT,
    "dbrx": _LLAMA3_SPLIT,
    "smaug-bpe": _LLAMA3_SPLIT,
    "tekken": _TEKKEN_SPLIT,
    "qwen2": _QWEN2_SPLIT,
    "deepseek-r1-qwen": _QWEN2_SPLIT,
    "kormo": _QWEN2_SPLIT,
    "f2llmv2": _QWEN2_SPLIT,
    "megrez": _QWEN2_SPLIT,
    "qwen35": _QWEN35_SPLIT,
    "gpt-4o": _GPT4O_SPLIT,
    "llama4": _GPT4O_SPLIT,
    "kanana2": _GPT4O_SPLIT,
    "talkie": _GPT4O_SPLIT,
    "minimax-m2": _GPT4O_SPLIT,
}


def set_split_regex(tokenizer, tokenizer_dict):
    """Split text the way this vocabulary's merges were learned."""
    split = GGUF_PRE_TOKENIZER_SPLITS.get(tokenizer_dict.get("pre_tokenizer_type"))
    if split is not None:
        tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(split), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
    return tokenizer


GGUF_SENTENCEPIECE_BUILDERS = {
    "gemma": gemma_tokenizer,
    "gemma2": gemma_tokenizer,
    "gemma3": gemma_tokenizer,
    "phi3": phi3_tokenizer,
}

GGUF_TOKENIZER_TYPES = {
    "gpt2": byte_level_tokenizer,
    "llama": sentencepiece_tokenizer,
    "t5": unigram_tokenizer,
    "gemma4": partial(sentencepiece_tokenizer, tokenizer_class=GemmaTokenizer),
}
