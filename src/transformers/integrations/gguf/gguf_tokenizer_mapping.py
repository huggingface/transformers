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

    architecture, section, config = get_gguf_tokenizer("model.gguf")
    backend, _ = convert_gguf_tokenizer(architecture, section)

To support a new file, add an entry to `GGUF_TOKENIZER_KINDS`, or a builder if it needs more.
"""

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

# `LLAMA_TOKEN_TYPE_CONTROL`, as llama.cpp numbers it.
_CONTROL_TOKEN = 3


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
    # Not at module scope: the reader needs torch, and `configuration_utils` reaches this file for
    # `GGUF_TOKENIZER_MAPPING` alone, on a path that has to import without it.
    from .reader import read_gguf_metadata

    # Only these two are needed in full; the reader leaves every other array as a count.
    metadata, _ = read_gguf_metadata(gguf_path, ("tokenizer.ggml.tokens", "tokenizer.ggml.merges"))
    architecture = metadata["general.architecture"]
    sections = {
        section: {
            name: metadata[f"tokenizer.{key}"] for key, name in renames.items() if f"tokenizer.{key}" in metadata
        }
        for section, renames in GGUF_TOKENIZER_MAPPING.items()
    }
    tokenizer, tokenizer_config = sections["tokenizer"], sections["tokenizer_config"]

    for name, key in _SPECIAL_TOKENS.items():
        token_id = metadata.get(f"tokenizer.ggml.{key}")
        tokenizer_config[name] = tokenizer["tokens"][token_id] if token_id is not None else None
    return architecture, tokenizer, tokenizer_config


def convert_gguf_tokenizer(architecture: str, tokenizer_dict: dict) -> tuple[Tokenizer, dict]:
    kind = tokenizer_dict.get("tokenizer_type")
    tokenizer = select_tokenizer_builder(architecture, kind)(tokenizer_dict)
    tokenizer = with_control_tokens(tokenizer, tokenizer_dict, byte_level=kind == "gpt2")
    tokenizer = with_split(tokenizer, tokenizer_dict)
    return tokenizer.backend_tokenizer, {}


def sentencepiece_tokenizer(section, tokenizer_class=LlamaTokenizer, ranks=None):
    """The vocabulary as the file spells it, and merges from the file or recovered from it."""
    vocab = {token: index for index, token in enumerate(section["tokens"])}
    return tokenizer_class(vocab=vocab, merges=get_merges(section, ranks))


def phi3_tokenizer(section):
    """Sentencepiece, but the prefix space comes from a normalizer, not from `Metaspace`.

    There is no `Phi3Tokenizer` to hand this to. Phi-3's shape lives in the `tokenizer.json` of its
    repo, which a GGUF does not carry, so we rebuild it.

    It only shows on text starting with a space: `"  hi"` becomes `\u2581\u2581\u2581hi` here and `\u2581\u2581hi`
    with `Metaspace`. Nothing in a GGUF says which a model wants, so this is picked by architecture.
    """
    tokenizer = sentencepiece_tokenizer(section)
    tokenizer.backend_tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Prepend(prepend="\u2581"), normalizers.Replace(pattern=" ", content="\u2581")]
    )
    tokenizer.backend_tokenizer.pre_tokenizer = None
    return tokenizer


def gemma_tokenizer(section):
    """Sentencepiece, with two fixes for Gemma files.

    1. The file writes a run of spaces as `"  "`, but `GemmaTokenizer` looks up `"\u2581\u2581"`. We rename
       those tokens, before deriving merges, so both halves of a merge match.
    2. Gemma's scores are mostly one placeholder, so they cannot order the merges. We order them by
       the token's position in the file instead.
    """
    respelled = dict(
        section,
        tokens=["\u2581" * len(t) if " " in t and not t.strip() else t for t in section["tokens"]],
    )
    positions = {token: -index for index, token in enumerate(respelled["tokens"])}
    return sentencepiece_tokenizer(respelled, GemmaTokenizer, ranks=positions)


def byte_level_tokenizer(section):
    """Byte-level BPE, with the vocabulary keyed the way the file's own merges spell it.
    """
    spelling = bytes_to_unicode()
    alphabet = set(spelling.values())
    vocab = {
        token if set(token) <= alphabet else "".join(spelling[byte] for byte in token.encode("utf-8")): index
        for index, token in enumerate(section["tokens"])
    }
    return GPT2Tokenizer(vocab=vocab, merges=get_merges(section))


def unigram_tokenizer(section):
    """A unigram vocabulary is `(token, score)` pairs, and has no merges at all."""
    return T5Tokenizer(vocab=list(zip(section["tokens"], section["scores"])))


def get_merges(section, ranks=None):
    """The merges the file gives, or the ones its vocabulary implies.

    """
    if "merges" in section:
        return [tuple(merge.split(" ")) for merge in section["merges"]]

    tokens = section["tokens"]
    logger.info("No merges in the file; deriving them from the vocabulary.")
    if ranks is None:
        ranks = dict(zip(tokens, section["scores"]))
    return generate_merges({token: index for index, token in enumerate(tokens)}, ranks)


def with_control_tokens(tokenizer, section, byte_level=False):
    """Tell the tokenizer which tokens are special, like `<|endoftext|>`.

    A GGUF marks them with a type per token rather than listing them, so only we can. Without this
    they are cut into pieces: `"<|begin_of_text|>hello<|eot_id|>"` comes out as 15 tokens, not 3.

    On a byte-level file we skip any token not spelled byte-level. That one is a byte sequence
    llama.cpp mislabelled, and adding it would leave the vocabulary larger than the model.
    """
    spelled = set(bytes_to_unicode().values()) if byte_level else None
    control = [
        AddedToken(token, normalized=False, special=True)
        for token, token_type in zip(section["tokens"], section.get("token_type") or ())
        if token_type == _CONTROL_TOKEN and (spelled is None or set(token) <= spelled)
    ]
    if control:
        tokenizer.add_special_tokens({"additional_special_tokens": control}, replace_extra_special_tokens=False)
    return tokenizer


def select_tokenizer_builder(architecture: str, tokenizer_type: str | None):
    """The builder for the tokenizer a file describes."""
    if tokenizer_type == "llama" and architecture in GGUF_SENTENCEPIECE_BUILDERS:
        return GGUF_SENTENCEPIECE_BUILDERS[architecture]
    if tokenizer_type in GGUF_TOKENIZER_KINDS:
        return GGUF_TOKENIZER_KINDS[tokenizer_type]
    raise ValueError(
        f"Cannot build a tokenizer from a GGUF file stating tokenizer {tokenizer_type!r}. "
        f"Supported: {sorted(GGUF_TOKENIZER_KINDS)}."
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


def with_split(tokenizer, section):
    """Cut text into the chunks this vocabulary's merges were learned on."""
    split = GGUF_PRE_TOKENIZER_SPLITS.get(section.get("pre_tokenizer_type"))
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

GGUF_TOKENIZER_KINDS = {
    "gpt2": byte_level_tokenizer,
    "llama": sentencepiece_tokenizer,
    "t5": unigram_tokenizer,
    "gemma4": partial(sentencepiece_tokenizer, tokenizer_class=GemmaTokenizer),
}
