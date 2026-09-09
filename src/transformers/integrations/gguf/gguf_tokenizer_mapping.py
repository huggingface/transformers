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

A GGUF repo ships no `tokenizer.json`: the vocabulary, merges and special token ids are metadata
keys under llama.cpp's own names. `GGUF_TOKENIZER_MAPPING` renames them, and `GGUF_TOKENIZER_KINDS`
maps what the file says its tokenizer is to the class that assembles one.
"""

from collections.abc import Sequence

import numpy as np
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram

from ...convert_slow_tokenizer import (
    GemmaConverter,
    GPT2Converter,
    LlamaConverter,
    T5Converter,
    bytes_to_unicode,
)
from ...utils import logging
from ...utils.logging import tqdm
from .reader import read_gguf_metadata


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


# What a tokenizer is constructed with, and the id llama.cpp states it as.
_SPECIAL_TOKENS = {
    "bos_token": "bos_token_id",
    "eos_token": "eos_token_id",
    "unk_token": "unknown_token_id",
    "pad_token": "padding_token_id",
}


def get_gguf_tokenizer(gguf_path: str) -> tuple[str, dict, dict]:
    """`(architecture, tokenizer_dict, tokenizer_config)` for the tokenizer this file describes."""
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
    """`(fast tokenizer, kwargs to construct it with)` from what `get_gguf_tokenizer` read."""
    converter = _select_converter(
        architecture, tokenizer_dict.get("tokenizer_type"), tokenizer_dict.get("tokens", ())
    )(tokenizer_dict)
    fast_tokenizer = converter.converted()
    return fast_tokenizer, converter.additional_kwargs


def _is_gemma_vocabulary(tokens: Sequence[str]) -> bool:
    """Gemma's giveaway: a run of spaces written as spaces, where sentencepiece writes `\u2581`."""
    return any(token and " " in token and not token.strip() for token in tokens)


def _select_converter(architecture: str, tokenizer_type: str | None, tokens: Sequence[str] = ()):
    """The class that assembles the tokenizer a file describes.

    `tokenizer.ggml.model` decides, as it does in llama.cpp; the architecture says nothing about the
    tokenizer, Mistral being sentencepiece up to v0.3 and byte-level after, both written as `llama`.

    `llama` holds more than one, because we rebuild the tokenizer a file was converted *from* and
    Gemma's and Phi-3's are not the plain sentencepiece one. The vocabulary gives Gemma away; nothing
    gives Phi-3 away, so it alone is read off the architecture.
    """
    if tokenizer_type == "llama":
        if architecture == "phi3":
            return GGUFPhi3Converter
        if _is_gemma_vocabulary(tokens):
            return GGUFGemmaConverter
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


_QWEN2_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


_QWEN35_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


# llama.cpp rewrites this one to dodge the character classes its own regex engine lacks; the form
# here is the original it records alongside, which is what `tokenizers` wants.
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


# `tokenizer.ggml.pre` names a known tokenizer, not a model: llama.cpp tokenizes a fixed string with
# every one it has seen and a file takes the name whose output its own reproduces, so several names
# share a split. Only splits differing from GPT-2's are listed; `ByteLevel` already applies that one.
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


def _split_before_byte_level(tokenizer, skeleton) -> None:
    """Cut text into the chunks this vocabulary's merges were learned on.

    BPE only merges within a chunk, so the regex deciding them decides the result: `def f(x):` is
    `f`, `(`, `x`, `):` under GPT-2's and `f`, `(x`, `):` under Llama-3's. `ByteLevel` cuts with
    GPT-2's, so a file naming another needs that one in front of it instead.
    """
    split = GGUF_PRE_TOKENIZER_SPLITS.get(getattr(skeleton, "pre_tokenizer_type", None))
    if split is not None:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(split), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )


class GGUFTokenizerSkeleton:
    def __init__(self, dict_, merge_ranks_from_vocab_order: bool = False):
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
            # What orders the merges: the scores, where they are sentencepiece log-probabilities, and
            # the position, where the vocabulary is already merge-ordered and the scores are ranks with
            # placeholders punched through them -- every whitespace run in a Gemma file scores -1000.
            rank = {t: -i for i, t in enumerate(tokens)} if merge_ranks_from_vocab_order else vocab

            logger.warning("Merges were not in checkpoint, building merges on the fly.")
            merges = []
            for merge in tqdm(vocab):
                local = []
                for index in range(1, len(merge)):
                    piece_l, piece_r = merge[:index], merge[index:]
                    # `vocab`, not `tokens`: same keys, but a list membership test is linear and this
                    # runs once per split of every token in the vocabulary.
                    if piece_l in vocab and piece_r in vocab:
                        local.append((piece_l, piece_r, rank[merge]))
                local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]), reverse=True)
                merges.extend(local)
            merges = sorted(merges, key=lambda val: val[2], reverse=True)
            merges = [(val[0], val[1]) for val in merges]
            self.merges = merges
        else:
            self.merges = [tuple(merge.split(" ")) for merge in self.merges]
            if not hasattr(self, "scores"):
                self.scores = [None for _ in range(len(self.tokens))]

        if not hasattr(self, "unk_token_id"):
            self.unk_token_id = None


class GGUFLlamaConverter(LlamaConverter):
    def __init__(self, tokenizer_dict):
        self.proto = GGUFTokenizerSkeleton(tokenizer_dict)
        # No GGUF states this, and every sentencepiece repo still published sets it: no normalizer, a
        # `Metaspace` pre-tokenizer instead. The legacy shape mistokenizes a leading run of spaces.
        self.proto.legacy = False
        self.original_tokenizer = self.proto
        self.additional_kwargs = {"legacy": False}

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
        eos_token = proto.tokens[proto.eos_token_id] if getattr(proto, "eos_token_id", None) is not None else None

        tokenizer = Tokenizer(
            BPE(
                bpe_vocab,
                merges,
                unk_token=unk_token,
                fuse_unk=True,
                byte_fallback=True,
            )
        )

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

        self.additional_kwargs["unk_token"] = unk_token
        self.additional_kwargs["eos_token"] = eos_token
        self.additional_kwargs["bos_token"] = bos_token

        return tokenizer

    def decoder(self, replacement, add_prefix_space):
        sequence = [
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Replace("▁", " "),
        ]

        if add_prefix_space:
            sequence += [decoders.Strip(content=" ", left=1)]
        return decoders.Sequence(sequence)


class GGUFPhi3Converter(LlamaConverter):
    def __init__(self, tokenizer_dict):
        self.proto = GGUFTokenizerSkeleton(tokenizer_dict)
        self.original_tokenizer = self.proto
        self.additional_kwargs = {}

    def vocab(self, proto):
        return list(zip(proto.tokens, proto.scores))

    def merges(self, proto):
        return proto.merges

    def tokenizer(self, proto):
        vocab_scores = self.vocab(self.proto)
        merges = self.merges(self.proto)
        bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}

        unk_token = proto.tokens[proto.unk_token_id] if proto.unk_token_id is not None else None
        tokenizer = Tokenizer(BPE(bpe_vocab, merges, unk_token=unk_token, fuse_unk=True, byte_fallback=True))
        # add the special tokens from phi3 tokenizer config
        tokenizer.add_special_tokens(
            [
                AddedToken("</s>", rstrip=True, lstrip=False, normalized=False, special=True),
                AddedToken("<|endoftext|>", normalized=False, special=True),
                AddedToken("<|assistant|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder1|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder2|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder3|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder4|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|system|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|end|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder5|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder6|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|user|>", rstrip=True, normalized=False, special=True),
            ]
        )

        self.additional_kwargs["unk_token"] = (
            proto.tokens[proto.unk_token_id] if proto.unk_token_id is not None else None
        )
        self.additional_kwargs["eos_token"] = (
            proto.tokens[proto.eos_token_id] if proto.eos_token_id is not None else None
        )
        self.additional_kwargs["bos_token"] = (
            proto.tokens[proto.bos_token_id] if proto.bos_token_id is not None else None
        )
        self.additional_kwargs["pad_token"] = (
            proto.tokens[proto.pad_token_id] if proto.pad_token_id is not None else None
        )

        return tokenizer

    def decoder(self, replacement, add_prefix_space):
        sequence = [
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Replace(replacement, " "),
        ]

        if add_prefix_space:
            sequence += [decoders.Strip(content=" ", left=1)]
        return decoders.Sequence(sequence)


_BYTE_LEVEL_ALPHABET = set(pre_tokenizers.ByteLevel.alphabet())


def _byte_level_vocab(tokens: list[str]) -> dict[str, int]:
    """The vocabulary of a byte-level file, keyed the way its merges spell it.

    llama.cpp writes these in GPT-2's byte-level spelling, but one whose bytes do not decode comes
    back raw: Phi-4 holds `\ufffd` where its merges say `ï¿½`, and a merge naming a token the
    vocabulary lacks is a hard error.
    """
    spelling = bytes_to_unicode()
    return {
        token if set(token) <= _BYTE_LEVEL_ALPHABET else "".join(spelling[b] for b in token.encode("utf-8")): i
        for i, token in enumerate(tokens)
    }


class GGUFGPTConverter(GPT2Converter):
    def __init__(self, tokenizer_dict):
        self.original_tokenizer = GGUFTokenizerSkeleton(tokenizer_dict)
        self.additional_kwargs = {}

    def converted(self) -> Tokenizer:
        vocab = _byte_level_vocab(self.original_tokenizer.tokens)
        merges = self.original_tokenizer.merges
        tokenizer = super().converted(vocab, merges)
        _split_before_byte_level(tokenizer, self.original_tokenizer)

        # Type 3 is `LLAMA_TOKEN_TYPE_CONTROL`. A respelled token is not one of them: it is a byte
        # sequence llama.cpp mislabelled, and registering it would take it out of the merges.
        token_type = getattr(self.original_tokenizer, "token_type", None) or []
        control = [
            AddedToken(token, normalized=False, special=True)
            for token, kind in zip(self.original_tokenizer.tokens, token_type)
            if kind == 3 and set(token) <= _BYTE_LEVEL_ALPHABET
        ]
        if control:
            tokenizer.add_special_tokens(control)
        return tokenizer


class GGUFT5Converter(T5Converter):
    def __init__(self, tokenizer_dict):
        # A unigram model has no merges, and stating one keeps the skeleton from deriving any.
        self.proto = GGUFTokenizerSkeleton(tokenizer_dict | {"merges": ["dummy text"]})
        self.token2id = {k: v for v, k in enumerate(self.proto.tokens)}
        self.original_tokenizer = self.proto
        self.additional_kwargs = {}

    def vocab(self, proto):
        return list(zip(proto.tokens, proto.scores))

    def normalizer(self, proto):
        # The file carries sentencepiece's own normalization table, which is what the reference uses.
        charsmap = getattr(proto, "precompiled_charsmap", None)
        if charsmap is not None:
            return normalizers.Precompiled(bytes(charsmap))
        return None

    def pre_tokenizer(self, replacement, add_prefix_space):
        # `WhitespaceSplit` first, or a run of spaces survives as one `▁` per space.
        return pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme="always", split=True),
            ]
        )

    def post_processor(self):
        return processors.TemplateProcessing(
            single=["$A", "</s>"],
            pair=["$A", "</s>", "$B", "</s>"],
            special_tokens=[
                ("</s>", self.token2id["</s>"]),
            ],
        )

    def converted(self) -> Tokenizer:
        vocab_scores = self.vocab(self.proto)
        tokenizer = Tokenizer(
            Unigram(
                vocab_scores,
                unk_id=self.proto.unk_token_id,
                byte_fallback=False,
            )
        )

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

        return tokenizer


class GGUFGemmaConverter(GemmaConverter):
    def __init__(self, tokenizer_dict):
        # Respelled before the skeleton derives merges, not after: it splits a token into pieces it can
        # find in the vocabulary, and `▁▁` is only reachable from `▁` once both are written that way.
        tokenizer_dict = tokenizer_dict | {"tokens": [self._spelled(token) for token in tokenizer_dict["tokens"]]}
        self.proto = GGUFTokenizerSkeleton(tokenizer_dict, merge_ranks_from_vocab_order=True)
        self.original_tokenizer = self.proto
        self.additional_kwargs = {}

    @staticmethod
    def _spelled(token: str) -> str:
        """The form the vocabulary holds a token in: a run of spaces written as `▁`."""
        if " " in token and len(token.strip()) == 0:
            return "▁" * len(token)
        return token

    def vocab(self, proto):
        return list(zip(proto.tokens, proto.scores))

    def normalizer(self, proto):
        return normalizers.Replace(" ", "▁")

    def decoder(self, replacement, add_prefix_space):
        sequence = [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]

        if add_prefix_space:
            sequence += [decoders.Strip(content=" ", left=1)]
        return decoders.Sequence(sequence)

    def converted(self) -> Tokenizer:
        # A BPE, not a Unigram: the file's "scores" are merge ranks, and a Unigram reads them as
        # costs and splits `capital` into its cheapest pieces.
        vocab_scores = self.vocab(self.proto)
        unk_token = self.proto.tokens[self.proto.unk_token_id] if self.proto.unk_token_id is not None else None
        tokenizer = Tokenizer(
            BPE(
                {word: i for i, (word, _score) in enumerate(vocab_scores)},
                self.proto.merges,
                unk_token=unk_token,
                fuse_unk=True,
                byte_fallback=True,
            )
        )

        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = True
        if hasattr(self.original_tokenizer, "add_prefix_space"):
            add_prefix_space = self.original_tokenizer.add_prefix_space

        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        return tokenizer


# `tokenizer.ggml.model`, which is the whole of what a file says its tokenizer is.
GGUF_TOKENIZER_KINDS = {
    "gpt2": GGUFGPTConverter,
    "llama": GGUFLlamaConverter,
    "t5": GGUFT5Converter,
    "gemma4": GGUFGemmaConverter,
}
