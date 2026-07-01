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

"""Conversion between Mistral tekken tokenizers and HuggingFace tokenizer formats."""

import base64
import json
from functools import lru_cache

from tokenizers import AddedToken, Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

from ...convert_slow_tokenizer import bytes_to_unicode
from ...utils import logging


logger = logging.get_logger(__name__)

_MAP_SPECIALS = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
}


class MistralConverter:
    """Converter from Mistral tekken BPE vocab to a HuggingFace `tokenizers.Tokenizer`."""

    def __init__(
        self,
        pattern: str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        add_prefix_space: bool = False,
        additional_special_tokens: list[AddedToken] | None = None,
        **kwargs,
    ):
        self.pattern = pattern
        self.add_prefix_space = add_prefix_space
        self.additional_special_tokens = additional_special_tokens
        self._precomputed_vocab: dict[str, int] | None = None
        self._precomputed_merges: list[tuple[str, str]] | None = None

    @classmethod
    def from_tekken_file(
        cls,
        vocab_file: str,
        add_prefix_space: bool = False,
    ) -> "MistralConverter":
        """Parse a raw `tekken.json` file and return a ready-to-use converter.

        Matches `mistral_common`'s `Tekkenizer.from_file` vocab and special-token layout.
        """
        with open(vocab_file, encoding="utf-8") as f:
            untyped = json.load(f)

        config = untyped["config"]
        pattern = config["pattern"]

        vocab_size = config.get("default_vocab_size")
        num_special_tokens = config.get("default_num_special_tokens")

        filler_template = "<SPECIAL_{id}>"

        special_tokens_dicts = untyped.get("special_tokens")
        if special_tokens_dicts is None:
            # Old tekken format has no special_tokens key; use mistral-common's defaults.
            from mistral_common.tokens.tokenizers.tekken import Tekkenizer

            filler_template = getattr(Tekkenizer, "SPECIAL_TOKEN_TEMPLATE", filler_template)
            special_tokens_dicts = list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)

        special_tokens_dicts = [
            {"rank": entry["rank"], "token_str": str(getattr(entry["token_str"], "value", entry["token_str"]))}
            for entry in special_tokens_dicts
        ]

        if num_special_tokens is None:
            num_special_tokens = len(special_tokens_dicts)

        special_filler = [
            {"rank": i, "token_str": filler_template.format(id=i)}
            for i in range(len(special_tokens_dicts), num_special_tokens)
        ]
        special_tokens_dicts = special_tokens_dicts + special_filler

        additional_special_tokens = [AddedToken(entry["token_str"], special=True) for entry in special_tokens_dicts]

        # Drop padded vocab: keep only the real tokens (matches mistral-common).
        bpe_ranks_raw = untyped["vocab"]
        if vocab_size is not None:
            inner_vocab_size = vocab_size - num_special_tokens
            bpe_ranks_raw = bpe_ranks_raw[:inner_vocab_size]

        bpe_ranks = [base64.b64decode(k["token_bytes"]) for k in bpe_ranks_raw]
        bpe_ranks_dict = {token: rank for rank, token in enumerate(bpe_ranks)}

        vocab, merges = cls._extract_merges(bpe_ranks_dict)

        vocab = {k: v + num_special_tokens for k, v in vocab.items()}
        for entry in special_tokens_dicts:
            vocab[entry["token_str"]] = entry["rank"]

        instance = cls(
            pattern=pattern,
            add_prefix_space=add_prefix_space,
            additional_special_tokens=additional_special_tokens,
        )
        instance._precomputed_vocab = vocab
        instance._precomputed_merges = merges

        return instance

    @staticmethod
    def _extract_merges(bpe_ranks: dict[bytes, int]) -> tuple[dict[str, int], list[tuple[str, str]]]:
        """Extract a unicode vocab and ordered BPE merge list from byte-level BPE ranks."""
        byte_encoder = bytes_to_unicode()

        @lru_cache
        def token_bytes_to_string(b: bytes) -> str:
            return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

        vocab: dict[str, int] = {}
        all_merges: list[tuple[bytes, bytes, int]] = []

        for token, rank in bpe_ranks.items():
            vocab[token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            local = []
            for index in range(1, len(token)):
                piece_l, piece_r = token[:index], token[index:]
                if piece_l in bpe_ranks and piece_r in bpe_ranks and (piece_l + piece_r) in bpe_ranks:
                    local.append((piece_l, piece_r, rank))
            local = sorted(local, key=lambda x: (bpe_ranks[x[0]], bpe_ranks[x[1]]))
            all_merges.extend(local)

        all_merges = sorted(all_merges, key=lambda val: val[2])
        merges = [(token_bytes_to_string(val[0]), token_bytes_to_string(val[1])) for val in all_merges]
        return vocab, merges

    def tokenizer(self) -> Tokenizer:
        """Build a raw `tokenizers.Tokenizer` with BPE model (no pre/post-processing)."""
        tokenizer = Tokenizer(BPE(self._precomputed_vocab, self._precomputed_merges, fuse_unk=False))
        if hasattr(tokenizer.model, "ignore_merges"):
            tokenizer.model.ignore_merges = True
        return tokenizer

    def converted(self) -> Tokenizer:
        """Build a fully configured `tokenizers.Tokenizer` with pre-tokenizer and decoder."""
        tokenizer = self.tokenizer()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(self.pattern), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=self.add_prefix_space, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.add_special_tokens(self.additional_special_tokens)

        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        return tokenizer
