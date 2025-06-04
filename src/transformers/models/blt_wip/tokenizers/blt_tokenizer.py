# Copyright (c) Meta Platforms, Inc. and affiliates.
import re

from .abstract_tokenizer import Tokenizer
from .sentence_piece_tokenizer import SentencePieceTokenizer


SEP = " "
BOS_ID: int = 1
EOS_ID: int = 2
PAD_ID: int = -1
BOE_ID: int = 0
BPE_ID: int = 3
OFFSET: int = 4

BYTE_UNITS: int = 256


def convert_to_bytes(s):
    # check if the output is a bytes like object of the format <0x00>
    if re.match(r"<0x[0-9a-fA-F]+>", s):
        return bytes.fromhex(s[3:-1])
    else:
        return bytes(s, "utf-8", errors="ignore")


def text2bytes_bpe_delims(
    text: str,
    *,
    bpe_tokenizer,
    bpe_id: int,
    offsetting_special_char: int,
    add_bos: bool,
    add_eos: bool,
):
    cur_bpe = bpe_tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
    # merge the leading space tokens
    leading_space_tokens = []
    other_bpe_tokens = []
    leading = True
    for token in cur_bpe:
        bpe_str = bpe_tokenizer.sp_model.id_to_piece(token)
        if leading and all(c == "▁" for c in bpe_str):
            leading_space_tokens.append(bpe_str)
        else:
            leading = False
            other_bpe_tokens.append(bpe_str)
    cur_bpe_strs = ["".join(leading_space_tokens)] + other_bpe_tokens

    # Remove the '▁' characters
    bpe_strs = []
    for i, bpe_str in enumerate(cur_bpe_strs):
        if (
            len(bpe_strs) <= 1
            and all([c == " " for s in bpe_strs for c in s])
            and not all(c == "▁" for c in bpe_str)
        ):
            # Remove leading space for first non space token.
            bpe_str = bpe_str.replace("▁", "")
        elif i == 0 and all(c == "▁" for c in bpe_str):
            bpe_str = " " * (len(text) - len(text.lstrip(" ")))
        else:
            bpe_str = bpe_str.replace("▁", " ")
        if len(bpe_str) > 0:
            bpe_strs.append(bpe_str)
    ex_seq = []
    # Convert bpe tokens to bytes
    for s in bpe_strs:
        byte_chunk = convert_to_bytes(s)
        proc_chunk = [int(unit) for unit in byte_chunk]
        ex_seq.extend([bpe_id - offsetting_special_char] + proc_chunk)

    return ex_seq


class BltTokenizer(Tokenizer):
    def __init__(
        self,
        *,
        vocab_size_unit_1: int = BYTE_UNITS,
        bpe_delim: bool = False,
        bpe_tokenizer_path="/home/artidoro/tokenizers/llama_v2.tokenizer.model",
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.vocab_size_unit_1 = vocab_size_unit_1
        self.boe_id = BOE_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.pad_id = PAD_ID
        self.bpe_id = BPE_ID
        self.bpe_tokenizer_path = bpe_tokenizer_path
        if bpe_delim:
            self.bpe_tokenizer = SentencePieceTokenizer(
                model_path=self.bpe_tokenizer_path
            )
        else:
            self.bpe_tokenizer = None
        self.bpe_delim = bpe_delim
        self.offsetting_special_char = OFFSET
        self.vocab_size_unit_1 = vocab_size_unit_1
        self.n_words = vocab_size_unit_1 + self.offsetting_special_char

    def get_vocab_size(self) -> int:
        return self.n_words

    def encode(
        self, text: str, add_bos: bool | None = None, add_eos: bool | None = None
    ):
        if add_bos is None:
            add_bos = self.add_bos
        if add_eos is None:
            add_eos = self.add_eos

        if self.bpe_delim:
            tokens = text2bytes_bpe_delims(
                text,
                bpe_tokenizer=self.bpe_tokenizer,
                bpe_id=self.bpe_id,
                offsetting_special_char=self.offsetting_special_char,
                add_bos=False,
                add_eos=False,
            )
        else:
            tokens = bytes(text, encoding="utf-8", errors="ignore")

        # Offsetting
        tokens = [int(unit) + self.offsetting_special_char for unit in tokens]

        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, tokens: list[int], cut_at_eos: bool = False):
        if cut_at_eos:
            for k, t in enumerate(tokens):
                if t == self.eos_id:
                    tokens = tokens[: k + 1]
                    break
        return bytes(
            [
                tok - self.offsetting_special_char
                for tok in tokens
                if tok - self.offsetting_special_char >= 0
            ]
        ).decode("utf-8", errors="ignore")

    def get_token_offsets(self, text: str, tokens: list[int] | None = None):
        # TODO: Figure out what this does
        raise NotImplementedError()
