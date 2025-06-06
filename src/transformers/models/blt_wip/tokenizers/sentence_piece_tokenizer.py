# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
import os

try:
    from sentencepiece import SentencePieceProcessor

    has_sp = True
except ImportError:
    has_sp = False

from .abstract_tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class SentencePieceTokenizer(Tokenizer):
    def __init__(
        self, model_path: str, add_bos: bool = True, add_eos: bool = True
    ) -> None:
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        self.add_bos = add_bos
        self.add_eos = add_eos
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def get_vocab_size(self) -> int:
        return self.n_words

    def encode(self, s: str, add_bos: bool | None = None, add_eos: bool | None = None):
        if add_bos is None:
            add_bos = self.add_bos

        if add_eos is None:
            add_eos = self.add_eos
        assert type(s) is str
        tokens = (
            [self.bos_id] * add_bos + self.sp_model.encode(s) + [self.eos_id] * add_eos
        )
        return tokens

    def decode(self, tokens: list[int]):
        return self.sp_model.decode(tokens)

    def get_token_offsets(
        self, text: str, tokens: list[int] | None = None
    ) -> tuple[list[str], list[int]]:
        pieces = self.sp_model.encode_as_immutable_proto(text).pieces
        substrs = [p.surface for p in pieces]
        offsets = [p.begin for p in pieces]
        return substrs, offsets
