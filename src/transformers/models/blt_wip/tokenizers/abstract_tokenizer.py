# Copyright (c) Meta Platforms, Inc. and affiliates.
import abc


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode(self, text: str, add_bos: bool, add_eos: bool):
        pass

    @abc.abstractmethod
    def decode(self, tokens: list[int]):
        pass

    @abc.abstractmethod
    def get_token_offsets(
        self, text: str, tokens: list[int] | None = None
    ) -> tuple[list[str], list[int]]:
        """Return the offsets of the tokens in the original text. Only used for evaluation."""
        pass

    @abc.abstractmethod
    def get_vocab_size(self) -> int:
        pass
