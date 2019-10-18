from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, cast

import lm
from sentencepiece import SentencePieceProcessor
from .tokenization_utils import PreTrainedTokenizer


class SentencePieceTokenizer(PreTrainedTokenizer):
    def __init__(self, model_path: Path, unk_token="<unk>", eot_token="<eot>", cls_token="<cls>", sep_token="<sep>", mask_token="<mask>", pad_token="<pad>", **kwargs):
        self.sp = self.load_sentencepieceprocessor(model_path)
        self.eot_token = eot_token
        super(SentencePieceTokenizer, self).__init__(unk_token=unk_token, sep_token=sep_token, pad_token=pad_token, cls_token=cls_token, mask_token=mask_token, **kwargs)

    @property
    def __len__(self):
        return self.vocab_size

    @property
    def vocab_size(self):
        return len(self.sp)

    @classmethod
    def from_pretrained(cls, path: Path, **kwargs):
        return cls(path, **kwargs)

    @classmethod
    def load_sentencepieceprocessor(cls, model_path: Path):
        sp = SentencePieceProcessor()
        sp.Load(f"{model_path}/sp.model")
        return sp

    def convert_tokens_to_ids(self, *tokens):
        ids = list(map(self.sp.piece_to_id, tokens))
        if len(ids) > 1:
            return ids
        return ids[0]

    def _convert_id_to_token(self, index: int) -> str:
        return self.sp.id_to_piece(index)

    def add_special_tokens_single_sequence(self, token_ids: List[int], no_eos: bool = False) -> List[int]:
        return [self.bos_token_id, *token_ids] + ([] if no_eos else [self.eos_token_id])

    def tokenize(self, text: str) -> List[str]:
        return cast(List[str], self.sp.encode_as_pieces(text))

    def encode(self, text: str, no_eos: bool = False) -> List[str]:
        return self.add_special_tokens_single_sequence(self.sp.EncodeAsIds(text), no_eos)

    def decode(self, ids: Sequence[int], **kwargs) -> List[str]:
        return list(map(self._convert_id_to_token, ids))

    def prettify(self, tokens: List[str]) -> str:
        return self.sp.decode_pieces(tokens)

    def print_tokens(self, ids: List[int], token: str):
        tokens, token = self.decode(ids), self.sp.piece_to_id(token)
        ending_puncts = "?!)])>:;}.,"
        starting_puncts = "([{<"

        tok_print = lambda tok: print(tok, end="", flush=True)

        normalized_token: str = token.replace(lm.END_OF_LINE, "\n").replace(lm.END_OF_TEXT, "\n").replace("▁", " ")
        if (len(normalized_token) > 1 and normalized_token[1] in ending_puncts) or (len(tokens) > 1 and tokens[-2].replace("▁", "") in starting_puncts):
            normalized_token = normalized_token.replace(" ", "")

        tok_print(normalized_token)

    @lru_cache()
    def stop_id(self):
        return self.sp.PieceToId(self.eot_token)
