from pathlib import Path
from transformers.tokenization_roberta import lru_cache
from typing import Iterable, Sequence
import lm
from lm.inference import ModelWrapper
from lm.model import OutputGetters
import sentencepiece as sntpc

from transformers.configuration_gpt2 import GPT2Config


class IamBotSentencePiece:
    def __init__(self, model_path: Path, model: sntpc.SentencePieceProcessor = None):
        self.sp = model or self.load_sentencepieceprocessor(model_path)

    @classmethod
    def from_pretrained(cls, path: Path):
        return cls(path)

    @classmethod
    def load_sentencepieceprocessor(cls, model_path: Path):
        sp = sntpc.SentencePieceProcessor()
        sp.Load(f'{model_path}/sp.model')
        return sp

    def encode(self, text: str):
        return [self.sp.PieceToId(lm.END_OF_TEXT), *self.sp.EncodeAsIds(text)]

    def decode(self, ids: Sequence[int], **kwargs):
        return list(map(self.sp.IdToPiece, ids))

    def prettify(self, tokens):
        return self.sp.DecodePieces(tokens)

    def print_tokens(self, tokens, token):
        tokens, token = list(map(self.sp.IdToPiece, tokens)), self.sp.IdToPiece(token)
        ending_puncts = "?!)])>:;}.,"
        starting_puncts = "([{<"

        tok_print = lambda tok: print(tok, end="", flush=True)

        normalized_token: str = token.replace(lm.END_OF_LINE, "\n").replace(lm.END_OF_TEXT, "\n").replace("▁", " ")
        if (len(normalized_token) > 1 and normalized_token[1] in ending_puncts) or (len(tokens) > 1 and tokens[-2].replace("▁", "") in starting_puncts):
            normalized_token = normalized_token.replace(" ", "")

        tok_print(normalized_token)

    @lru_cache()
    def stop_id(self):
        return self.sp.PieceToId(lm.END_OF_TEXT)

class IamBotGPT2:
    config = GPT2Config()

    def __init__(self, model: ModelWrapper):
        self.model = model

    @classmethod
    def from_pretrained(cls, path: Path):
        model = ModelWrapper.load_encoder(f'{path}/model.pt', True, False, 'cpu', output_getter=OutputGetters.raw)
        return cls(model)

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model = self.model.to(device)

    def __call__(self, input_ids, **kwargs):
        output = self.model(input_ids)['logits']
        return output,
