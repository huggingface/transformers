# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformers import T5Tokenizer, T5TokenizerFast, PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizerBase
import re
import sentencepiece as spm

# The special tokens of T5Tokenizer is hard-coded with <extra_id_{}>
# Created another class UDOPTokenizer extending it to add special visual tokens like <loc_{}>

class UdopTokenizer(T5Tokenizer):

    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        loc_extra_ids=501,
        other_extra_ids=200,
        additional_special_tokens=None,
        sp_model_kwargs=None,
        **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
            additional_special_tokens.extend(["<extra_l_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["</extra_l_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["<extra_t_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["</extra_t_id_{}>".format(i) for i in range(extra_ids)])
        
        elif extra_ids > 0 and additional_special_tokens is not None:
            extra_ids = 0

        if loc_extra_ids > 0 and not "<loc_0>" in additional_special_tokens:
            additional_special_tokens.extend(["<loc_{}>".format(i) for i in range(loc_extra_ids)])

        if other_extra_ids > 0 and not "<other_0>" in additional_special_tokens:
            additional_special_tokens.extend(["<other_{}>".format(i) for i in range(other_extra_ids)])
    
        PreTrainedTokenizer.__init__(
            self,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._loc_extra_ids = loc_extra_ids
        self._other_extra_ids = other_extra_ids

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
        
    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size() + self._extra_ids * 5 + self._loc_extra_ids + self._other_extra_ids

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(
            i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 4
        elif token.startswith("<extra_l_id_"):
            match = re.match(r"<extra_l_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 3
        elif token.startswith("</extra_l_id_"):
            match = re.match(r"</extra_l_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 2
        elif token.startswith("<extra_t_id_"):
            match = re.match(r"<extra_t_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids - self._extra_ids
        elif token.startswith("</extra_t_id_"):
            match = re.match(r"</extra_t_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids
        elif token.startswith("<loc_"):
            match = re.match(r"<loc_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids
        elif token.startswith("<other_"):
            match = re.match(r"<other_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            
            if index > self.sp_model.get_piece_size() + self._extra_ids * 5 + self._loc_extra_ids - 1:
                index_loc = self.vocab_size - 1 - index
                token = f"<other_{index_loc}>"           
            elif index > self.sp_model.get_piece_size() + self._extra_ids * 5 - 1:
                index_loc = self.vocab_size - self._other_extra_ids - 1 - index
                token = f"<loc_{index_loc}>"   
            elif index > self.sp_model.get_piece_size() + self._extra_ids * 4 - 1:
                token = "</extra_t_id_{}>".format(self.vocab_size - self._other_extra_ids - self._loc_extra_ids - 1 - index)
            elif index > self.sp_model.get_piece_size() + self._extra_ids * 3 - 1:
                token = "<extra_t_id_{}>".format(self.vocab_size - self._other_extra_ids - self._loc_extra_ids - self._extra_ids - 1 - index)
            elif index > self.sp_model.get_piece_size() + self._extra_ids * 2 - 1:
                token = "</extra_l_id_{}>".format(self.vocab_size - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 2 - 1 - index)
            elif index > self.sp_model.get_piece_size() + self._extra_ids - 1:
                token = "<extra_l_id_{}>".format(self.vocab_size - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 3 - 1 - index)
            elif index > self.sp_model.get_piece_size() - 1:
                token = "<extra_id_{}>".format(self.vocab_size - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 4 - 1 - index)
            else:
                raise
        return token


# Below are for Rust-based Fast Tokenizer

from transformers.convert_slow_tokenizer import SpmConverter
from tokenizers import Tokenizer, processors
from typing import List


class UdopConverter(SpmConverter):
    def vocab(self, proto):
        vocab = [(piece.piece, piece.score) for piece in proto.pieces]
        num_extra_ids = self.original_tokenizer._extra_ids
        vocab += [("<extra_id_{}>".format(i), 0.0)
                  for i in range(num_extra_ids - 1, -1, -1)]
        vocab += [("<extra_l_id_{}>".format(i), 0.0)
                  for i in range(num_extra_ids - 1, -1, -1)]
        vocab += [("</extra_l_id_{}>".format(i), 0.0)
                  for i in range(num_extra_ids - 1, -1, -1)]
        vocab += [("<extra_t_id_{}>".format(i), 0.0)
                  for i in range(num_extra_ids - 1, -1, -1)]
        vocab += [("</extra_t_id_{}>".format(i), 0.0)
                  for i in range(num_extra_ids - 1, -1, -1)]
        
        num_loc_extra_ids = self.original_tokenizer._loc_extra_ids
        vocab += [("<loc_{}>".format(i), 0.0)
                  for i in range(num_loc_extra_ids - 1, -1, -1)]

        num_other_extra_ids = self.original_tokenizer._other_extra_ids
        vocab += [("<other_0{}>".format(i), 0.0)
                  for i in range(num_other_extra_ids - 1, -1, -1)]
        
        return vocab

    def post_processor(self):
        return processors.TemplateProcessing(
            single=["$A", "</s>"],
            pair=["$A", "</s>", "$B", "</s>"],
            special_tokens=[
                ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
            ],
        )


def convert_slow_udoptokenizer(UdopTokenizer):
    return UdopConverter(UdopTokenizer).converted()


class UdopTokenizerFast(T5TokenizerFast):

    slow_tokenizer_class = UdopTokenizer
    prefix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file,
        tokenizer_file=None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        loc_extra_ids=201,
        other_extra_ids=200,
        additional_special_tokens=None,
        **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
            additional_special_tokens.extend(["<extra_l_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["</extra_l_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["<extra_t_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["</extra_t_id_{}>".format(i) for i in range(extra_ids)])

        if loc_extra_ids > 0 and not "<loc_0>" in additional_special_tokens:
            additional_special_tokens.extend(["<loc_{}>".format(i) for i in range(loc_extra_ids)])
            
        if other_extra_ids > 0 and not "<other_0>" in additional_special_tokens:
            additional_special_tokens.extend(["<other_{}>".format(i) for i in range(other_extra_ids)])
        
        slow_tokenizer = self.slow_tokenizer_class(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            loc_extra_ids=loc_extra_ids,
            other_extra_ids=other_extra_ids,
            **kwargs
        )
        fast_tokenizer = convert_slow_udoptokenizer(slow_tokenizer)
        self._tokenizer = fast_tokenizer

        PreTrainedTokenizerBase.__init__(
            self,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._loc_extra_ids = loc_extra_ids
        self._other_extra_ids = other_extra_ids
