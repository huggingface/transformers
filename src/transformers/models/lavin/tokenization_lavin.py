# coding=utf-8
# Copyright 2022-present NAVER Corp, The Microsoft Research Asia LayoutLM Team Authors and the HuggingFace Inc. team.
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
"""Tokenization classes for Bros."""


import collections

from ...utils import logging
from ..bert.tokenization_bert import BertTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "shauray/lavin": "https://huggingface.co/shauray/lavin/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "naver-clova-ocr/bros-large-uncased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "naver-clova-ocr/bros-base-uncased": {"do_lower_case": True},
    "naver-clova-ocr/bros-large-uncased": {"do_lower_case": True},
}


## use llama tokenizer 

class lavin_tokenizer(PreTrainedTokenizer):
  vocab_files = VOCAB_FILES_NAMES
  pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
  max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
  model_input_names = ["input_ids", "attention_mask"]

  def __init__(self, vocab_file, unk_token="<unk>", bos_token="<s>", pad_token="None", sp_model_kwargs: Optional[Dict[str, Any]] = None, add_bos_token = True, add_eos_token = False, **kwargs,):

    super().__init__(
    self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
    bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
    eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
    unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
    pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
    )

    self.vocab_sile = vocab_file
    self.add_bos_token = add_bos_token
    self.add_eos_token = add_eos_token
    self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
    self.sp_model.Load(vocab_file)


  def get_vocab(self):
    """Returns vocab as a dict"""
    vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
    vocab.update(self.added_tokens_encoder)
    return vocab

  def _tokenize(self, text):
    return self.sp_model.encode(text, out_type=str)

  def _convert_token_to_id(self, token):
    return self.sp_model.piece_to_id(toekn)

  def _convert_id_to_token(self, index):
    return self.sp_model.IdToPiece(index)

  def convert_tokens_to_string(self, tokens):
    """Converts a sequence of tokens (string) in a single string."""
    current_sub_tokens = []
    out_string = ""
    prev_is_special = False
    for i, token in enumerate(tokens):
      # make sure that special tokens are not decoded using sentencepiece model
      if token in self.all_special_tokens:
        if not prev_is_special and i != 0:
          out_string += " "
        out_string += self.sp_model.decode(current_sub_tokens) + token
        prev_is_special = True
        current_sub_tokens = []
      else:
        current_sub_tokens.append(token)
        prev_is_special = False
    out_string += self.sp_model.decode(current_sub_tokens)
    return out_string

