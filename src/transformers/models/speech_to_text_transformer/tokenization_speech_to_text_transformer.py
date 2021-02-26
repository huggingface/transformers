# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for Speech2TextTransformer."""

import json
from pathlib import Path
from shutil import copyfile
from typing import Dict, List, Optional, Tuple, Union

import sentencepiece

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "spm_file": "sentencepiece.bpe.model",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "s2t_librispeech_transformer_small": "https://huggingface.co/valhalla/s2t_librispeech_transformer_small/resolve/main/vocab.json",
    },
    "spm_file": {
        "s2t_librispeech_transformer_small": "https://huggingface.co/valhalla/s2t_librispeech_transformer_small/resolve/main/sentencepiece.bpe.model"
    },
}


class Speech2TextTransformerTokenizer(PreTrainedTokenizer):
    """
    Construct an Speech2TextTransformer tokenizer.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains some of the main methods.
    Users should refer to the superclass for more information regarding such methods.

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        spm_file (:obj:`str`):
            Path to the `SentencePiece <https://github.com/google/sentencepiece>`__ model file
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The beginning of sentence token.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sentence token.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        do_upper_case (:obj:`bool`, `optional`, defaults to :obj:`False`):
           Whether or not to uppercase the output when decoding.

        **kwargs
            Additional keyword arguments passed along to :class:`~transformers.PreTrainedTokenizer`
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        spm_file,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        do_upper_case=False,
        **kwargs,
    ):
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            do_upper_case=do_upper_case,
            **kwargs,
        )
        self.do_upper_case = do_upper_case

        self.encoder = load_json(vocab_file)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.spm_file = spm_file
        self.sp_model = load_spm(spm_file)

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the decoder."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()

        if self.do_upper_case:
            out_string = out_string.upper()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def get_vocab(self) -> Dict:
        vocab = self.encoder.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d: Dict) -> None:
        self.__dict__ = d
        self.sp_model = load_spm(self.spm_file)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        save_dir = Path(save_directory)
        assert save_dir.is_dir(), f"{save_directory} should be a directory"
        vocab_save_path = save_dir / (
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )
        spm_save_path = save_dir / (
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["spm_file"]
        )

        save_json(self.encoder, vocab_save_path)

        if not spm_save_path.exists():
            copyfile(self.spm_file, spm_save_path)

        return (str(vocab_save_path), str(spm_save_path))


def load_spm(path: str) -> sentencepiece.SentencePieceProcessor:
    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(str(path))
    return spm


def load_json(path: str) -> Union[Dict, List]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
