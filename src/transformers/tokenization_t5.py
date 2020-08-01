# coding=utf-8
# Copyright 2018 T5 Authors and HuggingFace Inc. team.
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
""" Tokenization class for model T5."""


import logging
import os
import re
from shutil import copyfile
from typing import List, Optional

from .tokenization_utils import BatchEncoding, PreTrainedTokenizer


logger = logging.getLogger(__name__)

SPIECE_UNDERLINE = "▁"

####################################################
# Mapping from the keyword arguments names of Tokenizer `__init__`
# to file names for serializing Tokenizer instances
####################################################
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

####################################################
# Mapping from the keyword arguments names of Tokenizer `__init__`
# to pretrained vocabulary URL for all the model shortcut names.
####################################################
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "t5-small": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-spiece.model",
        "t5-base": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-spiece.model",
        "t5-large": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-spiece.model",
        "t5-3b": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-spiece.model",
        "t5-11b": "https://s3.amazonaws.com/models.huggingface.co/bert/t5-spiece.model",
    }
}

####################################################
# Mapping from model shortcut names to max length of inputs
####################################################
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "t5-small": 512,
    "t5-base": 512,
    "t5-large": 512,
    "t5-3b": 512,
    "t5-11b": 512,
}


class T5Tokenizer(PreTrainedTokenizer):
    """
        Constructs a T5 tokenizer. Based on `SentencePiece <https://github.com/google/sentencepiece>`__ .

        This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
        should refer to the superclass for more information regarding methods.

        Args:
            vocab_file (:obj:`string`):
                `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a `.spm` extension) that
                contains the vocabulary necessary to instantiate a tokenizer.
            eos_token (:obj:`string`, `optional`, defaults to "</s>"):
                The end of sequence token.

                .. note::

                    When building a sequence using special tokens, this is not the token that is used for the end
                    of sequence. The token used is the :obj:`sep_token`.
            unk_token (:obj:`string`, `optional`, defaults to "<unk>"):
                The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
                token instead.
            pad_token (:obj:`string`, `optional`, defaults to "<pad>"):
                The token used for padding, for example when batching sequences of different lengths.
            extra_ids (:obj:`List[str]`, `optional`, defaults to :obj:`100`):
                Add a number of extra ids added to the end of the vocabulary for use as sentinels.
                These tokens are accessible as "<extra_id_{%d}>" where "{%d}" is a number between 0 and extra_ids-1.
                Extra tokens are indexed from the end of the vocabulary up to beginnning ("<extra_id_0>" is the last token in the vocabulary like in T5 preprocessing
                see: https://github.com/google-research/text-to-text-transfer-transformer/blob/9fd7b14a769417be33bc6c850f9598764913c833/t5/data/preprocessors.py#L2117)
            additional_special_tokens (:obj:`List[str]`, `optional`, defaults to :obj:`None`):
                Additional special tokens used by the tokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["attention_mask"]

    prefix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0:
            if additional_special_tokens is None:
                additional_special_tokens = []
            additional_special_tokens.extend(["<extra_id_{}>".format(i) for i in range(extra_ids)])

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use T5Tokenizer:"
                "https://github.com/google/sentencepiece"
                "pip install sentencepiece"
            )
            raise

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size() + self._extra_ids

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use T5Tokenizer: https://github.com/google/sentencepiece"
                "pip install sentencepiece"
            )
            raise
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text, sample=False):
        """ Take as input a string and return a list of strings (tokens) for words/sub-words
        """
        if not sample:
            pieces = self.sp_model.EncodeAsPieces(text)
        else:
            pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
        return pieces

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            token = "<extra_id_{}>".format(self.vocab_size - 1 - index)
        return token

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = self.sp_model.decode_pieces(tokens)
        return out_string

    def save_vocabulary(self, save_directory):
        """ Save the sentencepiece vocabulary (copy original file) and special tokens file
            to a directory.
        """
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        out_vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens. The special tokens depend on calling source text or target text.
        A T5 sequence has the following format, where ``X`` represents the sequence:
        - ``input_ids`` (for encoder) ``X [eos]``
        - ``decoder_input_ids``: (for decoder) ``[pad] X [eos]``
        Pairs of sequences are not the expected use case, but they will be handled without a separator.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        return_tensors: str = None,
        **kwargs,
    ) -> BatchEncoding:
        """Prepare a batch that can be passed directly to an instance of T5Model.
        
        Args:
            src_texts (:obj:`List[str]`):
                list of src texts
            tgt_texts (:obj:`List[str]`, `optional`):
                list of tgt texts
            max_length (:obj:`int`, `optional`):
                maximum length for the source text which defers to the config value of 512 for t5*
            max_target_length (:obj:`int`, `optional`):
                maximum length for the target text which defers to the config value of 512 for t5*
            padding (:obj:`str`, `optional`, defaults to "longest"):
                strategy for padding `input_ids` and `decoder_input_ids`. Should be "max_length" or "longest".
            return_tensors (:obj:`str`, `optional`):
                Can be set to ‘tf’, ‘pt’ or ‘np’ to return respectively TensorFlow `tf.constant`, PyTorch `torch.Tensor` or Numpy :oj: np.ndarray instead of a list of python integers.
            **kwargs:
                passed to self.__call__

        Returns:
            :class:`~transformers.BatchEncoding`: with keys input_ids, attention_mask, decoder_input_ids, decoder_attention_mask.
        """
        if max_length is None:
            max_length = self.max_len
        self.prefix_tokens = []
        model_inputs: BatchEncoding = self(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=True,
            **kwargs,
        )
        if tgt_texts is None:
            return model_inputs
        # Process tgt_texts
        if max_target_length is None:
            max_target_length = max_length
        # set prefix_tokens for target text
        self.prefix_tokens = [self.pad_token_id]
        decoder_inputs: BatchEncoding = self(
            tgt_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_target_length,
            truncation=True,
            **kwargs,
        )
        for k, v in decoder_inputs.items():
            model_inputs[f"decoder_{k}"] = v

        self.prefix_tokens = []
        return model_inputs

