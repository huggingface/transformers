# coding=utf-8
# Copyright 2019-present CNRS, Facebook Inc. and the HuggingFace Inc. team.
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
"""Tokenization classes for Flaubert, based on XLM."""


import logging
import unicodedata

import six

from .tokenization_xlm import XLMTokenizer


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "flaubert-small-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_small_cased/vocab.json",
        "flaubert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_base_uncased/vocab.json",
        "flaubert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_base_cased/vocab.json",
        "flaubert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_large_cased/vocab.json",
    },
    "merges_file": {
        "flaubert-small-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_small_cased/merges.txt",
        "flaubert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_base_uncased/merges.txt",
        "flaubert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_base_cased/merges.txt",
        "flaubert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/flaubert/flaubert_large_cased/merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "flaubert-small-cased": 512,
    "flaubert-base-uncased": 512,
    "flaubert-base-cased": 512,
    "flaubert-large-cased": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "flaubert-small-cased": {"do_lowercase": False},
    "flaubert-base-uncased": {"do_lowercase": True},
    "flaubert-base-cased": {"do_lowercase": False},
    "flaubert-large-cased": {"do_lowercase": False},
}


def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming UTF-8 input.
    """
    # six_ensure_text is copied from https://github.com/benjaminp/six
    def six_ensure_text(s, encoding="utf-8", errors="strict"):
        if isinstance(s, six.binary_type):
            return s.decode(encoding, errors)
        elif isinstance(s, six.text_type):
            return s
        else:
            raise TypeError("not expecting type '%s'" % type(s))

    return six_ensure_text(text, encoding="utf-8", errors="ignore")


class FlaubertTokenizer(XLMTokenizer):
    """
    BPE tokenizer for Flaubert

    - Moses preprocessing & tokenization
    - Normalize all inputs text
    - argument ``special_tokens`` and function ``set_special_tokens``, can be used to add additional symbols \
      (ex: "__classify__") to a vocabulary
    - `do_lowercase` controle lower casing (automatically set for pretrained vocabularies)

    This tokenizer inherits from :class:`~transformers.XLMTokenizer`. Please check the superclass for usage examples
    and documentation regarding arguments.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, do_lowercase=False, **kwargs):
        super().__init__(**kwargs)
        self.do_lowercase = do_lowercase
        self.do_lowercase_and_remove_accent = False

    def preprocess_text(self, text):
        text = text.replace("``", '"').replace("''", '"')
        text = convert_to_unicode(text)
        text = unicodedata.normalize("NFC", text)

        if self.do_lowercase:
            text = text.lower()

        return text

    def _tokenize(self, text, bypass_tokenizer=False):
        """
        Tokenize a string given language code using Moses.

        Details of tokenization:
        - [sacremoses](https://github.com/alvations/sacremoses): port of Moses
            - Install with `pip install sacremoses`

        Args:
            - bypass_tokenizer: Allow users to preprocess and tokenize the sentences externally (default = False)  (bool). If True, we only apply BPE.

        Returns:
            List of tokens.
        """
        lang = "fr"
        if lang and self.lang2id and lang not in self.lang2id:
            logger.error(
                "Supplied language code not found in lang2id mapping. Please check that your language is supported by the loaded pretrained model."
            )

        if bypass_tokenizer:
            text = text.split()
        else:
            text = self.preprocess_text(text)
            text = self.moses_pipeline(text, lang=lang)
            text = self.moses_tokenize(text, lang=lang)

        split_tokens = []
        for token in text:
            if token:
                split_tokens.extend([t for t in self.bpe(token).split(" ")])

        return split_tokens
