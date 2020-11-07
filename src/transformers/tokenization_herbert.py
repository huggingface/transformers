# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Allegro.pl, Facebook Inc. and the HuggingFace Inc. team.
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

from .tokenization_bert import BasicTokenizer
from .tokenization_xlm import XLMTokenizer
from .utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"allegro/herbert-base-cased": "https://cdn.huggingface.co/allegro/herbert-base-cased/vocab.json"},
    "merges_file": {"allegro/herbert-base-cased": "https://cdn.huggingface.co/allegro/herbert-base-cased/merges.txt"},
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"allegro/herbert-base-cased": 514}
PRETRAINED_INIT_CONFIGURATION = {}


class HerbertTokenizer(XLMTokenizer):
    """
    Construct a BPE tokenizer for HerBERT.

    Peculiarities:

    - uses BERT's pre-tokenizer: BaseTokenizer splits tokens on spaces, and also on punctuation. Each occurrence of a
      punctuation character will be treated separately.

    - Such pretokenized input is BPE subtokenized

    This tokenizer inherits from :class:`~transformers.XLMTokenizer` which contains most of the methods. Users should
    refer to the superclass for more information regarding methods.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, **kwargs):

        kwargs["cls_token"] = "<s>"
        kwargs["unk_token"] = "<unk>"
        kwargs["pad_token"] = "<pad>"
        kwargs["mask_token"] = "<mask>"
        kwargs["sep_token"] = "</s>"
        kwargs["do_lowercase_and_remove_accent"] = False
        kwargs["additional_special_tokens"] = []

        super().__init__(**kwargs)
        self.bert_pre_tokenizer = BasicTokenizer(
            do_lower_case=False, never_split=self.all_special_tokens, tokenize_chinese_chars=False, strip_accents=False
        )

    def _tokenize(self, text):

        pre_tokens = self.bert_pre_tokenizer.tokenize(text)

        split_tokens = []
        for token in pre_tokens:
            if token:
                split_tokens.extend([t for t in self.bpe(token).split(" ")])

        return split_tokens
