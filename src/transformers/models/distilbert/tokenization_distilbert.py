# Copyright 2018 The HuggingFace Inc. team.
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
"""Tokenization classes for DistilBERT."""

from ...models.bert.tokenization_bert import BertTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}


class DistilBertTokenizer(BertTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, *args, do_lower_case: bool = True, **kwargs):
        """
        Construct a DistilBERT tokenizer (backed by HuggingFace's tokenizers library). Based on WordPiece.

        This tokenizer inherits from [`BertTokenizer`] which contains most of the main methods. Users should refer to
        this superclass for more information regarding those methods.

        Args:
            do_lower_case (`bool`, *optional*, defaults to `True`):
                Whether or not to lowercase the input when tokenizing.
        """
        super().__init__(*args, do_lower_case=do_lower_case, **kwargs)


# DistilBertTokenizerFast is an alias for DistilBertTokenizer (since BertTokenizer is already a fast tokenizer)
DistilBertTokenizerFast = DistilBertTokenizer

__all__ = ["DistilBertTokenizer", "DistilBertTokenizerFast"]
