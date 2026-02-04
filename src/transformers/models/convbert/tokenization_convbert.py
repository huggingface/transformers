# Copyright The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for ConvBERT."""

from ...models.bert.tokenization_bert import BertTokenizer


class ConvBertTokenizer(BertTokenizer):
    r"""
    Construct a ConvBERT tokenizer (backed by HuggingFace's tokenizers library). Based on WordPiece.

    This tokenizer inherits from [`BertTokenizer`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    """

    pass


__all__ = ["ConvBertTokenizer"]
