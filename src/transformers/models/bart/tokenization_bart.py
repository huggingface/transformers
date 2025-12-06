# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Compatibility shims for BART tokenizers in v5.

In v5 we consolidate on the tokenizers-library backend and remove separate
"slow" vs "fast" implementations. BART uses the same byte-level BPE
tokenizer as RoBERTa, so we expose `BartTokenizer` and `BartTokenizerFast`
as aliases to `RobertaTokenizer` to preserve the public API expected by
existing code and tests.
"""

from ..roberta.tokenization_roberta import RobertaTokenizer as _RobertaTokenizer


# Public aliases maintained for backwards compatibility
BartTokenizer = _RobertaTokenizer
BartTokenizerFast = _RobertaTokenizer

__all__ = ["BartTokenizer", "BartTokenizerFast"]
