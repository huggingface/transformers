<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Utilities for Tokenizers

このページには、トークナイザーによって使用されるすべてのユーティリティ関数 (主にクラス) がリストされます。
[`~tokenization_utils_base.PreTrainedTokenizerBase`] 間の共通メソッドを実装します。
[`PreTrainedTokenizer`] と [`PreTrainedTokenizerFast`] およびミックスイン
[`~tokenization_utils_base.SpecialTokensMixin`]。

これらのほとんどは、ライブラリ内のトークナイザーのコードを学習する場合にのみ役に立ちます。

## PreTrainedTokenizerBase

[[autodoc]] tokenization_utils_base.PreTrainedTokenizerBase
    - __call__
    - all

## SpecialTokensMixin

[[autodoc]] tokenization_utils_base.SpecialTokensMixin

## Enums and namedtuples

[[autodoc]] tokenization_utils_base.TruncationStrategy

[[autodoc]] tokenization_utils_base.CharSpan

[[autodoc]] tokenization_utils_base.TokenSpan
