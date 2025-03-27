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

# 토크나이저를 위한 유틸리티 [[utilities-for-tokenizers]]

이 페이지는 토크나이저에서 사용되는 모든 유틸리티 함수들을 나열하며, 주로 [`PreTrainedTokenizer`]와 [`PreTrainedTokenizerFast`] 사이의 공통 메소드를 구현하는 [`~tokenization_utils_base.PreTrainedTokenizerBase`] 클래스와 [`~tokenization_utils_base.SpecialTokensMixin`]을 다룹니다.

이 함수들 대부분은 라이브러리의 토크나이저 코드를 연구할 때만 유용합니다.

## PreTrainedTokenizerBase [[transformers.PreTrainedTokenizerBase]]

[[autodoc]] tokenization_utils_base.PreTrainedTokenizerBase
   - __call__
   - all

## SpecialTokensMixin [[transformers.SpecialTokensMixin]]

[[autodoc]] tokenization_utils_base.SpecialTokensMixin

## Enums 및 namedtuples [[transformers.tokenization_utils_base.TruncationStrategy]]

[[autodoc]] tokenization_utils_base.TruncationStrategy

[[autodoc]] tokenization_utils_base.CharSpan

[[autodoc]] tokenization_utils_base.TokenSpan
