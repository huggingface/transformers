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

# 토크나이저 [[tokenizer]]

토크나이저는 모델에 대한 입력을 준비하는 역할을 담당합니다. 이 라이브러리는 모든 모델에 대한 토크나이저를 포함합니다. 
대부분의 토크나이저는 두 가지 버전으로 제공됩니다: 파이썬을 이용한 전체 구현 및 
Rust 라이브러리 [🤗 Tokenizers](https://github.com/huggingface/tokenizers)를 기반으로 한 "빠른" 구현이 있습니다. "빠른" 구현은 다음을 가능하게 합니다:

1. 일괄 토큰화 처리를 할 때 (특히) 상당한 속도 향상과
2. 원본 문자열(문자 및 단어)과 토큰 공간 사이의 매핑을 위한 추가적인 방법
  (예: 주어진 문자를 포함하는 토큰의 인덱스 또는 주어진 토큰에 해당하는 문자의 범위)

기본 클래스 [`PreTrainedTokenizer`]와 [`PreTrainedTokenizerFast`]는 
모델 입력(아래를 참조하세요)에서 문자열 입력을 인코딩하는 공통 메서드를 구현하고, 
로컬 파일 또는 디렉터리 또는 라이브러리에서 제공된 사전 훈련된 토크나이저(허깅페이스의 AWS S3 저장소에서 다운로드됨)에서 
파이썬 및 "빠른" 토크나이저를 인스턴스화하고 저장합니다. 
둘 다 공통 메서드가 포함된 [`~tokenization_utils_base.PreTrainedTokenizerBase`]와 
[`~tokenization_utils_base.SpecialTokensMixin`]에 의존합니다.

따라서 [`PreTrainedTokenizer`]와 [`PreTrainedTokenizerFast`]는 
모든 토크나이저를 사용하기 위한 주요 메서드를 구현합니다:

- 토크나이징(즉, 하위 단어 토큰 문자열 분할), 토큰 문자열을 id로 변환하고 다시 변환, 
  인코딩/디코딩(예를 들어 토크나이징 및 정수로 변환).
- 기본 구조(BPE, SentencePiece 등)와 독립적인 방법으로 어휘에 새 토큰 추가
- 특수 토큰(마스크, 문장의 시작 등) 관리: 
  추가, 토크나이저의 속성에 할당하여 쉽게 접근, 토큰화 중에 분할되지 않도록 힘

[`BatchEncoding`]은 
[`~tokenization_utils_base.PreTrainedTokenizerBase`]의 인코딩 메서드(`__call__`, `encode_plus`, `batch_encode_plus`)의 출력을 저장하고, 
파이썬 딕셔너리에서 파생됩니다. 
토크나이저가 순수 파이썬 토크나이저인 경우, 이 클래스는 표준 파이썬 딕셔너리처럼 동작하며, 
`input_ids`, `attention_mask` 등과 같이 계산된 다양한 모델 입력을 보유합니다.
토크나이저가 "빠른" 토크나이저(즉, 허깅페이스 [tokenizers library](https://github.com/huggingface/tokenizers)에 의해 백업된)인 경우, 
이 클래스는 원본 문자열(문자 및 단어)과 
토큰 공간(예를 들어 주어진 문자를 포함하는 토큰의 인덱스 확인 또는 주어진 토큰에 해당하는 문자의 범위) 간의 매핑을 위한 
몇 가지 고급 정렬 메서드를 추가로 제공합니다.


## PreTrainedTokenizer [[pretrainedtokenizer]]

[[autodoc]] PreTrainedTokenizer
    - __call__
    - batch_decode
    - decode
    - encode
    - push_to_hub
    - all

## PreTrainedTokenizerFast [[pretrainedtokenizerfast]]

[`PreTrainedTokenizerFast`]는 [tokenizers](https://huggingface.co/docs/tokenizers) 라이브러리에 의존합니다. 
🤗 tokenizers 라이브러리에서 얻은 토크나이저는 🤗 transformers에 매우 간단하게 로드할 수 있습니다. 이 작업을 수행하는 방법을 알아보려면 [Using tokenizers from 🤗 tokenizers](../fast_tokenizers) 페이지를 참조하세요.

[[autodoc]] PreTrainedTokenizerFast
    - __call__
    - batch_decode
    - decode
    - encode
    - push_to_hub
    - all

## BatchEncoding [[batchencoding]]

[[autodoc]] BatchEncoding
