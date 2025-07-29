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

# 토크나이저[[tokenizer]]

토크나이저는 모델의 입력을 준비하는 역할을 담당합니다. 이 라이브러리에는 모든 모델을 위한 토크나이저가 포함되어 있습니다. 대부분의 토크나이저는 두 가지 버전으로 제공됩니다. 완전한 파이썬 구현과 Rust 라이브러리 [🤗 Tokenizers](https://github.com/huggingface/tokenizers)에 기반한 "Fast" 구현입니다. "Fast" 구현은 다음을 가능하게 합니다:

1. 특히 배치 토큰화를 수행할 때 속도가 크게 향상됩니다.
2. 원본 문자열(문자 및 단어)과 토큰 공간 사이를 매핑하는 추가적인 메소드를 제공합니다. (예: 특정 문자를 포함하는 토큰의 인덱스를 얻거나, 특정 토큰에 해당하는 문자 범위를 가져오는 등).

기본 클래스인 [`PreTrainedTokenizer`]와 [`PreTrainedTokenizerFast`]는 문자열 입력을 인코딩하는 메소드를 구현하며(아래 참조), 로컬 파일이나 디렉토리, 또는 라이브러리에서 제공하는 사전 훈련된 토크나이저(HuggingFace의 AWS S3 저장소에서 다운로드된)로부터 파이썬 및 "Fast" 토크나이저를 인스턴스화하거나 저장하는 기능을 제공합니다. 이 두 클래스는 공통 메소드를 포함하는 [`~tokenization_utils_base.PreTrainedTokenizerBase`]와 [`~tokenization_utils_base.SpecialTokensMixin`]에 의존합니다.

[`PreTrainedTokenizer`]와 [`PreTrainedTokenizerFast`]는 모든 토크나이저에서 사용되는 주요 메소드들을 구현합니다:

- 토큰화(문자열을 하위 단어 토큰 문자열로 분할), 토큰 문자열을 ID로 변환 및 그 반대 과정, 그리고 인코딩/디코딩(즉, 토큰화 및 정수로 변환)을 수행합니다.
- 구조(BPE, SentencePiece 등)에 구애받지 않고 어휘에 새로운 토큰을 추가합니다.
- 특수 토큰(마스크, 문장 시작 등) 관리: 토큰을 추가하고, 쉽게 접근할 수 있도록 토크나이저의 속성에 할당하며, 토큰화 과정에서 분리되지 않도록 보장합니다.

[`BatchEncoding`]은 [`~tokenization_utils_base.PreTrainedTokenizerBase`]의 인코딩 메소드(`__call__`, `encode_plus`, `batch_encode_plus`)의 출력을 담고 있으며, 파이썬 딕셔너리를 상속받습니다. 토크나이저가 순수 파이썬 토크나이저인 경우 이 클래스는 표준 파이썬 딕셔너리처럼 동작하며, 이러한 메소드들로 계산된 다양한 모델 입력(`input_ids`, `attention_mask` 등)을 갖습니다. 토크나이저가 "Fast" 토크나이저일 경우(즉, HuggingFace [tokenizers 라이브러리](https://github.com/huggingface/tokenizers) 기반일 경우), 이 클래스는 추가적으로 원본 문자열(문자 및 단어)과 토큰 공간 사이를 매핑하는 데 사용할 수 있는 여러 고급 정렬 메소드를 제공합니다 (예: 특정 문자를 포함하는 토큰의 인덱스를 얻거나, 특정 토큰에 해당하는 문자 범위를 얻는 등).


# 멀티모달 토크나이저[[multimodal-tokenizer]]

그 외에도 각 토크나이저는 "멀티모달" 토크나이저가 될 수 있으며, 이는 토크나이저가 모든 관련 특수 토큰을 토크나이저 속성의 일부로 저장하여 더 쉽게 접근할 수 있도록 한다는 것을 의미합니다. 예를 들어, LLaVA와 같은 비전-언어 모델에서 토크나이저를 가져오면, `tokenizer.image_token_id`에 접근하여 플레이스홀더로 사용되는 특수 이미지 토큰을 얻을 수 있습니다.

모든 유형의 토크나이저에 추가 특수 토큰을 활성화하려면, 다음 코드를 추가하고 토크나이저를 저장해야 합니다. 추가 특수 토큰은 반드시 특정 모달리티와 관련될 필요는 없으며, 모델이 자주 접근해야 하는 어떤 것이든 될 수 있습니다. 아래 코드에서 `output_dir`에 저장된 토크나이저는 세 개의 추가 특수 토큰에 직접 접근할 수 있게 됩니다.

```python
vision_tokenizer = AutoTokenizer.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    extra_special_tokens={"image_token": "<image>", "boi_token": "<image_start>", "eoi_token": "<image_end>"}
)
print(vision_tokenizer.image_token, vision_tokenizer.image_token_id)
("<image>", 32000)
```

## PreTrainedTokenizer[[transformers.PreTrainedTokenizer]]

[[autodoc]] PreTrainedTokenizer
    - __call__
    - add_tokens
    - add_special_tokens
    - apply_chat_template
    - batch_decode
    - decode
    - encode
    - push_to_hub
    - all


## PreTrainedTokenizerFast[[transformers.PreTrainedTokenizerFast]]

[`PreTrainedTokenizerFast`]는 [tokenizers](https://huggingface.co/docs/tokenizers) 라이브러리에 의존합니다. 🤗 tokenizers 라이브러리에서 얻은 토크나이저는
🤗 transformers로 매우 간단하게 가져올 수 있습니다. 어떻게 하는지 알아보려면 [Using tokenizers from 🤗 tokenizers](../fast_tokenizers) 페이지를 참고하세요.

[[autodoc]] PreTrainedTokenizerFast
    - __call__
    - add_tokens
    - add_special_tokens
    - apply_chat_template
    - batch_decode
    - decode
    - encode
    - push_to_hub
    - all

## BatchEncoding[[transformers.BatchEncoding]]

[[autodoc]] BatchEncoding