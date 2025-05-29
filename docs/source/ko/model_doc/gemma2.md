
<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Gemma2 [[gemma2]]

## 개요 [[overview]]

Gemma2 모델은 Google의 Gemma2 팀이 작성한 [Gemma2: Open Models Based on Gemini Technology and Research](https://blog.google/technology/developers/google-gemma-2/)에서 제안되었습니다.
파라미터 크기가 각각 90억(9B)과 270억(27B)인 두 가지 Gemma2 모델이 출시되었습니다.

블로그 게시물의 초록은 다음과 같습니다:

*이제 우리는 전 세계의 연구자와 개발자들에게 Gemma 2를 공식적으로 출시합니다. 90억(9B)과 270억(27B) 파라미터 크기로 제공되는 Gemma 2는 1세대보다 더 높은 성능과 추론 효율성을 제공하며, 상당한 안전성 향상을 포함하고 있습니다. 사실 270억 규모의 모델은 크기가 두 배 이상인 모델과 비교해도 경쟁력 있는 대안을 제공하며, 이는 작년 12월까지만 해도 독점 모델에서만 가능했던 성능을 제공합니다.*

팁:

- 원본 체크포인트는 변환 스크립트 `src/transformers/models/Gemma2/convert_Gemma2_weights_to_hf.py`를 사용하여 변환할 수 있습니다.

<Tip warning={true}>

- Gemma2는 매 두 번째 레이어마다 슬라이딩 윈도우 어텐션을 사용하므로 [`~DynamicCache`] 또는 텐서의 튜플과 같은 일반적인 kv 캐싱에는 적합하지 않습니다. Gemma2의 forward 호출에서 캐싱을 활성화하려면 [`~HybridCache`] 인스턴스를 초기화하고 이를 `past_key_values`로 forward 호출에 전달해야 합니다. 또한 `past_key_values`에 이미 이전의 키와 값이 포함되어 있다면 `cache_position`도 준비해야 합니다.

</Tip>

이 모델은 [Arthur Zucker](https://huggingface.co/ArthurZ), [Pedro Cuenca](https://huggingface.co/pcuenq), [Tom Arsen]()이 기여했습니다.

## Gemma2Config [[transformers.Gemma2Config]]

[[autodoc]] Gemma2Config

## Gemma2Model [[transformers.Gemma2Model]]

[[autodoc]] Gemma2Model
    - forward

## Gemma2ForCausalLM [[transformers.Gemma2ForCausalLM]]

[[autodoc]] Gemma2ForCausalLM
    - forward

## Gemma2ForSequenceClassification [[transformers.Gemma2ForSequenceClassification]]

[[autodoc]] Gemma2ForSequenceClassification
    - forward

## Gemma2ForTokenClassification [[transformers.Gemma2ForTokenClassification]]

[[autodoc]] Gemma2ForTokenClassification
    - forward