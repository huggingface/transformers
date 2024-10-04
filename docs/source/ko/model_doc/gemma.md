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

# Gemma [[gemma]]

## 개요 [[overview]]

Gemma 모델은 Google의 Gemma 팀이 작성한 [Gemma: Open Models Based on Gemini Technology and Research](https://blog.google/technology/developers/gemma-open-models/)에서 제안되었습니다.

Gemma 모델은 6조 토큰으로 학습되었으며, 2b와 7b의 두 가지 버전으로 출시되었습니다.

논문의 초록은 다음과 같습니다:

*이 연구는 언어 이해, 추론 및 안전성에 대한 학술 벤치마크에서 뛰어난 성능을 보이는 새로운 오픈 언어 모델 계열인 Gemma를 소개합니다. 우리는 두 가지 크기(20억 및 70억 매개변수)의 모델을 출시하며, 사전 학습된 체크포인트와 미세 조정된 체크포인트를 모두 제공합니다. Gemma는 18개의 텍스트 기반 작업 중 11개에서 유사한 크기의 오픈 모델을 능가하며, 우리는 모델 개발에 대한 상세한 설명과 함께 안전성과 책임 측면에 대한 종합적인 평가를 제공합니다. 우리는 LLM의 책임감 있는 공개가 최첨단 모델의 안전성을 향상시키고 다음 세대의 LLM 혁신을 가능하게 하는 데 중요하다고 믿습니다.*

팁:

- 원본 체크포인트는 변환 스크립트 `src/transformers/models/gemma/convert_gemma_weights_to_hf.py`를 사용하여 변환할 수 있습니다.

이 모델은 [Arthur Zucker](https://huggingface.co/ArthurZ), [Younes Belkada](https://huggingface.co/ybelkada), [Sanchit Gandhi](https://huggingface.co/sanchit-gandhi), [Pedro Cuenca](https://huggingface.co/pcuenq)가 기여했습니다.

## GemmaConfig [[transformers.GemmaConfig]]

[[autodoc]] GemmaConfig

## GemmaTokenizer [[transformers.GemmaTokenizer]]

[[autodoc]] GemmaTokenizer


## GemmaTokenizerFast [[transformers.GemmaTokenizerFast]]

[[autodoc]] GemmaTokenizerFast

## GemmaModel [[transformers.GemmaModel]]

[[autodoc]] GemmaModel
    - forward

## GemmaForCausalLM [[transformers.GemmaForCausalLM]]

[[autodoc]] GemmaForCausalLM
    - forward

## GemmaForSequenceClassification [[transformers.GemmaForSequenceClassification]]

[[autodoc]] GemmaForSequenceClassification
    - forward

## GemmaForTokenClassification [[transformers.GemmaForTokenClassification]]

[[autodoc]] GemmaForTokenClassification
    - forward

## FlaxGemmaModel [[transformers.FlaxGemmaModel]]

[[autodoc]] FlaxGemmaModel
    - __call__

## FlaxGemmaForCausalLM [[transformers.FlaxGemmaForCausalLM]]

[[autodoc]] FlaxGemmaForCausalLM
    - __call__
