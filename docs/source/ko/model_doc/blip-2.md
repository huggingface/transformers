<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# BLIP-2[[blip-2]]

## 개요[[overview]]
BLIP-2 모델은 Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi의 [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) 논문에서 제안되었습니다. BLIP-2는 동결된 사전 학습 이미지 인코더와 대규모 언어 모델(LLM)을 연결하는 12층의 경량 Transformer 인코더를 학습시켜, 여러 비전-언어 작업에서 SOTA(현재 최고의 성능)을 달성했습니다. 특히, BLIP-2는 800억 개의 파라미터를 가진 Flamingo 모델보다 제로샷 VQAv2에서 8.7% 더 높은 성능을 기록했으며, 학습 가능한 파라미터 수는 Flamingo보다 54배 적습니다.

논문의 초록은 다음과 같습니다:

*비전-언어 사전 학습의 비용은 대규모 모델의 엔드-투-엔드 학습으로 인해 점점 더 부담스러워지고 있습니다. 본 논문은 사전 학습된 이미지 인코더와 대규모 언어 모델을 활용하여 비전-언어 사전 학습을 부트스트래핑하는 일반적이고 효율적인 사전 학습 전략인 BLIP-2를 제안합니다. BLIP-2는 경량화된 Querying Transformer를 통해 모달리티 간의 차이를 연결하며, 두 단계로 사전 학습됩니다. 첫 번째 단계는 동결된 이미지 인코더로부터 비전-언어 표현 학습을 부트스트래핑하고, 두 번째 단계는 동결된 언어 모델로부터 비전-언어 생성 학습을 부트스트래핑합니다. BLIP-2는 기존 방법들에 비해 훨씬 적은 학습 가능한 파라미터로 다양한 비전-언어 작업에서 최첨단 성능을 달성합니다. 예를 들어, 우리 모델은 제로샷 VQAv2에서 Flamingo80B보다 8.7% 높은 성능을 기록하며, 학습 가능한 파라미터 수는 54배 적습니다. 우리는 또한 자연어 명령을 따를 수 있는 제로샷 이미지-텍스트 생성의 새로운 기능을 입증했습니다.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/blip2_architecture.jpg"
alt="drawing" width="600"/> 

<small> BLIP-2 구조. <a href="https://arxiv.org/abs/2301.12597">원본 논문</a> 에서 발췌. </small>

이 모델은 [nielsr](https://huggingface.co/nielsr)가 기여했습니다. 원본 코드는 [여기](https://github.com/salesforce/LAVIS/tree/5ee63d688ba4cebff63acee04adaef2dee9af207)에서 확인할 수 있습니다.

## 사용 팁[[usage-tips]]

- BLIP-2는 이미지와 조건에 따라 텍스트 프롬프트를 입력받아 조건부 텍스트를 생성합니다. 추론 시 [`generate`] 메소드를 사용하는 것이 권장됩니다.
- [`Blip2Processor`]를 사용하여 모델에 이미지를 준비하고, 예측된 토큰 ID를 텍스트로 디코딩할 수 있습니다.

## 자료[[resources]]

BLIP-2를 시작하는 데 도움이 되는 공식 Hugging Face 및 커뮤니티(🌎 표시) 자료 목록입니다.

- 이미지 캡셔닝, 시각 질문 응답(VQA), 채팅과 같은 대화형 작업을 위한 BLIP-2 데모 노트북은 [여기](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/BLIP-2)에서 찾을 수 있습니다.

리소스를 제출하여 여기에 포함하고 싶다면 언제든지 풀 리퀘스트를 열어주세요! 리소스는 기존 리소스를 복제하지 않고 새로운 내용이어야 합니다.

## Blip2Config[[transformers.Blip2Config]]

[[autodoc]] Blip2Config
    - from_vision_qformer_text_configs

## Blip2VisionConfig[[transformers.Blip2VisionConfig]]

[[autodoc]] Blip2VisionConfig

## Blip2QFormerConfig[[transformers.Blip2QFormerConfig]]

[[autodoc]] Blip2QFormerConfig

## Blip2Processor[[transformers.Blip2Processor]]

[[autodoc]] Blip2Processor

## Blip2VisionModel[[transformers.Blip2VisionModel]]

[[autodoc]] Blip2VisionModel
    - forward

## Blip2QFormerModel[[transformers.Blip2QFormerModel]]

[[autodoc]] Blip2QFormerModel
    - forward

## Blip2Model[[transformers.Blip2Model]]

[[autodoc]] Blip2Model
    - forward
    - get_text_features
    - get_image_features
    - get_qformer_features

## Blip2ForConditionalGeneration[[transformers.Blip2ForConditionalGeneration]]

[[autodoc]] Blip2ForConditionalGeneration
    - forward
    - generate

## Blip2ForImageTextRetrieval[[transformers.Blip2ForImageTextRetrieval]]

[[autodoc]] Blip2ForImageTextRetrieval
    - forward

## Blip2TextModelWithProjection[[transformers.Blip2TextModelWithProjection]]

[[autodoc]] Blip2TextModelWithProjection

## Blip2VisionModelWithProjection[[transformers.Blip2VisionModelWithProjection]]

[[autodoc]] Blip2VisionModelWithProjection
