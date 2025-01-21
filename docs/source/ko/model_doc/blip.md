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

# BLIP[[blip]]

## 개요[[overview]]

BLIP 모델은 Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi의 [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086) 논문에서 제안되었습니다.

BLIP은 여러 멀티모달 작업을 수행할 수 있는 모델입니다:

- 시각 질문 응답 (Visual Question Answering, VQA)
- 이미지-텍스트 검색 (이미지-텍스트 매칭)
- 이미지 캡셔닝

논문의 초록은 다음과 같습니다:

*비전-언어 사전 학습(Vision-Language Pre-training, VLP)은 다양한 비전-언어 작업의 성능을 크게 향상시켰습니다. 하지만, 대부분의 기존 사전 학습 모델들은 이해 기반 작업이나 생성 기반 작업 중 하나에서만 뛰어난 성능을 발휘합니다. 또한 성능 향상은 주로 웹에서 수집한 노이즈가 많은 이미지-텍스트 쌍으로 데이터셋의 규모를 키우는 방식으로 이루어졌는데, 이는 최적의 지도 학습 방식이라고 보기 어렵습니다. 본 논문에서는 BLIP이라는 새로운 VLP 프레임워크를 제안합니다. 이 프레임워크는 비전-언어 이해 및 생성 작업 모두에 유연하게 적용될 수 있습니다. BLIP는 캡셔너가 합성 캡션을 생성하고 필터가 노이즈 캡션을 제거하는 부트스트래핑 방법을 통해 웹 데이터의 노이즈를 효과적으로 활용합니다. 우리는 이미지-텍스트 검색(Recall@1에서 +2.7%), 이미지 캡셔닝(CIDEr에서 +2.8%), 그리고 VQA(VQA 점수에서 +1.6%)와 같은 다양한 비전-언어 작업에서 최신 성과를 달성했습니다. 또한 BLIP은 제로샷 방식으로 비디오-언어 작업에 직접 전이될 때도 강력한 일반화 능력을 보여줍니다. 이 논문의 코드, 모델, 데이터셋은 공개되었습니다.*

![BLIP.gif](https://cdn-uploads.huggingface.co/production/uploads/1670928184033-62441d1d9fdefb55a0b7d12c.gif)

이 모델은 [ybelkada](https://huggingface.co/ybelkada)가 기여했습니다.
원본 코드는 [여기](https://github.com/salesforce/BLIP)에서 찾을 수 있습니다.

## 자료[[resources]]

- [Jupyter notebook](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_blip.ipynb): 사용자 정의 데이터셋에서 BLIP를 이미지 캡셔닝으로 미세 조정하는 방법

## BlipConfig[[transformers.BlipConfig]]

[[autodoc]] BlipConfig
    - from_text_vision_configs

## BlipTextConfig[[transformers.BlipTextConfig]]

[[autodoc]] BlipTextConfig

## BlipVisionConfig[[transformers.BlipVisionConfig]]

[[autodoc]] BlipVisionConfig

## BlipProcessor[[transformers.BlipProcessor]]

[[autodoc]] BlipProcessor

## BlipImageProcessor[[transformers.BlipImageProcessor]]

[[autodoc]] BlipImageProcessor
    - preprocess

<frameworkcontent>
<pt>

## BlipModel[[transformers.BlipModel]]

`BlipModel`은 향후 버전에서 더 이상 지원되지 않을 예정입니다. 목적에 따라 `BlipForConditionalGeneration`, `BlipForImageTextRetrieval` 또는 `BlipForQuestionAnswering`을 사용하십시오.

[[autodoc]] BlipModel
    - forward
    - get_text_features
    - get_image_features

## BlipTextModel[[transformers.BlipTextModel]]

[[autodoc]] BlipTextModel
    - forward

## BlipVisionModel[[transformers.BlipVisionModel]]

[[autodoc]] BlipVisionModel
    - forward

## BlipForConditionalGeneration[[transformers.BlipForConditionalGeneration]]

[[autodoc]] BlipForConditionalGeneration
    - forward

## BlipForImageTextRetrieval[[transformers.BlipForImageTextRetrieval]]

[[autodoc]] BlipForImageTextRetrieval
    - forward

## BlipForQuestionAnswering[[transformers.BlipForQuestionAnswering]]

[[autodoc]] BlipForQuestionAnswering
    - forward

</pt>
<tf>

## TFBlipModel[[transformers.TFBlipModel]]

[[autodoc]] TFBlipModel
    - call
    - get_text_features
    - get_image_features

## TFBlipTextModel[[transformers.TFBlipTextModel]]

[[autodoc]] TFBlipTextModel
    - call

## TFBlipVisionModel[[transformers.TFBlipVisionModel]]

[[autodoc]] TFBlipVisionModel
    - call

## TFBlipForConditionalGeneration[[transformers.TFBlipForConditionalGeneration]]

[[autodoc]] TFBlipForConditionalGeneration
    - call

## TFBlipForImageTextRetrieval[[transformers.TFBlipForImageTextRetrieval]]

[[autodoc]] TFBlipForImageTextRetrieval
    - call

## TFBlipForQuestionAnswering[[transformers.TFBlipForQuestionAnswering]]

[[autodoc]] TFBlipForQuestionAnswering
    - call
</tf>
</frameworkcontent>
