<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Swin Transformer [[swin-transformer]]

## 개요 [[overview]]

Swin Transformer는 Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo가 제안한 논문 [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://huggingface.co/papers/2103.14030)에서 소개되었습니다.

논문의 초록은 다음과 같습니다:

*이 논문은 Swin Transformer라는 새로운 비전 트랜스포머를 소개합니다. 이 모델은 컴퓨터 비전에서 범용 백본(backbone)으로 사용될 수 있습니다. 트랜스포머를 언어에서 비전으로 적용할 때의 어려움은 두 분야 간의 차이에서 비롯되는데, 예를 들어 시각적 객체의 크기가 크게 변동하며, 이미지의 픽셀 해상도가 텍스트의 단어에 비해 매우 높다는 점이 있습니다. 이러한 차이를 해결하기 위해, 우리는 'Shifted Windows'를 이용해 표현을 계산하는 계층적 트랜스포머를 제안합니다. Shifted Windows 방식은 겹치지 않는 로컬 윈도우에서 self-attention 계산을 제한하여 효율성을 높이는 동시에 윈도우 간 연결을 가능하게 합니다. 이 계층적 구조는 다양한 크기의 패턴을 모델링할 수 있는 유연성을 제공하며, 이미지 크기에 비례한 선형 계산 복잡성을 가지고 있습니다. Swin Transformer의 이러한 특징들은 이미지 분류(Imagenet-1K에서 87.3의 top-1 정확도) 및 객체 검출(COCO test-dev에서 58.7의 박스 AP, 51.1의 마스크 AP)과 같은 밀집 예측 작업, 의미적 분할(ADE20K val에서 53.5의 mIoU)과 같은 광범위한 비전 작업에 적합합니다. 이 모델은 COCO에서 이전 최고 성능을 박스 AP에서 +2.7, 마스크 AP에서 +2.6, ADE20K에서 mIoU에서 +3.2를 초과하는 성과를 보여주며, 트랜스포머 기반 모델이 비전 백본으로서의 잠재력을 입증했습니다. 계층적 설계와 Shifted Windows 방식은 순수 MLP 아키텍처에도 유리하게 작용합니다.* 

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/swin_transformer_architecture.png"
alt="drawing" width="600"/>

<small> Swin Transformer 아키텍처. <a href="https://huggingface.co/papers/2102.03334">원본 논문</a>에서 발췌.</small>

이 모델은 [novice03](https://huggingface.co/novice03)이 기여하였습니다. Tensorflow 버전은 [amyeroberts](https://huggingface.co/amyeroberts)가 기여했습니다. 원본 코드는 [여기](https://github.com/microsoft/Swin-Transformer)에서 확인할 수 있습니다.

## 사용 팁 [[usage-tips]]

- Swin은 입력의 높이와 너비가 `32`로 나누어질 수 있으면 어떤 크기든 지원할 수 있도록 패딩을 추가합니다.
- Swin은 *백본*으로 사용할 수 있습니다. `output_hidden_states = True`로 설정하면, `hidden_states`와 `reshaped_hidden_states`를 모두 출력합니다. `reshaped_hidden_states`는 `(batch, num_channels, height, width)` 형식을 가지며, 이는 `(batch_size, sequence_length, num_channels)` 형식과 다릅니다.

## 리소스 [[resources]]

Swin Transformer의 사용을 도울 수 있는 Hugging Face 및 커뮤니티(🌎로 표시)의 공식 자료 목록입니다.  

<PipelineTag pipeline="image-classification"/>

- [`SwinForImageClassification`]은 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)와 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)을 통해 지원됩니다.
- 관련 자료: [이미지 분류 작업 가이드](../tasks/image_classification)

또한:

- [`SwinForMaskedImageModeling`]은 이 [예제 스크립트](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining)를 통해 지원됩니다.

새로운 자료를 추가하고 싶으시다면, 언제든지 Pull Request를 열어주세요! 저희가 검토해 드릴게요. 이때, 추가하는 자료는 기존 자료와 중복되지 않고 새로운 내용을 보여주는 자료여야 합니다. 

## SwinConfig [[transformers.SwinConfig]]

[[autodoc]] SwinConfig


## SwinModel [[transformers.SwinModel]]

[[autodoc]] SwinModel
    - forward

## SwinForMaskedImageModeling [[transformers.SwinForMaskedImageModeling]]

[[autodoc]] SwinForMaskedImageModeling
    - forward

## SwinForImageClassification [[transformers.SwinForImageClassification]]

[[autodoc]] transformers.SwinForImageClassification
    - forward

