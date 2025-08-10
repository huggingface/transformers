<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ViTDet[[vitdet]]

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## 개요[[overview]]

ViTDet 모델은 Yanghao Li, Hanzi Mao, Ross Girshick, Kaiming He가 작성한 [Exploring Plain Vision Transformer Backbones for Object Detection](https://huggingface.co/papers/2203.16527) 논문에서 제안되었습니다.
ViTDet은 객체 탐지 작업을 위해 기본 [Vision Transformer](vit)를 활용합니다.

논문의 초록은 다음과 같습니다:

*우리는 객체 탐지를 위한 백본 네트워크로 계층적이지 않은 기본 Vision Transformer(ViT)를 탐구합니다. 이 설계를 통해 사전 학습을 위한 계층적 백본을 재설계할 필요 없이 원래의 ViT 아키텍처를 객체 탐지용으로 미세 조정할 수 있습니다. 최소한의 미세 조정 적응만으로도 우리의 기본 백본 탐지기는 경쟁력 있는 결과를 달성할 수 있습니다. 놀랍게도 우리는 다음을 관찰했습니다: (i) 단일 스케일 특징 맵에서 간단한 특징 피라미드를 구축하는 것만으로도 충분하며(일반적인 FPN 설계 없이도) (ii) 매우 적은 수의 크로스 윈도우 전파 블록의 도움을 받은 윈도우 어텐션(시프팅 없이)만으로도 충분합니다. Masked Autoencoders(MAE)로 사전 학습된 기본 ViT 백본을 사용하여, ViTDet이라고 명명된 우리의 탐지기는 모두 계층적 백본을 기반으로 한 이전의 선도적인 방법들과 경쟁할 수 있으며, ImageNet-1K 사전 학습만을 사용하여 COCO 데이터셋에서 최대 61.3 AP_box에 도달합니다. 우리는 이 연구가 기본적인 백본 탐지기 연구에 관심을 끌기를 바랍니다.*

이 모델은 [nielsr](https://huggingface.co/nielsr)이 기여했습니다.
원본 코드는 [여기](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet)에서 찾을 수 있습니다.

팁:

- 현재는 백본만 사용 가능합니다.

## VitDetConfig[[transformers.VitDetConfig]]

[[autodoc]] VitDetConfig

## VitDetModel[[transformers.VitDetModel]]

[[autodoc]] VitDetModel
   - forward