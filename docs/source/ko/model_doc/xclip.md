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

# X-CLIP[[x-clip]]

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## 개요[[overview]]

X-CLIP 모델은 Bolin Ni, Houwen Peng, Minghao Chen, Songyang Zhang, Gaofeng Meng, Jianlong Fu, Shiming Xiang, Haibin Ling이 [Expanding Language-Image Pretrained Models for General Video Recognition](https://huggingface.co/papers/2208.02816)에서 제안했습니다.
X-CLIP은 비디오를 위해 [CLIP](clip)을 최소한으로 확장한 모델입니다. 이 모델은 텍스트 인코더, 교차 프레임 비전 인코더, 다중 프레임 통합 Transformer, 그리고 비디오별 프롬프트 생성기로 구성됩니다.

논문의 초록은 아래와 같습니다.

*대조적 언어-이미지 사전 학습은 웹 스케일 데이터로부터 시각-텍스트 공동 표현을 학습하는 데 큰 성공을 거두었으며, 다양한 이미지 작업에 대해 뛰어난 "제로샷(zero-shot)" 일반화 능력을 보여주었습니다. 그러나 이러한 새로운 언어-이미지 사전 학습 방법을 비디오 도메인으로 효과적으로 확장하는 방법은 아직 해결되지 않은 문제입니다. 본 연구에서는 새로운 모델을 처음부터 사전 학습하는 대신, 사전 학습된 언어-이미지 모델을 비디오 인식에 직접 적용하는 간단하면서도 효과적인 접근 방식을 제시합니다. 더 구체적으로, 시간 차원에서 프레임 간의 장기적인 의존성을 포착하기 위해 프레임 간 정보를 명시적으로 교환하는 교차 프레임 어텐션 메커니즘을 제안합니다. 이러한 모듈은 가벼울 뿐만 아니라, 사전 학습된 언어-이미지 모델에 쉽게 통합될 수 있습니다. 또한, 비디오 콘텐츠 정보를 활용하여 식별력 있는 텍스트 프롬프트를 생성하는 비디오별 프롬프팅 기법을 제안합니다. 광범위한 실험을 통해 우리의 접근 방식이 효과적이며 다양한 비디오 인식 시나리오에 일반화될 수 있음을 입증합니다. 특히, 완전 지도 학습 환경에서 우리 접근 방식은 Kinectics-400에서 87.1%의 top-1 정확도를 달성하면서도 Swin-L 및 ViViT-H에 비해 FLOPs를 12배 적게 사용합니다. 제로샷 실험에서는 두 가지 인기 있는 프로토콜 하에서 top-1 정확도 측면에서 현재 최첨단 방법들을 +7.6% 및 +14.9% 능가합니다. 퓨샷(few-shot) 시나리오에서는 레이블이 지정된 데이터가 극히 제한적일 때 이전 최고 방법들을 +32.1% 및 +23.1% 능가합니다.*

팁:

- X-CLIP의 사용법은 [CLIP](clip)과 동일합니다.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/xclip_architecture.png"
alt="drawing" width="600"/>

<small> X-CLIP 아키텍처. <a href="https://huggingface.co/papers/2208.02816">원본 논문</a>에서 가져왔습니다. </small>

이 모델은 [nielsr](https://huggingface.co/nielsr)님이 기여했습니다.
원본 코드는 [여기](https://github.com/microsoft/VideoX/tree/master/X-CLIP)에서 찾을 수 있습니다.

## 리소스[[resources]]

X-CLIP을 시작하는 데 도움이 되는 공식 Hugging Face 및 커뮤니티(🌎로 표시) 리소스 목록입니다.

- X-CLIP 데모 노트북은 [여기](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/X-CLIP)에서 찾을 수 있습니다.

여기에 포함할 리소스를 제출하는 데 관심이 있다면, 언제든지 Pull Request를 열어주세요. 검토 후 반영하겠습니다! 리소스는 기존 리소스를 복제하는 대신 새로운 것을 보여주는 것이 이상적입니다.

## XCLIPProcessor[[xclipprocessor]]

[[autodoc]] XCLIPProcessor

## XCLIPConfig[[xclipconfig]]

[[autodoc]] XCLIPConfig

## XCLIPTextConfig[[xcliptextconfig]]

[[autodoc]] XCLIPTextConfig

## XCLIPVisionConfig[[xclipvisionconfig]]

[[autodoc]] XCLIPVisionConfig

## XCLIPModel[[xclipmodel]]

[[autodoc]] XCLIPModel
    - forward
    - get_text_features
    - get_video_features

## XCLIPTextModel[[xcliptextmodel]]

[[autodoc]] XCLIPTextModel
    - forward

## XCLIPVisionModel[[xclipvisionmodel]]

[[autodoc]] XCLIPVisionModel
    - forward