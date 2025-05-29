<!--Copyright 2023 IBM and HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# PatchTSMixer[[patchtsmixer]]

## 개요[[overview]]

PatchTSMixer 모델은 Vijay Ekambaram, Arindam Jati, Nam Nguyen, Phanwadee Sinthong, Jayant Kalagnanam이 제안한 [TSMixer: 다변량 시계열 예측을 위한 경량 MLP-Mixer 모델](https://arxiv.org/pdf/2306.09364.pdf)이라는 논문에서 소개되었습니다.


PatchTSMixer는 MLP-Mixer 아키텍처를 기반으로 한 경량 시계열 모델링 접근법입니다. 허깅페이스 구현에서는 PatchTSMixer의 기능을 제공하여 패치, 채널, 숨겨진 특성 간의 경량 혼합을 쉽게 수행하여 효과적인 다변량 시계열 모델링을 가능하게 합니다. 또한 간단한 게이트 어텐션부터 사용자 정의된 더 복잡한 셀프 어텐션 블록까지 다양한 어텐션 메커니즘을 지원합니다. 이 모델은 사전 훈련될 수 있으며 이후 예측, 분류, 회귀와 같은 다양한 다운스트림 작업에 사용될 수 있습니다.


해당 논문의 초록입니다:

*TSMixer는 패치 처리된 시계열의 다변량 예측 및 표현 학습을 위해 설계된 다층 퍼셉트론(MLP) 모듈로만 구성된 경량 신경망 아키텍처입니다. 우리의 모델은 컴퓨터 비전 분야에서 MLP-Mixer 모델의 성공에서 영감을 받았습니다. 우리는 Vision MLP-Mixer를 시계열에 적용하는 데 따르는 과제를 보여주고, 정확도를 향상시키기 위해 경험적으로 검증된 구성 요소들을 도입합니다. 여기에는 계층 구조 및 채널 상관관계와 같은 시계열 특성을 명시적으로 모델링하기 위해 MLP-Mixer 백본에 온라인 조정 헤드를 부착하는 새로운 설계 패러다임이 포함됩니다. 또한 기존 패치 채널 혼합 방법의 일반적인 문제인 노이즈가 있는 채널 상호작용을 효과적으로 처리하고 다양한 데이터셋에 걸쳐 일반화하기 위한 하이브리드 채널 모델링 접근법을 제안합니다. 추가로, 중요한 특성을 우선시하기 위해 백본에 간단한 게이트 주의 메커니즘을 도입합니다. 이러한 경량 구성 요소들을 통합함으로써, 우리는 단순한 MLP 구조의 학습 능력을 크게 향상시켜 최소한의 컴퓨팅 사용으로 복잡한 트랜스포머 모델들을 능가하는 성능을 달성합니다. 더욱이, TSMixer의 모듈식 설계는 감독 학습과 마스크 자기 감독 학습 방법 모두와 호환되어 시계열 기초 모델의 유망한 구성 요소가 됩니다. TSMixer는 예측 작업에서 최첨단 MLP 및 트랜스포머 모델들을 상당한 차이(8-60%)로 능가합니다. 또한 최신의 강력한 Patch-Transformer 모델 벤치마크들을 메모리와 실행 시간을 크게 줄이면서(2-3배) 성능 면에서도 앞섭니다(1-2%).*

이 모델은 [ajati](https://huggingface.co/ajati), [vijaye12](https://huggingface.co/vijaye12), 
[gsinthong](https://huggingface.co/gsinthong), [namctin](https://huggingface.co/namctin),
[wmgifford](https://huggingface.co/wmgifford), [kashif](https://huggingface.co/kashif)가 기여했습니다.

## 사용 예[[usage-example]]

아래의 코드 스니펫은 PatchTSMixer 모델을 무작위로 초기화하는 방법을 보여줍니다. 
PatchTSMixer 모델은 [Trainer API](../trainer.md)와 호환됩니다.

```python

from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction
from transformers import Trainer, TrainingArguments,


config = PatchTSMixerConfig(context_length = 512, prediction_length = 96)
model = PatchTSMixerForPrediction(config)
trainer = Trainer(model=model, args=training_args, 
            train_dataset=train_dataset,
            eval_dataset=valid_dataset)
trainer.train()
results = trainer.evaluate(test_dataset)
```

## 사용 팁[[usage-tips]]

이 모델은 시계열 분류와 시계열 회귀에도 사용될 수 있습니다. 각각[`PatchTSMixerForTimeSeriesClassification`]와 [`PatchTSMixerForRegression`] 클래스를 참조하세요.

## 자료[[resources]]

- PatchTSMixer를 자세히 설명하는 블로그 포스트는 여기에서 찾을 수 있습니다 [이곳](https://huggingface.co/blog/patchtsmixer). 이 블로그는 Google Colab에서도 열어볼 수 있습니다.

## PatchTSMixerConfig[[transformers.PatchTSMixerConfig]]

[[autodoc]] PatchTSMixerConfig


## PatchTSMixerModel[[transformers.PatchTSMixerModel]]

[[autodoc]] PatchTSMixerModel
    - forward


## PatchTSMixerForPrediction[[transformers.PatchTSMixerForPrediction]]

[[autodoc]] PatchTSMixerForPrediction
    - forward


## PatchTSMixerForTimeSeriesClassification[[transformers.PatchTSMixerForTimeSeriesClassification]]

[[autodoc]] PatchTSMixerForTimeSeriesClassification
    - forward


## PatchTSMixerForPretraining[[transformers.PatchTSMixerForPretraining]]

[[autodoc]] PatchTSMixerForPretraining
    - forward


## PatchTSMixerForRegression[[transformers.PatchTSMixerForRegression]]

[[autodoc]] PatchTSMixerForRegression
    - forward