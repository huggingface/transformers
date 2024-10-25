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

# PatchTST[[patchtst]]

## 개요[[overview]]

The PatchTST 모델은 Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam이 제안한 [시계열 하나가 64개의 단어만큼 가치있다: 트랜스포머를 이용한 장기예측](https://arxiv.org/abs/2211.14730)라는 논문에서 소개되었습니다.

이 모델은 고수준에서 시계열을 주어진 크기의 패치로 벡터화하고, 결과로 나온 벡터 시퀀스를 트랜스포머를 통해 인코딩한 다음 적절한 헤드를 통해 예측 길이의 예측을 출력합니다. 모델은 다음 그림과 같이 도식화됩니다:

![모델](https://github.com/namctin/transformers/assets/8100/150af169-29de-419a-8d98-eb78251c21fa)

해당 논문의 초록입니다:

*우리는 다변량 시계열 예측과 자기 감독 표현 학습을 위한 효율적인 트랜스포머 기반 모델 설계를 제안합니다. 이는 두 가지 주요 구성 요소를 기반으로 합니다: 

(i) 시계열을 하위 시리즈 수준의 패치로 분할하여 트랜스포머의 입력 토큰으로 사용
(ii) 각 채널이 모든 시리즈에 걸쳐 동일한 임베딩과 트랜스포머 가중치를 공유하는 단일 단변량 시계열을 포함하는 채널 독립성. 패칭 설계는 자연스럽게 세 가지 이점을 가집니다: 
    - 지역적 의미 정보가 임베딩에 유지됩니다; 
    - 동일한 룩백 윈도우에 대해 어텐션 맵의 계산과 메모리 사용량이 제곱으로 감소합니다
    - 모델이 더 긴 과거를 참조할 수 있습니다. 
    우리의 채널 독립적 패치 시계열 트랜스포머(PatchTST)는 최신 트랜스포머 기반 모델들과 비교했을 때 장기 예측 정확도를 크게 향상시킬 수 있습니다. 또한 모델을 자기지도 사전 훈련 작업에 적용하여, 대규모 데이터셋에 대한 지도 학습을 능가하는 아주 뛰어난 미세 조정 성능을 달성했습니다. 한 데이터셋에서 마스크된 사전 훈련 표현을 다른 데이터셋으로 전이하는 것도 최고 수준의 예측 정확도(SOTA)를 산출했습니다.*

이 모델은 [namctin](https://huggingface.co/namctin), [gsinthong](https://huggingface.co/gsinthong), [diepi](https://huggingface.co/diepi), [vijaye12](https://huggingface.co/vijaye12), [wmgifford](https://huggingface.co/wmgifford), [kashif](https://huggingface.co/kashif)에 의해 기여 되었습니다. 원본코드는 [이곳](https://github.com/yuqinie98/PatchTST)에서 확인할 수 있습니다.

## 사용 팁[[usage-tips]]

이 모델은 시계열 분류와 시계열 회귀에도 사용될 수 있습니다. 각각 [`PatchTSTForClassification`]와 [`PatchTSTForRegression`] 클래스를 참조하세요.

## 자료[[resources]]

- PatchTST를 자세히 설명하는 블로그 포스트는 [이곳](https://huggingface.co/blog/patchtst)에서 찾을 수 있습니다. 
이 블로그는 Google Colab에서도 열어볼 수 있습니다.

## PatchTSTConfig[[transformers.PatchTSTConfig]]

[[autodoc]] PatchTSTConfig

## PatchTSTModel[[transformers.PatchTSTModel]]

[[autodoc]] PatchTSTModel
    - forward

## PatchTSTForPrediction[[transformers.PatchTSTForPrediction]]

[[autodoc]] PatchTSTForPrediction
    - forward

## PatchTSTForClassification[[transformers.PatchTSTForClassification]]

[[autodoc]] PatchTSTForClassification
    - forward

## PatchTSTForPretraining[[transformers.PatchTSTForPretraining]]

[[autodoc]] PatchTSTForPretraining
    - forward

## PatchTSTForRegression[[transformers.PatchTSTForRegression]]

[[autodoc]] PatchTSTForRegression
    - forward
