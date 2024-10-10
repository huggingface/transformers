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

# 시계열 트랜스포머[[time-series-transformer]]

## 개요[[overview]]

이 시계열 트랜스포머 모델은 시계열 예측을 위한 기본적인 인코더-디코더 구조의 트랜스포머 입니다.
이 모델은 [kashif](https://huggingface.co/kashif)에 의해 기여되었습니다.

## 사용 팁[[usage-tips]]

- 다른 라이브러리의 모델들과 마찬가지로, [`TimeSeriesTransformerModel`]은 상단에 헤드가 없는 기본적인 트랜스포머 입니다. [`TimeSeriesTransformerForPrediction`]은 상단에 분포 헤드를 추가하여 시계열 예측에 사용할 수 있습니다. 이 모델은 이른바 확률적 예측 모델이며, 포인트 예측 모델이 아닙니다. 즉 샘플링할 수 있는 분포를 학습하며, 값을 직접 출력 하지는 않습니다.
- [`TimeSeriesTransformerForPrediction`]은 두개의 블록으로 구성되어 있습니다. 인코더는 `context_length`의  시계열 값을 입력(`past_values`라고 부름)으로 받아들이며, 디코더는 미래의 `prediction_length`만큼 시계열 값을 예측합니다(`future_values`라고 부름). 학습중에는 모델에 `past_values` 와 `future_values`쌍을 모델에 제공해야 합니다.
- 가공하지 않은 `past_values` 와 `future_values` 쌍 외에도, 일반적으로 모델에 추가적인 특징을 제공합니다. 다음은 그 특징들에 대해 소개합니다:
    - `past_time_features`: 모델이 `past_values`에 추가할 시간적 특성. 이는 트랜스포머 인코더의 "위치 인코딩" 역할을 합니다.
    예를 들어 "월의 일", "연도의 월" 등을 스칼라 값으로 (그리고 벡터로 쌓아서) 나타냅니다.
    예시: 특정 시계열 값이 8월 11일에 기록되었다면, [11, 8]을 시간 특성 벡터로 사용할 수 있습니다 (11은 "월의 일", 8은 "연도의 월").
    - `future_time_features`: 모델이 `future_values`에 추가할 시간적 특성. 이는 트랜스포머 디코더의 "위치 인코딩" 역할을 합니다.
    예를 들어 "월의 일", "연도의 월" 등을 스칼라 값으로 (그리고 벡터로 쌓아서) 나타냅니다.
    예: 특정 시계열 값이 8월 11일에 얻어졌다면, [11, 8]을 시간 특성 벡터로 사용할 수 있습니다 (11은 "월의 일", 8은 "연도의 월").
    - `static_categorical_features`: 시간에 따라 변하지 않는 범주형 특성 (즉, 모든 `past_values`와 `future_values`에 대해 동일한 값을 가짐).
    예를 들어 특정 시계열을 식별하는 매장 ID나 지역 ID가 있습니다.
    이러한 특성은 모든 데이터 포인트(미래의 데이터 포인트 포함)에 대해 알려져 있어야 합니다.
    - `static_real_features`: 시간에 따라 변하지 않는 실수값 특성 (즉, 모든 `past_values`와 `future_values`에 대해 동일한 값을 가짐).
    예를 들어 시계열 값을 가진 제품의 이미지 표현 (시계열이 신발 판매에 관한 것이라면 "신발" 사진의 [ResNet](resnet) 임베딩 처럼)이 있습니다.
    이러한 특성은 모든 데이터 포인트(미래의 데이터 포인트 포함)에 대해 알려져 있어야 합니다.
- 이 모델은 기계 번역을 위한 트랜스포머 훈련과 유사하게 "교사 강제(teacher-forcing)" 방식으로 훈련됩니다. 즉, 훈련 중에 `future_values`를 디코더의 입력으로 오른쪽으로 한 위치 이동시키고, `past_values`의 마지막 값을 앞에 붙입니다. 각 시간 단계에서 모델은 다음 타겟을 예측해야 합니다. 따라서 훈련 설정은 언어를 위한 GPT 모델과 유사하지만, `decoder_start_token_id` 개념이 없습니다 (우리는 단순히 컨텍스트의 마지막 값을 디코더의 초기 입력으로 사용합니다).
- 추론 시에는 `past_values`의 최종 값을 디코더의 입력으로 제공합니다. 그 다음, 모델에서 샘플링하여 다음 시간 단계에서의 예측을 만들고, 이를 디코더에 공급하여 다음 예측을 만듭니다 (자기회귀 생성이라고도 함).

## 자료 [[resources]]

시작하는 데 도움이 되는 Hugging Face와 community 자료 목록(🌎로 표시됨) 입니다. 여기에 포함될 자료를 제출하고 싶으시다면 PR(Pull Request)를 열어주세요. 리뷰 해드리겠습니다! 자료는 기존 자료를 복제하는 대신 새로운 내용을 담고 있어야 합니다.

- HuggingFace 블로그에서 시계열 트랜스포머 포스트를 확인하세요: [🤗 트랜스포머와 확률적 시계열 예측](https://huggingface.co/blog/time-series-transformers)

## TimeSeriesTransformerConfig[[transformers.TimeSeriesTransformerConfig]]

[[autodoc]] TimeSeriesTransformerConfig

## TimeSeriesTransformerModel[[transformers.TimeSeriesTransformerModel]]

[[autodoc]] TimeSeriesTransformerModel
    - forward

## TimeSeriesTransformerForPrediction[[transformers.TimeSeriesTransformerForPrediction]]

[[autodoc]] TimeSeriesTransformerForPrediction
    - forward
