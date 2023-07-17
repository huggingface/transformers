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

# 어텐션 메커니즘[[attention_mechanisms]]

대부분의 트랜스포머 모델은 정방행렬인 전체 어텐션을 사용합니다. 
하지만 이는 긴 텍스트를 다룰 때는 큰 계산 병목 현상을 유발할 수 있습니다. 
`Longformer`와 `Reformer`는 훈련 속도를 높이기 위해 어텐션 행렬의 희소 버전을 사용하여 효율을 높이려는 모델입니다.

## LSH 어텐션[[lsh_attention]]


[Reformer](#reformer)는 LSH(Locality Sensitive Hashing) 어텐션을 사용합니다. softmax(QK^t)에서는 행렬 QK^t의 (softmax 차원에서) 가장 큰 요소들만 유용한 기여를 할 것입니다. 
따라서 각각의 쿼리 q에 대해, q와 가까운 키 k만 고려할 수 있습니다. 해시 함수는 q와 k가 가까운지 여부를 결정하는 데 사용됩니다. 
어텐션 마스크는 현재 토큰을 마스킹하여 변경됩니다. 이 때 첫 번째 위치의 토큰은 제외합니다. 왜냐하면 쿼리와 키가 동일한 값을 갖게 되기 때문입니다(서로 매우 유사함). 
해시는 약간의 무작위성을 가질 수 있으므로, 실제로는 여러 개의 해시 함수가 사용되고 (`n_rounds` 매개변수에 의해 결정됨) 그 후에 평균값을 취하게 됩니다.

## 지역 어텐션[[local_attention]]

[Longformer](#longformer)는 지역 어텐션을 사용합니다. 종종 특정 토큰에 대해 지역 컨텍스트(예: 왼쪽과 오른쪽에 있는 두 개의 토큰은 무엇인가요?)만으로도 작업을 수행하는데 충분합니다. 
또한 작은 창(window)을 가진 어텐션 레이어를 쌓음으로써 마지막 레이어는 창 내의 토큰뿐만 아니라 더 많은 수의 토큰에 대한 수용 영역(receptive field)을 갖게 되어 전체 문장의 표현을 구축할 수 있습니다.

사전에 선택된 일부 입력 토큰들은 전역 어텐션을 받습니다. 이 몇 개의 토큰에 대해서는 어텐션 행렬이 모든 토큰에 접근할 수 있으며, 이 과정은 대칭적으로 이루어집니다. 
다른 모든 토큰들은 로컬 창 내의 토큰들에 더해 해당 특정 토큰들에도 접근할 수 있습니다. 이는 논문의 Figure 2d에서 나타나며, 아래에 샘플 어텐션 마스크가 제시되어 있습니다:


<div class="flex justify-center">
    <img scale="50 %" align="center" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/local_attention_mask.png"/>
</div>


적은 파라미터의 어텐션 행렬을 사용하면 모델이 더 큰 시퀀스 입력 길이를 가질 수 있습니다.

## 다른 방법들[[other_tricks]]

### 축별 위치 인코딩[[axial_positional_encodings]]

[Reformer](#reformer)는 축별 위치 인코딩(axial positional encodings)을 사용합니다. 기존의 트랜스포머 모델에서는 위치 인코딩 행렬 E는 크기가 \\(l \times d\\)인 행렬이며, 
여기서 \\(l\\)은 시퀀스 길이(sequence length)이고 \\(d\\)는 숨겨진 상태(hidden state)의 차원입니다. 매우 긴 텍스트의 경우, 이 행렬은 매우 크며 GPU 상에서 공간을 많이 차지할 수 있습니다. 
이를 완화하기 위해, 축별 위치 인코딩은 큰 행렬 E를 두 개의 작은 행렬 E1과 E2로 분해합니다. 이때 E1의 크기는 \\(l_{1} \times d_{1}\\)이고, E2의 크기는 \\(l_{2} \times d_{2}\\)입니다. 
이때 \\(l_{1} \times l_{2} = l\\)이고 \\(d_{1} + d_{2} = d\\)(길이에 대한 곱셈 연산을 사용하면 훨씬 작아집니다). E의 시간 단계 j에 대한 임베딩은 E1에서 시간 단계 \\(j \% l1\\)의 임베딩과 E2에서 시간 단계  \\(j // l1\\)의 임베딩을 연결하여 얻습니다.