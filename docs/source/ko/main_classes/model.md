<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 모델

기본 클래스 [`PreTrainedModel`], [`TFPreTrainedModel`], [`FlaxPreTrainedModel`]는 로컬 파일과 디렉토리로부터 모델을 로드하고 저장하거나 또는 (허깅페이스 AWS S3 리포지토리로부터 다운로드된) 라이브러리에서 제공하는 사전 훈련된 모델 설정을 로드하고 저장하는 것을 지원하는 기본 메소드를 구현하였습니다.   

[`PreTrainedModel`]과 [`TFPreTrainedModel`]은 또한 모든 모델들을 공통적으로 지원하는 메소드 여러개를 구현하였습니다:

- 새 토큰이 단어장에 추가될 때, 입력 토큰 임베딩의 크기를 조정합니다.
- 모델의 어텐션 헤드를 가지치기합니다.

각 모델에 공통인 다른 메소드들은 다음의 클래스에서 정의됩니다. 
- [`~modeling_utils.ModuleUtilsMixin`](파이토치 모델용)
- 텍스트 생성을 위한 [`~modeling_tf_utils.TFModuleUtilsMixin`](텐서플로 모델용)
- [`~generation.GenerationMixin`](파이토치 모델용)
- [`~generation.FlaxGenerationMixin`](Flax/JAX 모델용)

## PreTrainedModel

[[autodoc]] PreTrainedModel
    - push_to_hub
    - all

사용자 정의 모델은 초고속 초기화(superfast init)가 특정 모델에 적용될 수 있는지 여부를 결정하는 `_supports_assign_param_buffer`도 포함해야 합니다.
`test_save_and_load_from_pretrained` 실패 시, 모델이 `_supports_assign_param_buffer`를 필요로 하는지 확인하세요.
필요로 한다면 `False`로 설정하세요. 

## ModuleUtilsMixin

[[autodoc]] modeling_utils.ModuleUtilsMixin

## TFPreTrainedModel

[[autodoc]] TFPreTrainedModel
    - push_to_hub
    - all

## TFModelUtilsMixin

[[autodoc]] modeling_tf_utils.TFModelUtilsMixin

## FlaxPreTrainedModel

[[autodoc]] FlaxPreTrainedModel
    - push_to_hub
    - all

## 허브에 저장하기

[[autodoc]] utils.PushToHubMixin

## 공유된 체크포인트

[[autodoc]] modeling_utils.load_sharded_checkpoint
