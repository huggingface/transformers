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

# 트레이너(Trainer)를 위한 유틸리티

이 페이지는 [`Trainer`]에서 사용되는 모든 유틸리티 함수들을 나열합니다.

이 함수들 대부분은 라이브러리 내 Trainer 코드를 연구할 때 유용합니다.

## 유틸리티

[[autodoc]] EvalPrediction

[[autodoc]] IntervalStrategy

[[autodoc]] enable_full_determinism

[[autodoc]] set_seed

[[autodoc]] torch_distributed_zero_first

## 콜백 내부 처리

[[autodoc]] trainer_callback.CallbackHandler

## 분산 평가

[[autodoc]] trainer_pt_utils.DistributedTensorGatherer

## 트레이너 인자 파서

[[autodoc]] HfArgumentParser

## 디버그 유틸리티

[[autodoc]] debug_utils.DebugUnderflowOverflow