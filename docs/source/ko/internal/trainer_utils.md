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

# Trainer를 위한 유틸리티 (Utilities for Trainer) [[utilities-for-trainer]]

이 페이지는 [`Trainer`]에서 사용되는 모든 유틸리티 함수들을 나열합니다.

이 함수들 대부분은 라이브러리에 있는 Trainer 코드를 자세히 알아보고 싶을 때만 유용합니다.

## 유틸리티 (Utilities) [[transformers.EvalPrediction]]

[[autodoc]] EvalPrediction

[[autodoc]] IntervalStrategy

[[autodoc]] enable_full_determinism

[[autodoc]] set_seed

[[autodoc]] torch_distributed_zero_first

## 콜백 내부 (Callbacks internals) [[transformers.trainer_callback.CallbackHandler]]

[[autodoc]] trainer_callback.CallbackHandler

## 분산 평가 (Distributed Evaluation) [[transformers.trainer_pt_utils.DistributedTensorGatherer]]

[[autodoc]] trainer_pt_utils.DistributedTensorGatherer

## Trainer 인자 파서 (Trainer Argument Parser) [[transformers.HfArgumentParser]]

[[autodoc]] HfArgumentParser

## 디버그 유틸리티 (Debug Utilities) [[transformers.debug_utils.DebugUnderflowOverflow]]

[[autodoc]] debug_utils.DebugUnderflowOverflow