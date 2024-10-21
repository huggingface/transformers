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

# 파이프라인을 위한 유틸리티 [[utilities-for-pipelines]]

이 페이지는 라이브러리에서 파이프라인을 위해 제공하는 모든 유틸리티 함수들을 나열합니다.

이 함수들 대부분은 라이브러리 내 모델의 코드를 연구할 때만 유용합니다.

## 인자 처리 [[transformers.pipelines.ArgumentHandler]]

[[autodoc]] pipelines.ArgumentHandler

[[autodoc]] pipelines.ZeroShotClassificationArgumentHandler

[[autodoc]] pipelines.QuestionAnsweringArgumentHandler

## 데이터 형식 [[transformers.PipelineDataFormat]]

[[autodoc]] pipelines.PipelineDataFormat

[[autodoc]] pipelines.CsvPipelineDataFormat

[[autodoc]] pipelines.JsonPipelineDataFormat

[[autodoc]] pipelines.PipedPipelineDataFormat

## 유틸리티 [[transformers.pipelines.PipelineException]]

[[autodoc]] pipelines.PipelineException
