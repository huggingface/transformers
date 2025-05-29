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

# 구성[[configuration]]

기본 클래스 [`PretrainedConfig`]는 로컬 파일이나 디렉토리, 또는 라이브러리에서 제공하는 사전 학습된 모델 구성(HuggingFace의 AWS S3 저장소에서 다운로드됨)으로부터 구성을 불러오거나 저장하는 공통 메서드를 구현합니다. 각 파생 구성 클래스는 모델별 특성을 구현합니다. 

모든 구성 클래스에 존재하는 공통 속성은 다음과 같습니다: `hidden_size`, `num_attention_heads`, `num_hidden_layers`. 텍스트 모델은 추가로 `vocab_size`를 구현합니다.


## PretrainedConfig[[transformers.PretrainedConfig]]

[[autodoc]] PretrainedConfig
    - push_to_hub
    - all
