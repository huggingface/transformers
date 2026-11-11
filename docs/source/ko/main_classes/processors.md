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

# 프로세서[[processors]]

## 멀티모달 프로세서[[transformers.ProcessorMixin]]

모든 멀티모달 모델은 여러 모달리티(텍스트, 비전, 오디오)를 그룹화하는 데이터를 인코딩하거나 디코딩하는 객체가 필요한데, 이것은 프로세서라고 불리는 객체가 담당합니다. 프로세서는 토크나이저(텍스트 모달리티용), 이미지 프로세서(비전용), 특성 추출기(오디오용) 같이 두 개 이상의 처리 객체를 하나로 묶습니다.

이러한 프로세서는 저장 및 로딩 기능을 구현하는 다음 기본 클래스를 상속받습니다:

[[autodoc]] ProcessorMixin
