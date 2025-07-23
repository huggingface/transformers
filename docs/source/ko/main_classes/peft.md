<!--Copyright 2024 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# PEFT[[transformers.integrations.PeftAdapterMixin]]

[`~integrations.PeftAdapterMixin`]은 Transformers 라이브러리와 함께 어댑터를 관리할 수 있도록 [PEFT](https://huggingface.co/docs/peft/index) 라이브러리의 함수들을 제공합니다. 이 믹스인은 현재 LoRA, IA3, AdaLora를 지원합니다. 프리픽스 튜닝 방법들(프롬프트 튜닝, 프롬프트 학습)은 torch 모듈에 삽입할 수 없는 구조이므로 지원되지 않습니다.

[[autodoc]] integrations.PeftAdapterMixin
    - load_adapter
    - add_adapter
    - set_adapter
    - disable_adapters
    - enable_adapters
    - active_adapters
    - get_adapter_state_dict
