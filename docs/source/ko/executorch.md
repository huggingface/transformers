<!--Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->


# ExecuTorch [[executorch]]

[`ExecuTorch`](https://github.com/pytorch/executorch) 는 웨어러블, 임베디드 장치, 마이크로컨트롤러를 포함한 모바일 및 엣지 장치에서 온디바이스 추론 기능을 가능하게 하는 종합 솔루션입니다. PyTorch 생태계에 속해있으며, 이식성, 생산성, 성능에 중점을 둔 PyTorch 모델 배포를 지원합니다.

ExecuTorch는 백엔드 위임, 사용자 정의 컴파일러 변환, 메모리 계획 등 모델, 장치 또는 특정 유즈케이스 맞춤 최적화를 수행할 수 있는 진입점을 명확하게 정의합니다. ExecuTorch를 사용해 엣지 장치에서 PyTorch 모델을 실행하는 첫 번째 단계는 모델을 익스포트하는 것입니다. 이 작업은 PyTorch API인 [`torch.export`](https://pytorch.org/docs/stable/export.html)를 사용하여 수행합니다.


## ExecuTorch 통합 [[transformers.TorchExportableModuleWithStaticCache]]

`torch.export`를 사용하여 🤗 Transformers를 익스포트 할 수 있도록  통합 지점이 개발되고 있습니다. 이 통합의 목표는 익스포트뿐만 아니라, 익스포트한 아티팩트가 `ExecuTorch`에서 효율적으로 실행될 수 있도록 더 축소하고 최적화하는 것입니다. 특히 모바일 및 엣지 유즈케이스에 중점을 두고 있습니다.

[[autodoc]] integrations.executorch.TorchExportableModuleWithStaticCache
    - forward

[[autodoc]] integrations.executorch.convert_and_export_with_cache
