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

# DeepSpeed[[deepspeed]]

Zero Redundancy Optimizer (ZeRO)로 구동되는 [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)는 매우 큰 모델을 GPU에서 훈련하고 피팅하기 위한 최적화 라이브러리입니다. 여러 ZeRO 단계로 제공되며, 각 단계는 옵티마이저 상태, 기울기, 매개변수를 분할하고 CPU 또는 NVMe로의 오프로딩을 가능하게 하여 점진적으로 더 많은 GPU 메모리를 절약합니다. DeepSpeed는 [`Trainer`] 클래스와 통합되어 있으며 대부분의 설정이 자동으로 처리됩니다.

하지만 [`Trainer`] 없이 DeepSpeed를 사용하려는 경우를 위해 Transformers에서는 [`HfDeepSpeedConfig`] 클래스를 제공합니다.

<Tip>

[`Trainer`]와 함께 DeepSpeed를 사용하는 방법에 대해서는 [DeepSpeed](../deepspeed) 가이드를 참조하세요.

</Tip>

## HfDeepSpeedConfig[[transformers.integrations.HfDeepSpeedConfig]]

[[autodoc]] integrations.HfDeepSpeedConfig
    - all
