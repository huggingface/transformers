<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CPU에서 효율적인 추론하기 [[efficient-inference-on-cpu]]

이 가이드는 CPU에서 대규모 모델을 효율적으로 추론하는 방법에 중점을 두고 있습니다.

### JIT 모드와 함께하는 IPEX 그래프 최적화 [[ipex-graph-optimization-with-jitmode]]
Intel® Extension for PyTorch(IPEX)는 Transformers 계열 모델의 jit 모드에서 추가적인 최적화를 제공합니다. jit 모드와 더불어 Intel® Extension for PyTorch(IPEX)를 활용하시길 강력히 권장드립니다. Transformers 모델에서 자주 사용되는 일부 연산자 패턴은 이미 jit 모드 연산자 결합(operator fusion)의 형태로 Intel® Extension for PyTorch(IPEX)에서 지원되고 있습니다. Multi-head-attention, Concat Linear, Linear+Add, Linear+Gelu, Add+LayerNorm 결합 패턴 등이 이용 가능하며 활용했을 때 성능이 우수합니다. 연산자 결합의 이점은 사용자에게 고스란히 전달됩니다. 분석에 따르면, 질의 응답, 텍스트 분류 및 토큰 분류와 같은 가장 인기 있는 NLP 태스크 중 약 70%가 이러한 결합 패턴을 사용하여 Float32 정밀도와 BFloat16 혼합 정밀도 모두에서 성능상의 이점을 얻을 수 있습니다.

[IPEX 그래프 최적화](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/features/graph_optimization.html)에 대한 자세한 정보를 확인하세요.

#### IPEX 설치: [[ipex-installation]]

IPEX 배포 주기는 PyTorch를 따라서 이루어집니다. 자세한 정보는 [IPEX 설치 방법](https://intel.github.io/intel-extension-for-pytorch/)을 확인하세요.
