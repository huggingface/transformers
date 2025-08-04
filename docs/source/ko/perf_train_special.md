<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Apple 실리콘[[apple-silicon]]

Apple 실리콘(M 시리즈)은 통합 메모리 아키텍처를 기반으로 하여, 대규모 모델을 로컬 환경에서 효율적으로 학습할 수 있도록 설계되었습니다. 또한 데이터 접근 지연을 줄여 전반적인 성능 향상에 기여합니다. [Metal Performance Shaders (MPS)](https://pytorch.org/docs/stable/notes/mps.html)와의 통합 덕분에, PyTorch로 모델을 학습할 때 이러한 하드웨어적 이점을 그대로 활용할 수 있습니다.

`mps` 백엔드를 사용하려면 macOS 12.3 이상 버전이 필요합니다.

> [!WARNING]
> 일부 PyTorch 연산은 아직 MPS에서 구현되지 않았습니다. 오류를 방지하려면 환경 변수 `PYTORCH_ENABLE_MPS_FALLBACK=1`을 설정하여 CPU 커널로 대체 실행되도록 하세요. 다른 문제가 발생하면 [PyTorch](https://github.com/pytorch/pytorch/issues) 저장소에 이슈를 등록해 주세요.

[`TrainingArguments`]와 [`Trainer`]는 Apple 실리콘 기기를 감지하면, 자동으로 백엔드 디바이스를 `mps`로 설정하므로, 별도의 설정 없이 해당 기기에서 바로 학습을 진행할 수 있습니다. 

`mps` 백엔드는 [분산 학습(distributed training)](https://pytorch.org/docs/stable/distributed.html#backends)을 지원하지 않습니다.

## 자료[[resources]]

MPS 백엔드에 대한 자세한 내용은 [Introducing Accelerated PyTorch Training on Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/) 블로그 글에서 확인하실 수 있습니다.</file>