<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Apple Silicon[[apple-silicon]]

Apple Silicon(M 시리즈)는 통합 메모리 아키텍처를 갖추고 있어, 대규모 모델을 로컬에서 효율적으로 학습할 수 있으며, 데이터 접근 지연을 줄여 성능을 향상시킵니다. [Metal Performance Shaders (MPS)](https://pytorch.org/docs/stable/notes/mps.html)와의 통합 덕분에 PyTorch를 사용할 때 Apple Silicon을 학습에 활용할 수 있습니다.

`mps` 백엔드는 macOS 12.3 이상에서 사용 가능합니다.

> [!WARNING]
> 일부 PyTorch 연산은 아직 MPS에서 구현되지 않았습니다. 오류를 방지하려면 환경 변수 `PYTORCH_ENABLE_MPS_FALLBACK=1`을 설정하여 CPU 커널로 대체 실행되도록 하세요. 다른 문제가 발생하면 [PyTorch](https://github.com/pytorch/pytorch/issues) 저장소에 이슈를 등록해 주세요.

[`TrainingArguments`]와 [`Trainer`]는 Apple Silicon 디바이스가 사용 가능한 경우 자동으로 백엔드 디바이스를 `mps`로 설정합니다. 별도의 설정 없이도 학습이 가능합니다.

`mps` 백엔드는 [분산 학습](https://pytorch.org/docs/stable/distributed.html#backends)을 지원하지 않습니다.

## 자료[[resources]]

MPS 백엔드에 대한 자세한 내용은 [Introducing Accelerated PyTorch Training on Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/) 블로그 글을 참고하세요.