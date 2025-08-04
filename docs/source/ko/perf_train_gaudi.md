<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Intel Gaudi[[intel-gaudi]]

Intel Gaudi AI 가속기 제품군에는 [Intel Gaudi 1](https://habana.ai/products/gaudi/), [Intel Gaudi 2](https://habana.ai/products/gaudi2/), [Intel Gaudi 3](https://habana.ai/products/gaudi3/)가 포함됩니다. 각 서버에는 Habana Processing Unit(HPU)이라 불리는 장치가 8개 탑재되어 있으며, Gaudi 3에는 128GB, Gaudi 2에는 96GB, 1세대 Gaudi에는 32GB의 메모리가 제공됩니다. 기본 하드웨어 아키텍처에 대한 자세한 내용은 [Gaudi Architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html) 개요 문서를 참고하세요.

[`TrainingArguments`], [`Trainer`], [`Pipeline`]는 Intel Gaudi 장치를 자동으로 감지하여 백엔드 디바이스를 `hpu`로 설정합니다. 학습과 추론을 위해 별도의 설정을 추가할 필요는 없습니다.

Transformers의 일부 모델링 코드는 HPU의 lazy 모드에 최적화되어 있지 않습니다. 오류가 발생하는 경우에는 다음 환경 변수를 설정하여 eager 모드를 사용하세요:

```
PT_HPU_LAZY_MODE=0
```

경우에 따라 long integer 형식의 변환 문제를 피하려면 int64 지원을 활성화해야 합니다:

```
PT_ENABLE_INT64_SUPPORT=1
```

자세한 내용은 [Gaudi 문서](https://docs.habana.ai/en/latest/index.html)를 참고하세요.

> [!TIP]  
> Gaudi에 최적화된 모델 구현을 활용한 학습 및 추론을 위해서는 [Optimum for Intel Gaudi](https://huggingface.co/docs/optimum/main/en/habana/index) 사용을 권장합니다.
