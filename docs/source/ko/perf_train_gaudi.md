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

Intel Gaudi AI 가속기 제품군에는 [Intel Gaudi 1](https://habana.ai/products/gaudi/), [Intel Gaudi 2](https://habana.ai/products/gaudi2/), [Intel Gaudi 3](https://habana.ai/products/gaudi3/)가 포함되어 있습니다. 각 서버에는 Habana Processing Units(HPUs)라는 장치가 8개씩 탑재되어 있으며, Gaudi 3는 128GB, Gaudi 2는 96GB, Gaudi 1은 32GB 메모리가 제공됩니다. 하드웨어 구조에 대한 더 자세한 내용은 [Gaudi Architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html)를 참고하세요.

[`TrainingArguments`], [`Trainer`], [`Pipeline`]는 Intel Gaudi 장치를 자동으로 감지해, 백엔드 디바이스를 `hpu`로 설정합니다. 별도 설정 없이 곧바로 학습과 추론을 실행할 수 있습니다.

Transformers의 일부 모델링 코드는 HPU의 지연 실행(lazy) 모드에 최적화되어 있지 않습니다. 오류가 발생하는 경우, 아래처럼 환경 변수를 설정하여 즉시 실행(eager) 모드로 전환하세요.

```
PT_HPU_LAZY_MODE=0
```

경우에 따라 long integer의 자료형 변환 문제를 피하려면 아래와 같이 int64 지원을 활성화해야 하는 경우도 있습니다.

```
PT_ENABLE_INT64_SUPPORT=1
```

자세한 내용은 [Gaudi docs](https://docs.habana.ai/en/latest/index.html)를 참조하세요.

> [!TIP]
> Gaudi에 최적화된 모델로 학습하거나 추론할 경우, [Optimum for Intel Gaudi](https://huggingface.co/docs/optimum/main/en/habana/index)를 사용하는 것이 좋습니다.