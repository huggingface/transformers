<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# FP-Quant[[fp-quant]]

[FP-Quant](https://github.com/IST-DASLab/FP-Quant)는 **Nvidia Blackwell 세대 GPU**에 최적화된 양자화(Quantization) 알고리즘 모음입니다. 목적은 [MXFP4 및 NVFP4 데이터 타입](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)에서 대규모 언어 모델(LLM)에 대해 효율적인 사후 학습 양자화(PTQ)와 양자화 인지 학습(QAT)을 제공하는 것입니다.

현재는 **MXFP4 기반 PTQ**만 지원됩니다. 모델은 다음과 같이 `quantization_config=FPQuantConfig()`를 사용하여 **실행 시(On-the-fly)** 양자화하거나,

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, FPQuantConfig
import torch

model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen3-8B",
    quantization_config=FPQuantConfig(),
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)
```

[GPTQ](https://github.com/IST-DASLab/FP-Quant)를 통해 **사전 처리(Pre-processing)** 하여 더 높은 품질을 얻을 수도 있습니다.

커널을 실행하려면 **Blackwell 세대 GPU가 필수**입니다. 런타임 지원은 [QuTLASS](https://github.com/IST-DASLab/qutlass) 라이브러리와 경량 PyTorch 인터페이스인 [`fp_quant`](https://github.com/IST-DASLab/FP-Quant/tree/master/inference_lib)로 구현되어 있습니다. 전자는 **소스에서 직접 설치**하시길 권장하며, 후자는 `pip install fp_quant`로 설치하면 됩니다.

Blackwell 세대 GPU가 **없으신** 경우에는 `quantization_config=FPQuantConfig(pseudoquant=True)` 옵션을 통해 QuTLASS 설치 없이도 양자화를 **완전히 에뮬레이션**할 수 있습니다. 속도 향상은 없지만, 양자화 효과를 동일하게 재현합니다.

> [!TIP]
> FP-Quant로 **사전 양자화된** 모델은 ISTA-DASLab의 [공식 컬렉션](https://huggingface.co/collections/ISTA-DASLab/fp-quant-6877c186103a21d3a02568ee)에서 확인하실 수 있습니다.

## torch.compile

FP-Quant는 [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)과 **완전 호환**됩니다.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, FPQuantConfig

model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen3-8B",
    quantization_config=FPQuantConfig(),
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

model.forward = torch.compile(model.forward, mode="max-autotune", fullgraph=True)
```

## 속도 향상(Speedups)[[speedups]]

FP-Quant는 **매우 큰 배치(batch) 크기**에서 최고의 성능을 발휘합니다.

자세한 속도 비교는 [QuTLASS README](https://github.com/IST-DASLab/qutlass/blob/main/README.md)를 참고해 주세요.
