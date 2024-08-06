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

# AQLM [[aqlm]]

> [!TIP]
> Try AQLM on [Google Colab](https://colab.research.google.com/drive/1-xZmBRXT5Fm3Ghn4Mwa2KRypORXb855X?usp=sharing)!

Additive Quantization of Language Models([AQLM](https://arxiv.org/abs/2401.06118))은 대규모 언어 모델 압축 방법입니다. 여러 가중치를 함께 양자화하고 가중치 간의 상호 의존성을 활용합니다. AQLM은 8-16개의 가중치 그룹을 여러 벡터 코드의 합으로 나타냅니다.


AQLM에 대한 추론 지원은 `aqlm` 라이브러리에서 실현됩니다. 모델을 실행하기 위해 반드시 설치하십시오(aqlm은 python>=3.10에서만 작동합니다):
```bash
pip install aqlm[gpu,cpu]
```

`aqlm` 라이브러리는 GPU 및 CPU 추론 및 훈련을 위한 효율적인 커널을 제공합니다.

모델을 직접 양자화하는 방법과 모든 관련 코드는 해당 GitHub [repository](https://github.com/Vahe1994/AQLM)에서 확인할 수 있습니다. AQLM 모델을 실행하려면 AQLM으로 양자화된 모델을 로드하기만 하면 됩니다:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

quantized_model = AutoModelForCausalLM.from_pretrained(
    "ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf",
    torch_dtype="auto", 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf")
```

## PEFT [[peft]]
AQLM은 `aqlm 1.0.2` 버전부터 [PEFT](https://huggingface.co/blog/peft) 라이브러리에 통합된 [LoRA](https://huggingface.co/docs/peft/package_reference/lora) 형태의 파라미터 효율적인 미세 조정을 지원합니다.

## AQLM 구성 [[aqlm-configurations]]

AQLM 양자화는 주로 사용되는 코드북의 개수와 비트 단위의 코드북 크기에 따라 다르게 구성할 수 있습니다. 가장 일반적인 구성과 각 구성이 지원하는 추론 커널은 다음과 같습니다:
 
| 커널 | 코드북의 개수 | 코드북의 크기, 비트 | 표기 | 정확도 | 최대 속도     | 빠른 GPU 추론 | 빠른 CPU 추론 |
|---|---------------------|---------------------|----------|-------------|-------------|--------------------|--------------------|
| Triton | K                   | N                  | KxN     | -        | 최대 0.7x | ✅                  | ❌                  |
| CUDA | 1                   | 16                  | 1x16     | Best        | 최대 ~1.3x | ✅                  | ❌                  |
| CUDA | 2                   | 8                   | 2x8      | OK          | 최대 ~3.0x | ✅                  | ❌                  |
| Numba | K                   | 8                   | Kx8      | Good        | 최대 ~4.0x | ❌                  | ✅                  |