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
*이 모델은 2025년 7월 10일에 출시되었으며, 2025년 7월 10일에 Hugging Face Transformers에 추가되었습니다.*

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# LFM2

## 개요[[overview]]

[LFM2](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models)는 Liquid AI가 개발한 차세대 Liquid Foundation Model로 egde AI와 온디바이스 배포에 특화되어 설계되었습니다.

이 모델들은 350M, 700M, 1.2B, 2.6B의 네 가지 크기의 매개변수로 제공되며, CPU, GPU, NPU 하드웨어에서 효율적으로 실행되도록 설계되었습니다. 이로 인해 특히 낮은 지연 시간, 오프라인 작동 및 개인 정보 보호가 필요한 애플리케이션에 적합합니다.

## 아키텍처[[architecture]]

아키텍처는 게이트가 있는 짧은 합성곱 블록과 QK 레이어 정규화가 적용된 그룹 쿼리 어텐션 블록으로 구성됩니다. 이 설계는 선형 연산이 입력 의존적인 게이트에 의해 조절되는 동적 시스템 개념에서 비롯되었습니다. 짧은 합성곱은 특히 임베디드 SoC CPU에 최적화되어 있어, 클라우드 연결에 의존하지 않고 빠르고 로컬화된 추론이 필요한 장치에 이상적입니다.

LFM2는 제한된 속도와 메모리 환경에서 품질을 최대화되도록 설계되었습니다. 이는 퀄컴 스냅드래곤 프로세서에서 실제 최대 메모리 사용량과 추론 속도를 측정하여, 임베디드 하드웨어에서의 실제 성능에 맞게 모델을 최적화하기 위한 체계적인 아키텍처 탐색을 통해 달성되었습니다. 그 결과, 비슷한 크기의 모델에 비해 2배 빠른 디코딩 및 프리필 성능을 달성하면서도, 지식, 수학, 지시 사항 따르기, 다국어 작업 전반에서 우수한 벤치마크 성능을 유지하는 모델이 탄생했습니다.

## 예시[[example]]

다음 예시는 `AutoModelForCausalLM` 클래스를 사용하여 답변을 생성하는 방법을 보여줍니다.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델과 토크나이저를 가져옵니다
model_id = "LiquidAI/LFM2-1.2B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="bfloat16",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 답변 생성
prompt = "What is C. elegans?"
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
)

output = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.3,
    min_p=0.15,
    repetition_penalty=1.05,
    max_new_tokens=512,
)

print(tokenizer.decode(output[0], skip_special_tokens=False))
```

## Lfm2Config [[transformers.Lfm2Config]]

[[autodoc]] Lfm2Config

## Lfm2Model [[transformers.Lfm2Model]]

[[autodoc]] Lfm2Model
    - forward

## Lfm2ForCausalLM [[transformers.Lfm2ForCausalLM]]

[[autodoc]] Lfm2ForCausalLM
    - forward