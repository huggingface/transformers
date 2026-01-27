<!--Copyright 2026 The LG AI Research and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-12-31 and added to Hugging Face Transformers on 2026-01-25.*

# EXAONE MoE

## Overview

**[K-EXAONE](https://github.com/LG-AI-EXAONE/K-EXAONE)** 모델은 LG AI연구원이 개발한 대규모 다국어 언어 모델입니다. `EXAONE-MoE`라는 Mixture-of-Experts 기반 구조를 채택해 총 236B 개의 파라미터를 갖고 추론 시 23B 개의 파라미터가 활성화됩니다. 다양한 벤치마크를 통한 성능 평가를 통해 K-EXAONE은 추론 능력, 에이전틱 작동 능력, 범용 지식, 다국어 이해, 그리고 긴 문맥 처리 능력을 증명했습니다. 

### 핵심 구조 및 기능

- **구조적 개선과 효율성:** 236B의 fine-grained MoE 구조(활성화 23B)를 채택했고, **Multi-Token Prediction (MTP)** 형태의 self-speculative decoding을 적용해 약 1.5배의 추론 throughput을 달성했습니다. 
- **긴 문맥 처리 능력:** 256K 문맥 크기를 자체적으로 지원하며, **3:1 hybrid attention** 구조와 **128-token sliding window**를 활용해 긴 문서 처리 시의 메모리 사용량을 크게 줄였습니다.
- **다국어 지원:** 한국어, 영어, 스페인어, 독일어, 일본어, 베트남어의 총 6개 언어를 공식 지원하며, 새로 디자인된 **SuperBPE** 기반 토크나이저와 **150k의 어휘 크기**를 통해 토큰 효율을 약 30% 향상했습니다.
- **에이전틱 처리 능력:** **멀티 에이전트 전략**을 통해 뛰어난 도구 사용 및 검색 능력을 보여줍니다.
- **안전성 & 윤리:** 다른 모델들이 종종 간과하는 한국 문화적, 역사적 맥락에 따른 민감한 주제에 올바르고 안전한 답변을 하도록 설계되었습니다. 그 외에도 보편적인 인간 가치에 맞추어 다양한 위험 요소에서도 높은 신뢰성을 보여줍니다.

더 자세한 정보는 [기술 보고서](https://www.lgresearch.ai/data/cdn/upload/K-EXAONE_Technical_Report.pdf)나 [공식 GitHub](https://huggingface.co/collections/LGAI-EXAONE/k-exaone) 페이지를 참고해주시길 바랍니다.

공개된 모든 모델 체크포인트는 [Huggingface 콜렉션](https://huggingface.co/collections/LGAI-EXAONE/k-exaone)에서 확인할 수 있습니다.


## 모델 세부 정보

- Number of Parameters: 236B in total and 23B activated
- Number of Parameters (without embeddings): 234B
- Hidden Dimension: 6,144
- Number of Layers: 48 Main layers + 1 MTP layers
  - Hybrid Attention Pattern: 12 x (3 Sliding window attention + 1 Global attention)
- Sliding Window Attention
  - Number of Attention Heads: 64 Q-heads and 8 KV-heads
  - Head Dimension: 128 for both Q/KV
  - Sliding Window Size: 128
- Global Attention
  - Number of Attention Heads: 64 Q-heads and 8 KV-heads
  - Head Dimension: 128 for both Q/KV
  - No Rotary Positional Embedding Used (NoPE)
- Mixture of Experts:
  - Number of Experts: 128
  - Number of Activated Experts: 8
  - Number of Shared Experts: 1
  - MoE Intermediate Size: 2,048
- Vocab Size: 153,600
- Context Length: 262,144 tokens
- Knowledge Cutoff: Dec 2024 (2024/12)

## 사용 팁

### 사용 시 주의사항

> [!IMPORTANT]
> 모델이 설계된 성능을 내기 위해서는 아래 설정들을 지켜주시길 바랍니다.
> - 가장 나은 결과를 얻기 위해 `temperature=1.0`, `top_p=0.95`, `presence_penalty=0.0` 를 사용하길 권고합니다.
> - 이전 EXAONE-4.0 모델들과는 다르게 K-EXAONE은 기본적으로 `enable_thinking=True` 를 사용합니다. 따라서 non-reasoning mode를 사용하기 위해서는 `enable_thinking=False`를 설정해줘야 합니다.
>

### Reasoning mode

정확한 결과가 필요한 작업을 할 때, 아래처럼 K-EXAONE 모델을 reasoning mode로 사용할 수 있습니다.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LGAI-EXAONE/K-EXAONE-236B-A23B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="bfloat16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "You are K-EXAONE, a large language model developed by LG AI Research in South Korea, built to serve as a helpful and reliable assistant."},
    {"role": "user", "content": "Which one is bigger, 3.9 vs 3.12?"}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=True,   # skippable (default: True)
)

generated_ids = model.generate(
    **input_ids.to(model.device),
    max_new_tokens=16384,
    temperature=1.0,
    top_p=0.95,
)
output_ids = generated_ids[0][input_ids['input_ids'].shape[-1]:]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

### Non-reasoning mode

정확도보다 속도가 더 중요한 상황에서는, 아래처럼 K-EXAONE 모델을 non-reasoning mode로 사용할 수 있습니다.

```python
messages = [
    {"role": "system", "content": "You are K-EXAONE, a large language model developed by LG AI Research in South Korea, built to serve as a helpful and reliable assistant."},
    {"role": "user", "content": "Explain how wonderful you are"}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=False,
)

generated_ids = model.generate(
    **input_ids.to(model.device),
    max_new_tokens=1024,
    temperature=1.0,
    top_p=0.95,
)
output_ids = generated_ids[0][input_ids['input_ids'].shape[-1]:]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

### Agentic tool use

AI 기반 에이전트를 구성할 때 K-EXAONE의 도구 활용 능력이 발휘됩니다.
K-EXAONE 모델은 OpenAI 및 HuggingFace의 도구 활용 명세를 따릅니다. 
아래는 HuggingFace의 docstring을 도구 스키마로 변환하는 유틸리티를 사용해 도구 활용 기능을 이용하는 예시입니다.

K-EXAONE을 활용한 검색 에이전트의 실제 대화 기록을 살펴보려면 [GitHub의 예시 파일](https://github.com/LG-AI-EXAONE/K-EXAONE/blob/main/examples/example_output_search.txt)을 참고하세요.


```python
from transformers.utils import get_json_schema

def roll_dice(max_num: int):
    """
    Roll a dice with the number 1 to N. User can select the number N.

    Args:
        max_num: The maximum number on the dice.
    """
    return random.randint(1, max_num)

tool_schema = get_json_schema(roll_dice)
tools = [tool_schema]

messages = [
    {"role": "system", "content": "You are K-EXAONE, a large language model developed by LG AI Research in South Korea, built to serve as a helpful and reliable assistant."},
    {"role": "user", "content": "Roll a D20 twice and sum the results."}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    tools=tools,
)

generated_ids = model.generate(
    **input_ids.to(model.device),
    max_new_tokens=16384,
    temperature=1.0,
    top_p=0.95,
)
output_ids = generated_ids[0][input_ids['input_ids'].shape[-1]:]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
```

## ExaoneMoeConfig

[[autodoc]] ExaoneMoeConfig

## ExaoneMoeModel

[[autodoc]] ExaoneMoeModel
    - forward

## ExaoneMoeForCausalLM

[[autodoc]] ExaoneMoeForCausalLM
    - forward
