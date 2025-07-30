<!--Copyright 2025 The LG AI Research and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# EXAONE 4

## 개요

**[EXAONE 4.0](https://github.com/LG-AI-EXAONE/EXAONE-4.0)** 모델군은 [EXAONE 3.5](https://github.com/LG-AI-EXAONE/EXAONE-3.5) 모델군의 높은 실용성과 [EXAONE Deep](https://github.com/LG-AI-EXAONE/EXAONE-Deep) 모델군의 향상된 사고 추론 능력을 각각 Non-reasoning mode와 Reasoning mode로 통합한 자연어 모델(language model)입니다. 에이전틱(agentic) AI 시대에 발맞춰 EXAONE 4.0은 에이전틱 도구 사용 능력과 같은 핵심 기능을 통합했고, 기존의 다국어 능력을 영어, 한국어와 더불어 스페인어까지 확장했습니다. 

EXAONE 4.0 모델군은 두 개의 모델: 높은 성능을 위해 최적화된 32B 중형 모델, 그리고 온-디바이스 활용을 위해 디자인된 1.2B 소형 모델으로 구성되어 있습니다.

EXAONE 4.0의 모델 구조는 이전 EXAONE 모델들과 다른 아키텍처 디자인을 채택했습니다.

1. **Hybrid Attention**: 32B 모델은 *Local attention (sliding window attention)*과 *Global attention (full attention)*을 3:1 비율로 연결한 hybrid attention 구조를 채택했습니다. 또한 전체 문맥을 더 잘 이해할 수 있도록 global attention에서 RoPE를 사용하지 않았습니다.
2. **QK-Reorder-Norm**: 더 나은 downstream tasks 성능을 위해 연산량의 증가를 감수하며 전통적으로 사용되고 있던 Pre-LN 방식을 변경했습니다. LayerNorm의 위치를 attention과 MLP의 출력에 적용되도록 재배치했고, Q와 K projection 직후에도 RMS normalization을 추가했습니다. 

더 자세한 정보는 [기술 보고서](https://arxiv.org/abs/2507.11407), [HuggingFace 논문](https://huggingface.co/papers/2507.11407), [블로그](https://www.lgresearch.ai/blog/view?seq=576), [공식 GitHub](https://github.com/LG-AI-EXAONE/EXAONE-4.0) 페이지를 참고해주시길 바랍니다.

공개된 모든 모델 체크포인트는 [HuggingFace 콜렉션](https://huggingface.co/collections/LGAI-EXAONE/exaone-40-686b2e0069800c835ed48375)에서 확인할 수 있습니다.


## 모델 세부 정보

| Model Configuration | 32B | 1.2B |
|:-------------------|:-----:|:------:|
| d_model | 5,120 | 2,048 |
| Number of layers | 64 | 30 |
| Normalization | QK-Reorder-LN | QK-Reorder-LN |
| Non-linearity | SwiGLU | SwiGLU |
| Feedforward dimension | 27,392 | 4,096 |
| Attention type | Hybrid (3:1 Local-Global) | Global |
| Head type | GQA | GQA |
| Number of heads | 40 | 32 |
| Number of KV heads | 8 | 8 |
| Head size | 128 | 64 |
| Max sequence length | 131,072 | 65,536 |
| RoPE theta | 1,000,000 | 1,000,000 |
| Tokenizer | BBPE | BBPE |
| Vocab size | 102,400 | 102,400 |
| Tied word embedding | False | True |
| Knowledge cut-off | Nov. 2024 | Nov. 2024 |


## 사용 팁

### Non-reasoning mode

일반적인 대화의 경우 아래 예제와 같이 EXAONE 4.0을 사용할 수 있습니다.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LGAI-EXAONE/EXAONE-4.0-32B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="bfloat16",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 원하는 입력을 선택하세요
prompt = "Explain how wonderful you are"
prompt = "Explica lo increíble que eres"
prompt = "너가 얼마나 대단한지 설명해 봐"

messages = [
    {"role": "user", "content": prompt}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=128,
    do_sample=False,
)
print(tokenizer.decode(output[0]))
```

### Reasoning mode

The EXAONE 4.0 models have reasoning capabilities for handling complex problems. You can activate reasoning mode by using the `enable_thinking=True` argument with the tokenizer, which opens a reasoning block that starts with `<think>` tag without closing it.

EXAONE 4.0 모델군은 복잡한 문제를 해결하기 위한 사고 추론 능력을 갖추고 있습니다. 토크나이저에서 `enable_thinking=True` 인자를 사용해서 reasoning mode로 모델을 사용할 수 있습니다. 이 경우 `<think>` 토큰으로 추론 블록을 연 뒤, 닫지 않고 추론을 시작합니다.

```python
messages = [
    {"role": "user", "content": "Which one is bigger, 3.12 vs 3.9?"}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=True,
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=128,
    do_sample=True,
    temperature=0.6,
    top_p=0.95
)
print(tokenizer.decode(output[0]))
```

> [!IMPORTANT]
> 모델을 reasoning mode로 사용할 경우, 생성되는 답변이 sampling parameters에 굉장히 민감합니다. 따라서 더 나은 생성 품질을 위해 공식 [Usage Guideline](https://github.com/LG-AI-EXAONE/EXAONE-4.0#usage-guideline)를 참조해 주시길 바랍니다.

### Agentic tool use

EXAONE 4.0 모델은 도구 사용 능력을 갖춘 덕분에 Agent로 사용할 수 있습니다. 이를 위해서는 아래 예제와 같이 도구 명세를 모델에게 제공해 주어야 합니다.

```python
import random

def roll_dice(max_num: int):
    return random.randint(1, max_num)

tools = [
    {
        "type": "function",
        "function": {
            "name": "roll_dice",
            "description": "Roll a dice with the number 1 to N. User can select the number N.",
            "parameters": {
                "type": "object",
                "required": ["max_num"],
                "properties": {
                    "max_num": {
                        "type": "int",
                        "description": "Max number of the dice"
                    }
                }
            }
        }
    }
]

messages = [
    {"role": "user", "content": "Roll D6 dice twice!"}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    tools=tools,
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
)
print(tokenizer.decode(output[0]))
```

## Exaone4Config

[[autodoc]] Exaone4Config

## Exaone4Model

[[autodoc]] Exaone4Model
    - forward

## Exaone4ForCausalLM

[[autodoc]] Exaone4ForCausalLM
    - forward

## Exaone4ForSequenceClassification

[[autodoc]] Exaone4ForSequenceClassification
    - forward

## Exaone4ForTokenClassification

[[autodoc]] Exaone4ForTokenClassification
    - forward

## Exaone4ForQuestionAnswering

[[autodoc]] Exaone4ForQuestionAnswering
    - forward