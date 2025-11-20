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

# DeepSeek-V3[[deepseek-v3]]

## 개요[[overview]]

DeepSeek-V3 모델은 [DeepSeek-V3 기술 보고서](https://huggingface.co/papers/2412.19437)에서 DeepSeek-AI 팀에 의해 제안되었습니다.

논문의 초록은 다음과 같습니다.
총 671B개의 파라미터를 가지며 토큰당 37B개가 활성화되는 강력한 Mixture-of-Experts(MoE) 언어 모델인 DeepSeek-V3를 소개합니다. 효율적인 추론과 비용 효율적인 훈련을 달성하기 위해, DeepSeek-V3는 DeepSeek-V2에서 철저히 검증된 Multi-head Latent Attention(MLA) 및 DeepSeekMoE 아키텍처를 채택했습니다. 나아가 DeepSeek-V3는 로드 밸런싱을 위한 보조 손실 없는 전략을 개척하고, 더 강력한 성능을 위해 다중 토큰 예측 훈련 목표를 설정합니다. 저희는 14.8조 개의 다양하고 고품질의 토큰으로 DeepSeek-V3를 사전 훈련했으며, 그 잠재력을 완전히 활용하기 위해 지도 파인튜닝 및 강화 학습 단계를 거쳤습니다. 종합적인 평가 결과, DeepSeek-V3는 다른 오픈 소스 모델들을 능가하며 선도적인 비공개 소스 모델들과 필적하는 성능을 달성했음을 보여줍니다. 뛰어난 성능에도 불구하고 DeepSeek-V3의 전체 훈련에는 278.8만 H800 GPU 시간만이 소요되었습니다. 또한, 훈련 과정이 매우 안정적입니다. 전체 훈련 과정 동안 복구 불가능한 손실 급증을 경험하거나 롤백을 수행한 적이 없습니다. 모델 체크포인트는 https://github.com/deepseek-ai/DeepSeek-V3 에서 확인할 수 있습니다.

## 한계 및 기여 요청![[limitations-and-call-for-contribution!]]

저희는 이 코드를 커뮤니티 기반으로 만들게 되어 매우 기쁘며, 여러분이 다음 사항들을 어떻게 최적화할 수 있는지 확인하고 싶습니다.

- 현재 구현은 "기본적인" 어텐션 계산을 사용합니다. 따라서 실제 Multi-head Latent Attention (MLA) 가 아닙니다.
- 현재 구현은 전문가들을 순회하는 루프를 사용합니다. 이는 교체되어야 하기에 `integrations/tensor_parallel`의 `get_packed_weights`를 사용하는 것을 제안합니다.
- 현재 구현에서는 ROPE에 EleutherAI의 수식을 사용하지만, 원본 수식을 적용하면 더 효율적일 것입니다! (단, 기존 API는 그대로 준수해야 합니다)
- generation config 또는 config shape의 문제일 것으로 추정되는 문제로 인해 정적 캐시는 지원되지 않습니다.

### 사용 팁[[usage-tips]]
이 모델은 효율적인 추론과 비용 효율적인 훈련을 위해 Multi-head Latent Attention (MLA) 및 DeepSeekMoE 아키텍처를 사용합니다. 로드 밸런싱을 위한 보조 손실이 없는 전략과 다중 토큰 예측 훈련 목표를 채택합니다. 이 모델은 14.8조 개의 토큰으로 사전 훈련되고 지도 파인튜닝 및 강화 학습 단계를 거친 후 다양한 언어 작업에 사용될 수 있습니다.

`FP8`로 모델을 자동으로 실행할 수 있으며, 8개의 H100으로 구성된 2개 노드면 충분할 것입니다!

```python
# `run_deepseek_v1.py`
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(30)

tokenizer = AutoTokenizer.from_pretrained("deepseek-r1")

chat = [
  {"role": "user", "content": "안녕하세요, 어떻게 지내세요?"},
  {"role": "assistant", "content": "저는 잘 지내요. 오늘 무엇을 도와드릴까요?"},
  {"role": "user", "content": "채팅 템플릿이 어떻게 작동하는지 보여주고 싶어요!"},
]


model = AutoModelForCausalLM.from_pretrained("deepseek-r1", device_map="auto", torch_dtype=torch.bfloat16)
inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
import time
start = time.time()
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.batch_decode(outputs))
print(time.time()-start)
```
생성된 결과는 다음과 같습니다.

``````
<｜Assistant｜><think>
좋아요, 사용자는 채팅 템플릿이 어떻게 작동하는지 보여주고 싶어 하는군요. 이게 무슨 의미인지 분석해 보겠습니다. 채팅 템플릿은 대화 데이터를 구조화하는 것인데, 특히 특정 입력 형식이 필요한 모델에 중요합니다. 아마도 OpenAI 같은 API에서 메시지가 역할(사용자, 어시스턴트, 시스템)과 함께 형식화되는 방식을 말하는 것일 수 있습니다.

먼저, 채팅 템플릿이 무엇인지 설명해야겠습니다. 이는 모델이 이해할 수 있는 구조화된 형식으로 대화 데이터를 포맷하는 과정입니다. 여기에는 보통 역할과 콘텐츠가 포함됩니다. 예를 들어, 사용자 메시지, 어시스턴트 응답, 시스템 메시지는 각각 고유한 역할 태그를 가집니다.

사용자는 예시를 원할 수 있습니다. 간단한 대화를 생각해 보죠. 사용자가 "안녕하세요, 잘 지내세요?"라고 말하고 어시스턴트가 "네, 잘 지내요. 오늘 무엇을 도와드릴까요?"라고 답합니다. 그런 다음 사용자가 채팅 템플릿을 보여주고 싶다고 이어갑니다. 따라서 예시에는 대화 기록과 새 메시지가 포함되어야 합니다.

Hugging Face의 Transformers와 같은 프레임워크에서는 Jinja2 템플릿을 사용하여 채팅 템플릿이 적용됩니다. 템플릿은 시스템 메시지를 결합한 다음, 적절한 태그와 함께 사용자와 어시스턴트 메시지를 반복하는 형식일 수 있습니다. 예를 들어, {% for message in messages %}를 사용하고 <|user|>, <|assistant|>와 같은 역할을 할당하는 것입니다.

메시지 배열 예시를 구성하고, 각 역할과 내용을 보여주어야겠습니다. 그런 다음 가상의 템플릿을 적용하여 모델이 사용하는 형식화된 문자열로 변환하는 과정을 보여줍니다. 또한, 모델마다 특수 토큰을 사용하거나 역할 레이블이 다른 것처럼 템플릿 요구 사항이 다르다는 점도 언급해야 합니다.

잠깐, 사용자가 "채팅 템플릿"을 보여주는 맥락에서 언급했습니다. 아마도 발표할 수 있는 실용적인 예시를 원하는 것일 수 있습니다. 따라서 코드 스니펫이나 구조화된 데이터 예시를 제공하는 것이 도움이 될 것입니다. 일반적인 메시지 배열과 템플릿이 적용된 결과물을 개략적으로 설명해 보겠습니다.

또한, 적절한 템플릿팅은 모델이 대화 흐름을 이해하도록 보장하며, 이는 일관된 응답을 생성하는 데 중요합니다. 왜 이것이 중요한지에 대한 메모, 예를 들어 컨텍스트 유지 및 역할별 처리의 중요성을 포함할 수도 있겠습니다.

흔한 실수나 피해야 할 점이 있는지 확인해 보겠습니다. 예를 들어, 태그를 제대로 닫지 않거나 역할이 일치하지 않는 경우입니다. 하지만 사용자가 묻지 않는 한 너무 자세할 수 있습니다. 먼저 긍정적인 예시에 집중합시다.

모든 것을 종합하면, 응답에는 예시 메시지 배열, 적용된 템플릿, 그리고 최종적으로 형식화된 문자열이 포함되어야 합니다. 자리 표시자로 꺾쇠괄호나 특수 토큰을 사용할 수 있습니다. 또한, 이것이 구조화된 데이터로 모델을 훈련하거나 파인튜닝하는 데 도움이 된다는 점도 언급해야 합니다.

이것이 확실한 접근 방식인 것 같습니다. 명확하게 만들기 위해 단계별로 구조화해 보겠습니다.
</think>

채팅 템플릿은 대화 데이터(예: 사용자/어시스턴트 상호작용)를 언어 모델이 이해할 수 있는 형식으로 구조화하는 방법입니다. 이는 특히 입력에서 역할(사용자, 어시스턴트, 시스템 등)과 메시지를 명시적으로 구분해야 하는 다중 턴 대화를 처리하도록 훈련된 모델에 중요합니다. 예시와 함께 자세히 살펴보겠습니다!

---

### **1단계: 원본 대화 기록**
다음과 같은 대화가 있다고 가정해 보겠습니다:
- **사용자**: "안녕하세요, 어떻게 지내세요?"
- **어시스턴트**: "저는 잘 지내요. 오늘 무엇을 도와드릴까요?"
- **사용자**: "채팅 템플릿이 어떻게 작동하는지 보여주고 싶어요!"

---

### **2단계: 구조화된 메시지**
Hugging Face Transformers나 OpenAI 같은 프레임워크에서는 대화가 종종 `role`과 `content`를 가진 딕셔너리 리스트로 형식화됩니다.
```python
messages = [
    {"role": "user", "content": "안녕하세요, 어떻게 지내세요?"},
    {"role": "assistant", "content": "저는 잘 지내요. 오늘 무엇을 도와드릴까요?"},
    {"role": "user", "content": "채팅 템플릿이 어떻게 작동하는지 보여주고 싶어요!"},
]
```

---

### **3단계: 채팅 템플릿 적용**
**채팅 템플릿**은 이 구조화된 데이터를 모델에 맞는 단일 문자열로 변환합니다. 예를 들어, (Hugging Face에서 흔히 사용되는) Jinja 스타일 템플릿을 사용하면 다음과 같습니다.

```jinja
{% for message in messages %}
    {% if message['role'] == 'user' %}
        <|user|>{{ message['content'] }}<|end|>
    {% elif message['role'] == 'assistant' %}
        <|assistant|>{{ message['content'] }}<|end|>
    {% endif %}
{% endfor %}
<|assistant|>
```

---

### **4단계: 최종 템플릿 결과물**
위 템플릿을 저희 `messages` 리스트에 적용하면 다음과 같은 결과가 생성됩니다:
```text
<|user|>안녕하세요, 어떻게 지내세요?<|end|>
<|assistant|>저는 잘 지내요. 오늘 무엇을 도와드릴까요?<|end|>
<|user|>채팅 템플릿이 어떻게 작동하는지 보여주고 싶어요!<|end|>
<|assistant|>
```

이는 모델에게 다음을 알려줍니다:
1. 대화 기록 (사용자/어시스턴트 턴).
2. 모델이 응답을 생성할 차례 (`<|assistant|>`가 끝에 있음).

---

### **주요 참고사항**:
- **역할 분리**: `<|user|>`와 `<|assistant|>` 같은 태그는 모델이 화자를 구별하는 데 도움이 됩니다.
- **특수 토큰**: 모델은 종종 메시지 경계를 표시하기 위해 `<|end|>`와 같은 고유한 토큰을 사용합니다.
- **유연성**: 템플릿은 모델마다 다릅니다 (예: OpenAI는 태그 대신 `{"role": "user", "content": "..."}` 형식을 사용합니다).

---

### **이것이 왜 중요한가**:
- **일관성**: 모델이 대화 구조를 이해하도록 보장합니다.
- **컨텍스트 보존**: 다중 턴 대화의 흐름을 유지합니다.
- **정렬**: 더 나은 성능을 위해 모델이 훈련된 형식과 일치시킵니다.

더 자세히 알아보거나 특정 프레임워크(예: OpenAI, Llama, Mistral)의 구현을 보고 싶으신가요? 알려주세요! 😊<｜end of sentence｜>
``````

다음을 사용하여 실행하세요
```bash
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0|1 --rdzv-id an_id --rdzv-backend c10d --rdzv-endpoint master_addr:master_port run_deepseek_r1.py
```

만약 다음과 같은
```bash
[rank0]: ncclInternalError: Internal check failed.
[rank0]: Last error:
[rank0]: Bootstrap : no socket interface found
```
오류가 발생한다면, NCCL이 로드되지 않았을 가능성이 높다는 의미입니다.


## DeepseekV3Config[[deepseekv3config]]

[[autodoc]] DeepseekV3Config

## DeepseekV3Model[[deepseekv3model]]

[[autodoc]] DeepseekV3Model
    - forward

## DeepseekV3ForCausalLM[[deepseekv3forcausallm]]

[[autodoc]] DeepseekV3ForCausalLM
    - forward