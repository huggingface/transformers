<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 채팅 모델을 위한 템플릿[[templates-for-chat-models]]

## 소개[[introduction]]

요즘 LLM의 가장 흔한 활용 사례 중 하나는 **채팅**입니다. 채팅은 일반적인 언어 모델처럼 단일 문자열을 이어가는 대신 여러 개의 **메시지**로 구성된 대화를 이어갑니다. 이 대화에는 "사용자"나 "어시스턴트"와 같은 **역할**과 메시지 텍스트가 포함됩니다.

토큰화와 마찬가지로, 다양한 모델은 채팅에 대해 매우 다른 입력 형식을 기대합니다. 이것이 우리가 **채팅 템플릿**을 기능으로 추가한 이유입니다. 채팅 템플릿은 토크나이저의 일부입니다. 채팅 템플릿은 대화 목록을 모델이 기대하는 형식인 '단일 토큰화가 가능한 문자열'로 변환하는 방법을 지정합니다.

`BlenderBot` 모델을 사용한 간단한 예제를 통해 이를 구체적으로 살펴보겠습니다. BlenderBot은 기본적으로 매우 간단한 템플릿을 가지고 있으며, 주로 대화 라운드 사이에 공백을 추가합니다:

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

>>> chat = [
...    {"role": "user", "content": "Hello, how are you?"},
...    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...    {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>>> tokenizer.apply_chat_template(chat, tokenize=False)
" Hello, how are you?  I'm doing great. How can I help you today?   I'd like to show off how chat templating works!</s>"
```

전체 채팅이 하나의 문자열로 압축된 것을 확인할 수 있습니다. 기본 설정인 `tokenize=True`를 사용하면, 그 문자열도 토큰화됩니다. 더 복잡한 템플릿을 사용하기 위해 `mistralai/Mistral-7B-Instruct-v0.1` 모델을 사용해 보겠습니다.

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

>>> chat = [
...   {"role": "user", "content": "Hello, how are you?"},
...   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
...   {"role": "user", "content": "I'd like to show off how chat templating works!"},
... ]

>>> tokenizer.apply_chat_template(chat, tokenize=False)
"<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]"
```

이번에는 토크나이저가 [INST]와 [/INST] 제어 토큰을 추가하여 사용자 메시지의 시작과 끝을 표시했습니다(어시스턴트 메시지 제외). Mistral-instruct는 이러한 토큰으로 훈련되었지만, BlenderBot은 그렇지 않았습니다.

## 채팅 템플릿을 어떻게 사용하나요?[[how-do-i-use-chat-templates]]

위의 예에서 볼 수 있듯이 채팅 템플릿은 사용하기 쉽습니다. `role`과 `content` 키가 포함된 메시지 목록을 작성한 다음, [`~PreTrainedTokenizer.apply_chat_template`] 메서드에 전달하기만 하면 됩니다. 이렇게 하면 바로 사용할 수 있는 출력이 생성됩니다! 모델 생성의 입력으로 채팅 템플릿을 사용할 때, `add_generation_prompt=True`를 사용하여 [생성 프롬프트](#what-are-generation-prompts)를 추가하는 것도 좋은 방법입니다.

다음은 `Zephyr` 어시스턴트 모델을 사용하여 `model.generate()`의 입력을 준비하는 예제입니다:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)   # 여기서 bfloat16 사용 및/또는 GPU로 이동할 수 있습니다.


messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenizer.decode(tokenized_chat[0]))
```
이렇게 하면 Zephyr가 기대하는 입력 형식의 문자열이 생성됩니다.
```text
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s> 
<|user|>
How many helicopters can a human eat in one sitting?</s> 
<|assistant|>
```

이제 입력이 Zephyr에 맞게 형식이 지정되었으므로 모델을 사용하여 사용자의 질문에 대한 응답을 생성할 수 있습니다:

```python
outputs = model.generate(tokenized_chat, max_new_tokens=128) 
print(tokenizer.decode(outputs[0]))
```

이렇게 하면 다음과 같은 결과가 나옵니다:

```text
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s> 
<|user|>
How many helicopters can a human eat in one sitting?</s> 
<|assistant|>
Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all.
```

이제 쉬워졌죠!

## 채팅을 위한 자동화된 파이프라인이 있나요?[[is-there-an-automated-pipeline-for-chat]]

네, 있습니다! 우리의 텍스트 생성 파이프라인은 채팅 입력을 지원하여 채팅 모델을 쉽게 사용할 수 있습니다. 이전에는 "ConversationalPipeline" 클래스를 사용했지만, 이제는 이 기능이 [`TextGenerationPipeline`]에 통합되었습니다. 이번에는 파이프라인을 사용하여 `Zephyr` 예제를 다시 시도해 보겠습니다:

```python
from transformers import pipeline

pipe = pipeline("text-generation", "HuggingFaceH4/zephyr-7b-beta")
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
print(pipe(messages, max_new_tokens=128)[0]['generated_text'][-1])  # 어시스턴트의 응답을 출력합니다.
```

```text
{'role': 'assistant', 'content': "Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all."}
```

파이프라인은 토큰화와 `apply_chat_template` 호출 의 세부 사항을 모두 처리해주기 때문에, 모델에 채팅 템플릿이 있으면 파이프라인을 초기화하고 메시지 목록을 전달하기만 하면 됩니다!


## "생성 프롬프트"란 무엇인가요?[[what-are-generation-prompts]]

`apply_chat_template` 메서드에는 `add_generation_prompt` 인수가 있다는 것을 눈치챘을 것입니다. 이 인수는 템플릿에 봇 응답의 시작을 나타내는 토큰을 추가하도록 지시합니다. 예를 들어, 다음과 같은 채팅을 고려해 보세요:

```python
messages = [
    {"role": "user", "content": "Hi there!"},
    {"role": "assistant", "content": "Nice to meet you!"},
    {"role": "user", "content": "Can I ask a question?"}
]
```

Zephyr 예제에서 보았던 것과 같이, 생성 프롬프트 없이 ChatML 템플릿을 사용한다면 다음과 같이 보일 것입니다:

```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
"""<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
"""
```

생성 프롬프트가 **있는** 경우는 다음과 같습니다:

```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
"""<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant
"""
```

이번에는 봇 응답의 시작을 나타내는 토큰을 추가한 것을 주목하세요. 이렇게 하면 모델이 텍스트를 생성할 때 사용자의 메시지를 계속하는 대신 봇 응답을 작성하게 됩니다. 기억하세요, 채팅 모델은 여전히 언어 모델일 뿐이며, 그들에게 채팅은 특별한 종류의 텍스트일 뿐입니다! 적절한 제어 토큰으로 안내해야 채팅 모델이 무엇을 해야 하는지 알 수 있습니다.

모든 모델이 생성 프롬프트를 필요로 하는 것은 아닙니다. BlenderBot과 LLaMA 같은 일부 모델은 봇 응답 전에 특별한 토큰이 없습니다. 이러한 경우 `add_generation_prompt` 인수는 효과가 없습니다. `add_generation_prompt`의 정확한 효과는 사용 중인 템플릿에 따라 다릅니다.



## 채팅 템플릿을 훈련에 사용할 수 있나요?[[can-i-use-chat-templates-in-training]]

네! 이 방법은 채팅 템플릿을 모델이 훈련 중에 보는 토큰과 일치하도록 하는 좋은 방법입니다. 데이터 세트에 대한 전처리 단계로 채팅 템플릿을 적용하는 것이 좋습니다. 그 후에는 다른 언어 모델 훈련 작업과 같이 계속할 수 있습니다. 훈련할 때는 일반적으로 `add_generation_prompt=False`로 설정해야 합니다. 어시스턴트 응답을 프롬프트하는 추가 토큰은 훈련 중에는 도움이 되지 않기 때문입니다. 예제를 보겠습니다:

```python
from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

chat1 = [
    {"role": "user", "content": "Which is bigger, the moon or the sun?"},
    {"role": "assistant", "content": "The sun."}
]
chat2 = [
    {"role": "user", "content": "Which is bigger, a virus or a bacterium?"},
    {"role": "assistant", "content": "A bacterium."}
]

dataset = Dataset.from_dict({"chat": [chat1, chat2]})
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
print(dataset['formatted_chat'][0])
```
다음과 같은 결과를 얻을 수 있습니다:
```text
<|user|>
Which is bigger, the moon or the sun?</s>
<|assistant|>
The sun.</s>
```

여기서부터는 일반적인 언어 모델 작업과 같이 `formatted_chat` 열을 사용하여 훈련을 계속하면 됩니다.

<Tip>
`apply_chat_template(tokenize=False)`로 텍스트를 형식화한 다음 별도의 단계에서 토큰화하는 경우, `add_special_tokens=False` 인수를 설정해야 합니다. `apply_chat_template(tokenize=True)`를 사용하는 경우에는 이 문제를 걱정할 필요가 없습니다!
기본적으로 일부 토크나이저는 토큰화할 때 `<bos>` 및 `<eos>`와 같은 특별 토큰을 추가합니다. 채팅 템플릿은 항상 필요한 모든 특별 토큰을 포함해야 하므로, 기본 `add_special_tokens=True`로 추가적인 특별 토큰을 추가하면 잘못되거나 중복되는 특별 토큰을 생성하여 모델 성능이 저하될 수 있습니다.
</Tip>

## 고급: 채팅 템플릿에 추가 입력 사용[[advanced-extra-inputs-to-chat-templates]]

`apply_chat_template`가 필요한 유일한 인수는 `messages`입니다. 그러나 `apply_chat_template`에 키워드 인수를 전달하면 템플릿 내부에서 사용할 수 있습니다. 이를 통해 채팅 템플릿을 다양한 용도로 사용할 수 있는 자유를 얻을 수 있습니다. 이러한 인수의 이름이나 형식에는 제한이 없어 문자열, 리스트, 딕셔너리 등을 전달할 수 있습니다.

그렇긴 하지만, 이러한 추가 인수의 일반적인 사용 사례로 '함수 호출을 위한 도구'나 '검색 증강 생성을 위한 문서'를 전달하는 것이 있습니다. 이러한 일반적인 경우에 대해 인수의 이름과 형식에 대한 몇 가지 권장 사항이 있으며, 이는 아래 섹션에 설명되어 있습니다. 우리는 모델 작성자에게 도구 호출 코드를 모델 간에 쉽게 전송할 수 있도록 채팅 템플릿을 이 형식과 호환되도록 만들 것을 권장합니다.

## 고급: 도구 사용 / 함수 호출[[advanced-tool-use--function-calling]]

"도구 사용" LLM은 답변을 생성하기 전에 외부 도구로서 함수를 호출할 수 있습니다. 도구 사용 모델에 도구를 전달할 때는 단순히 함수 목록을 `tools` 인수로 전달할 수 있습니다:

```python
import datetime

def current_time():
    """현재 현지 시간을 문자열로 가져옵니다."""
    return str(datetime.now())

def multiply(a: float, b: float):
    """
    두 숫자를 곱하는 함수
    
    인수:
        a: 곱할 첫 번째 숫자
        b: 곱할 두 번째 숫자
    """
    return a * b

tools = [current_time, multiply]

model_input = tokenizer.apply_chat_template(
    messages,
    tools=tools
)
```

이것이 올바르게 작동하려면 함수를 위 형식으로 작성해야 도구로 올바르게 구문 분석할 수 있습니다. 구체적으로 다음 규칙을 따라야 합니다:

- 함수는 설명적인 이름을 가져야 합니다.
- 모든 인수에는 타입 힌트가 있어야 합니다.
- 함수에는 표준 Google 스타일의 도크스트링이 있어야 합니다(즉, 초기 함수 설명 다음에 인수를 설명하는 `Args:` 블록이 있어야 합니다). 
- `Args:` 블록에는 타입을 포함하지 마세요. 즉, `a (int): The first number to multiply` 대신 `a: The first number to multiply`라고 작성해야 합니다. 타입 힌트는 함수 헤더에 있어야 합니다.
- 함수에는 반환 타입과 도크스트링에 `Returns:` 블록이 있을 수 있습니다. 그러나 대부분의 도구 사용 모델은 이를 무시하므로 이는 선택 사항입니다.


### 도구 결과를 모델에 전달하기[[passing-tool-results-to-the-model]]

위의 예제 코드는 모델에 사용할 수 있는 도구를 나열하는 데 충분하지만, 실제로 사용하고자 하는 경우는 어떻게 해야 할까요? 이러한 경우에는 다음을 수행해야 합니다:

1. 모델의 출력을 파싱하여 도구 이름과 인수를 가져옵니다.
2. 모델의 도구 호출을 대화에 추가합니다.
3. 해당 인수에 대응하는 함수를 호출합니다.
4. 결과를 대화에 추가합니다.

### 도구 사용 예제[[a-complete-tool-use-example]]

도구 사용 예제를 단계별로 살펴보겠습니다. 이 예제에서는 도구 사용 모델 중에서 성능이 가장 우수한 8B `Hermes-2-Pro` 모델을 사용할 것입니다. 메모리가 충분하다면, 더 큰 모델인 [Command-R](https://huggingface.co/CohereForAI/c4ai-command-r-v01) 또는 [Mixtral-8x22B](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)를 사용하는 것도 고려할 수 있습니다. 이 두 모델 모두 도구 사용을 지원하며 더 강력한 성능을 제공합니다.

먼저 모델과 토크나이저를 로드해 보겠습니다:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "NousResearch/Hermes-2-Pro-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision="pr/13")
model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16, device_map="auto")
```

다음으로, 도구 목록을 정의해 보겠습니다:

```python
def get_current_temperature(location: str, unit: str) -> float:
    """
    특정 위치의 현재 온도를 가져옵니다.
    
    인수:
        위치: 온도를 가져올 위치, "도시, 국가" 형식
        단위: 온도 단위 (선택지: ["celsius", "fahrenheit"])
    반환값:
        지정된 위치의 현재 온도를 지정된 단위로 반환, float 형식.
    """
    return 22.  # 이 함수는 실제로 온도를 가져와야 할 것입니다!

def get_current_wind_speed(location: str) -> float:
    """
    주어진 위치의 현재 풍속을 km/h 단위로 가져옵니다.
    
    인수:
        위치(location): 풍속을 가져올 위치, "도시, 국가" 형식
    반환값:
        주어진 위치의 현재 풍속을 km/h 단위로 반환, float 형식.
    """
    return 6.  # 이 함수는 실제로 풍속을 가져와야 할 것입니다!

tools = [get_current_temperature, get_current_wind_speed]
```

이제 봇을 위한 대화를 설정해 보겠습니다:

```python
messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
]
```

이제 채팅 템플릿을 적용하고 응답을 생성해 보겠습니다:

```python
inputs = tokenizer.apply_chat_template(messages, chat_template="tool_use", tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

결과는 다음과 같습니다:

```text
<tool_call>
{"arguments": {"location": "Paris, France", "unit": "celsius"}, "name": "get_current_temperature"}
</tool_call><|im_end|>
```

모델이 함수 호출을 유효한 인수로 수행했으며, 함수 도크스트링에 요청된 형식으로 호출했음을 알 수 있습니다. 모델은 우리가 프랑스의 파리를 지칭하고 있다는 것을 추론했고, 프랑스가 SI 단위의 본고장임을 기억하여 온도를 섭씨로 표시해야 한다고 판단했습니다.

모델의 도구 호출을 대화에 추가해 보겠습니다. 여기서 임의의 `tool_call_id`를 생성합니다. 이 ID는 모든 모델에서 사용되는 것은 아니지만, 여러 도구 호출을 한 번에 발행하고 각 응답이 어느 호출에 해당하는지 추적할 수 있게 해줍니다. 이 ID는 대화 내에서 고유해야 합니다.

```python
tool_call_id = "vAHdf3"  # 임의의 ID, 각 도구 호출마다 고유해야 함
tool_call = {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
messages.append({"role": "assistant", "tool_calls": [{"id": tool_call_id, "type": "function", "function": tool_call}]})
```


이제 도구 호출을 대화에 추가했으므로, 함수를 호출하고 결과를 대화에 추가할 수 있습니다. 이 예제에서는 항상 22.0을 반환하는 더미 함수를 사용하고 있으므로, 결과를 직접 추가하면 됩니다. 다시 한 번, `tool_call_id`는 도구 호출에 사용했던 ID와 일치해야 합니다.

```python
messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": "get_current_temperature", "content": "22.0"})
```

마지막으로, 어시스턴트가 함수 출력을 읽고 사용자와 계속 대화할 수 있도록 하겠습니다:

```python
inputs = tokenizer.apply_chat_template(messages, chat_template="tool_use", tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

결과는 다음과 같습니다:

```text
The current temperature in Paris, France is 22.0 ° Celsius.<|im_end|>
```

이것은 더미 도구와 단일 호출을 사용한 간단한 데모였지만, 동일한 기술을 사용하여 여러 실제 도구와 더 긴 대화를 처리할 수 있습니다. 이를 통해 실시간 정보, 계산 도구 또는 대규모 데이터베이스에 접근하여 대화형 에이전트의 기능을 확장할 수 있습니다.

<Tip>
위에서 보여준 도구 호출 기능은 모든 모델에서 사용되는 것은 아닙니다. 일부 모델은 도구 호출 ID를 사용하고, 일부는 함수 이름만 사용하여 결과와 도구 호출을 순서에 따라 매칭하며, 혼동을 피하기 위해 한 번에 하나의 도구 호출만 발행하는 모델도 있습니다. 가능한 많은 모델과 호환되는 코드를 원한다면, 여기에 보여준 것처럼 도구 호출을 구성하고, 모델이 발행한 순서대로 도구 결과를 반환하는 것을 권장합니다. 각 모델의 채팅 템플릿이 나머지 작업을 처리할 것입니다.
</Tip>

### 도구 스키마 이해하기[[understanding-tool-schemas]]

`apply_chat_template`의 `tools` 인수에 전달하는 각 함수는 [JSON 스키마](https://json-schema.org/learn/getting-started-step-by-step)로 변환됩니다. 이러한 스키마는 모델 채팅 템플릿에 전달됩니다. 즉, 도구 사용 모델은 함수 자체를 직접 보지 않으며, 함수 내부의 실제 코드를 보지 않습니다. 도구 사용 모델이 관심을 가지는 것은 함수 **정의**와 **인수**입니다. 함수가 무엇을 하고 어떻게 사용하는지에 관심이 있을 뿐, 어떻게 작동하는지는 중요하지 않습니다! 모델의 출력을 읽고 모델이 도구 사용을 요청했는지 감지하여, 인수를 도구 함수에 전달하고 채팅에서 응답을 반환하는 것은 여러분의 몫입니다.

위의 규격을 따른다면, 템플릿에 전달할 JSON 스키마 생성을 자동화하고 보이지 않게 처리하는 것이 좋습니다. 그러나 문제가 발생하거나 변환을 더 제어하고 싶다면 수동으로 변환을 처리할 수 있습니다. 다음은 수동 스키마 변환 예제입니다.

```python
from transformers.utils import get_json_schema

def multiply(a: float, b: float):
    """
    두 숫자를 곱하는 함수
    
    인수:
        a: 곱할 첫 번째 숫자
        b: 곱할 두 번째 숫자
    """
    return a * b

schema = get_json_schema(multiply)
print(schema)
```

이 결과는 다음과 같습니다:

```json
{
  "type": "function", 
  "function": {
    "name": "multiply", 
    "description": "A function that multiplies two numbers", 
    "parameters": {
      "type": "object", 
      "properties": {
        "a": {
          "type": "number", 
          "description": "The first number to multiply"
        }, 
        "b": {
          "type": "number",
          "description": "The second number to multiply"
        }
      }, 
      "required": ["a", "b"]
    }
  }
}
```

원한다면 이러한 스키마를 편집하거나 `get_json_schema`를 전혀 사용하지 않고 처음부터 직접 작성할 수도 있습니다. JSON 스키마는 `apply_chat_template`의 `tools` 인수에 직접 전달할 수 있습니다. 이를 통해 더 복잡한 함수에 대한 정밀한 스키마를 정의할 수 있게 됩니다. 그러나 스키마가 복잡할수록 모델이 처리하는 데 혼란을 겪을 가능성이 높아집니다! 가능한 한 간단한 함수 서명을 유지하고, 인수(특히 복잡하고 중첩된 인수)를 최소화하는 것을 권장합니다.

여기 직접 스키마를 정의하고 이를 `apply_chat_template`에 전달하는 예제가 있습니다:

```python
# 인수를 받지 않는 간단한 함수
current_time = {
  "type": "function", 
  "function": {
    "name": "current_time",
    "description": "Get the current local time as a string.",
    "parameters": {
      'type': 'object',
      'properties': {}
    }
  }
}

# 두 개의 숫자 인수를 받는 더 완전한 함수
multiply = {
  'type': 'function',
  'function': {
    'name': 'multiply',
    'description': 'A function that multiplies two numbers', 
    'parameters': {
      'type': 'object', 
      'properties': {
        'a': {
          'type': 'number',
          'description': 'The first number to multiply'
        }, 
        'b': {
          'type': 'number', 'description': 'The second number to multiply'
        }
      }, 
      'required': ['a', 'b']
    }
  }
}

model_input = tokenizer.apply_chat_template(
    messages,
    tools = [current_time, multiply]
)
```

## 고급: 검색 증강 생성[[advanced-retrieval-augmented-generation]]

"검색 증강 생성" 또는 "RAG" LLM은 쿼리에 응답하기 전에 문서의 코퍼스를 검색하여 정보를 얻을 수 있습니다. 이를 통해 모델은 제한된 컨텍스트 크기 이상으로 지식 기반을 크게 확장할 수 있습니다. RAG 모델에 대한 우리의 권장 사항은 템플릿이 `documents` 인수를 허용해야 한다는 것입니다. 이 인수는 각 "문서"가 `title`과 `contents` 키를 가지는 단일 dict인 문서 목록이어야 합니다. 이 형식은 도구에 사용되는 JSON 스키마보다 훨씬 간단하므로 별도의 도우미 함수가 필요하지 않습니다.


다음은 RAG 템플릿이 작동하는 예제입니다:


```python
document1 = {
    "title": "The Moon: Our Age-Old Foe",
    "contents": "Man has always dreamed of destroying the moon. In this essay, I shall..."
}

document2 = {
    "title": "The Sun: Our Age-Old Friend",
    "contents": "Although often underappreciated, the sun provides several notable benefits..."
}

model_input = tokenizer.apply_chat_template(
    messages,
    documents=[document1, document2]
)
```

## 고급: 채팅 템플릿은 어떻게 작동하나요?[[advanced-how-do-chat-templates-work]]

모델의 채팅 템플릿은 `tokenizer.chat_template` 속성에 저장됩니다. 채팅 템플릿이 설정되지 않은 경우 해당 모델 클래스의 기본 템플릿이 대신 사용됩니다. `BlenderBot`의 템플릿을 살펴보겠습니다:

```python

>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

>>> tokenizer.chat_template
"{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"
```

약간 복잡해 보일 수 있습니다. 읽기 쉽게 정리해 보겠습니다. 이 과정에서 추가하는 줄바꿈과 들여쓰기가 템플릿 출력에 포함되지 않도록 해야 합니다. 아래는 [공백을 제거하는](#trimming-whitespace) 팁입니다:

```
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- ' ' }}
    {%- endif %}
    {{- message['content'] }}
    {%- if not loop.last %}
        {{- '  ' }}
    {%- endif %}
{%- endfor %}
{{- eos_token }}
```

만약 이와 같은 형식을 처음 본다면, 이것은 [Jinja 템플릿](https://jinja.palletsprojects.com/en/3.1.x/templates/)입니다.
Jinja는 텍스트를 생성하는 간단한 코드를 작성할 수 있는 템플릿 언어입니다. 많은 면에서 코드와 구문이 파이썬과 유사합니다. 순수 파이썬에서는 이 템플릿이 다음과 같이 보일 것입니다:


```python
for idx, message in enumerate(messages):
    if message['role'] == 'user':
        print(' ')
    print(message['content'])
    if not idx == len(messages) - 1:  # Check for the last message in the conversation
        print('  ')
print(eos_token)
```

이 템플릿은 세 가지 일을 합니다:
1. 각 메시지에 대해, 메시지가 사용자 메시지인 경우 공백을 추가하고, 그렇지 않으면 아무것도 출력하지 않습니다.
2. 메시지 내용을 추가합니다.
3. 메시지가 마지막 메시지가 아닌 경우 두 개의 공백을 추가합니다. 마지막 메시지 후에는 EOS 토큰을 출력합니다.

이것은 매우 간단한 템플릿입니다. 제어 토큰을 추가하지 않으며, 이후 대화에서 모델이 어떻게 동작해야 하는지 지시하는 "시스템" 메시지를 지원하지 않습니다. 하지만 Jinja는 이러한 작업을 수행할 수 있는 많은 유연성을 제공합니다! LLaMA가 입력을 형식화하는 방식과 유사한 형식의 Jinja 템플릿을 살펴보겠습니다(실제 LLaMA 템플릿은 기본 시스템 메시지 처리와 일반적인 시스템 메시지 처리를 포함하고 있습니다 - 실제 코드에서는 이 템플릿을 사용하지 마세요!).

```
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + message['content'] + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- ' '  + message['content'] + ' ' + eos_token }}
    {%- endif %}
{%- endfor %}
```

이 템플릿을 잠시 살펴보면 무엇을 하는지 이해할 수 있습니다. 먼저, 각 메시지의 "role"에 따라 특정 토큰을 추가하여 누가 메시지를 보냈는지 모델에게 명확하게 알려줍니다. 또한 사용자, 어시스턴트 및 시스템 메시지는 각각 고유한 토큰으로 래핑되어 모델이 명확하게 구분할 수 있습니다.

## 고급: 채팅 템플릿 추가 및 편집[[advanced-adding-and-editing-chat-templates]]

### 채팅 템플릿을 어떻게 만들 수 있나요?[[how-do-i-create-a-chat-template]]

간단합니다. Jinja 템플릿을 작성하고 `tokenizer.chat_template`에 설정하기만 하면 됩니다. 다른 모델의 기존 템플릿을 시작점으로 사용하고 필요에 맞게 편집하는 것이 더 쉬울 것 입니다! 예를 들어, 위의 LLaMA 템플릿을 가져와 어시스턴트 메시지에 "[ASST]" 및 "[/ASST]"를 추가할 수 있습니다:

```
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- bos_token + '[INST] ' + message['content'].strip() + ' [/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '[ASST] '  + message['content'] + ' [/ASST]' + eos_token }}
    {%- endif %}
{%- endfor %}
```

이제 `tokenizer.chat_template` 속성을 설정하기만 하면 됩니다. 이렇게 하면 다음에 [`~PreTrainedTokenizer.apply_chat_template`]를 사용할 때 새롭게 설정한 템플릿이 사용됩니다! 이 속성은 `tokenizer_config.json` 파일에 저장되므로, [`~utils.PushToHubMixin.push_to_hub`]를 사용하여 새 템플릿을 허브에 업로드하고 모든 사용자가 모델에 맞는 템플릿을 사용할 수 있도록 할 수 있습니다!

```python
template = tokenizer.chat_template
template = template.replace("SYS", "SYSTEM")  # 시스템 토큰 변경
tokenizer.chat_template = template  # 새 템플릿 설정
tokenizer.push_to_hub("model_name")  # 새 템플릿을 허브에 업로드!
```

채팅 템플릿을 사용하는 [`~PreTrainedTokenizer.apply_chat_template`] 메소드는 [`TextGenerationPipeline`] 클래스에서 호출되므로, 올바른 채팅 템플릿을 설정하면 모델이 자동으로 [`TextGenerationPipeline`]과 호환됩니다.

<Tip>
모델을 채팅 용도로 미세 조정하는 경우, 채팅 템플릿을 설정하는 것 외에도 새 채팅 제어 토큰을 토크나이저에 특별 토큰으로 추가하는 것이 좋습니다. 특별 토큰은 절대로 분할되지 않으므로, 제어 토큰이 여러 조각으로 토큰화되는 것을 방지합니다. 또한, 템플릿에서 어시스턴트 생성의 끝을 나타내는 토큰으로 토크나이저의 `eos_token` 속성을 설정해야 합니다. 이렇게 하면 텍스트 생성 도구가 텍스트 생성을 언제 중지해야 할지 정확히 알 수 있습니다.
</Tip>


### 왜 일부 모델은 여러 개의 템플릿을 가지고 있나요?[[why-do-some-models-have-multiple-templates]]

일부 모델은 다른 사용 사례에 대해 다른 템플릿을 사용합니다. 예를 들어, 일반 채팅을 위한 템플릿과 도구 사용 또는 검색 증강 생성에 대한 템플릿을 별도로 사용할 수 있습니다. 이러한 경우 `tokenizer.chat_template`는 딕셔너리입니다. 이것은 약간의 혼란을 초래할 수 있으며, 가능한 한 모든 사용 사례에 대해 단일 템플릿을 사용하는 것을 권장합니다. `if tools is defined`와 같은 Jinja 문장과 `{% macro %}` 정의를 사용하여 여러 코드 경로를 단일 템플릿에 쉽게 래핑할 수 있습니다.

토크나이저에 여러 개의 템플릿이 있는 경우, `tokenizer.chat_template`는 템플릿 이름이 키인 `딕셔너리`입니다. `apply_chat_template` 메소드는 특정 템플릿 이름에 대한 특별한 처리를 합니다: 일반적으로 `default`라는 템플릿을 찾고, 찾을 수 없으면 오류를 발생시킵니다. 그러나 사용자가 `tools` 인수를 전달할 때 `tool_use`라는 템플릿이 존재하면 대신 그것을 사용합니다. 다른 이름의 템플릿에 접근하려면 `apply_chat_template()`의 `chat_template` 인수에 원하는 템플릿 이름을 전달하면 됩니다.

사용자에게 약간의 혼란을 줄 수 있으므로, 템플릿을 직접 작성하는 경우 가능한 한 단일 템플릿에 모든 것을 넣는 것을 권장합니다!

### 어떤 템플릿을 사용해야 하나요?[[what-template-should-i-use]]

이미 채팅용으로 훈련된 모델에 템플릿을 설정할 때는 템플릿이 훈련 중 모델이 본 메시지 형식과 정확히 일치하도록 해야 합니다. 그렇지 않으면 성능 저하를 경험할 가능성이 큽니다. 이는 모델을 추가로 훈련할 때도 마찬가지입니다. 채팅 토큰을 일정하게 유지하는 것이 최상의 성능을 얻는 방법입니다. 이는 토큰화와 매우 유사합니다. 훈련 중에 사용된 토큰화를 정확히 일치시킬 때 추론이나 미세 조정에서 최고의 성능을 얻을 수 있습니다.

반면에 처음부터 모델을 훈련시키거나 채팅용으로 기본 언어 모델을 미세 조정하는 경우, 적절한 템플릿을 선택할 수 있는 많은 자유가 있습니다. LLM은 다양한 입력 형식을 처리할 만큼 충분히 똑똑합니다. 인기 있는 선택 중 하나는 `ChatML` 형식이며, 이는 많은 사용 사례에 유연하게 사용할 수 있는 좋은 선택입니다. 다음과 같습니다:

```
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
{%- endfor %}
```

이 템플릿이 마음에 든다면, 코드에 바로 복사하여 사용할 수 있는 한 줄 버전을 제공하겠습니다. 이 한 줄 버전은 [생성 프롬프트](#what-are-generation-prompts)에 대한 편리한 지원도 포함하고 있지만, BOS나 EOS 토큰을 추가하지 않는다는 점에 유의하세요! 모델이 해당 토큰을 기대하더라도, `apply_chat_template`에 의해 자동으로 추가되지 않습니다. 즉, 텍스트는 `add_special_tokens=False`에 의해 토큰화됩니다. 이는 템플릿과 `add_special_tokens` 논리 간의 잠재적인 충돌을 피하기 위함입니다. 모델이 특별 토큰을 기대하는 경우, 템플릿에 직접 추가해야 합니다!


```python
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
```

이 템플릿은 각 메시지를 `<|im_start|>` 와 `<|im_end|>`토큰으로 감싸고, 역할을 문자열로 작성하여 훈련 시 사용하는 역할에 대한 유연성을 제공합니다. 출력은 다음과 같습니다:


```text
<|im_start|>system
You are a helpful chatbot that will do its best not to say anything so stupid that people tweet about it.<|im_end|>
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm doing great!<|im_end|>
```

"사용자", "시스템" 및 "어시스턴트" 역할은 채팅의 표준이며, 가능할 때 이를 사용하는 것을 권장합니다. 특히 모델이 [`TextGenerationPipeline`]과 잘 작동하도록 하려면 그렇습니다. 그러나 이러한 역할에만 국한되지 않습니다. 템플릿은 매우 유연하며, 어떤 문자열이든 역할로 사용할 수 있습니다.



### 채팅 템플릿을 추가하고 싶습니다! 어떻게 시작해야 하나요?[[i-want-to-add-some-chat-templates-how-should-i-get-started]]

채팅 모델이 있는 경우, 해당 모델의 `tokenizer.chat_template` 속성을 설정하고 [`~PreTrainedTokenizer.apply_chat_template`]를 사용하여 테스트한 다음 업데이트된 토크나이저를 허브에 푸시해야 합니다. 이는 모델 소유자가 아닌 경우에도 적용됩니다. 빈 채팅 템플릿을 사용하는 모델이나 여전히 기본 클래스 템플릿을 사용하는 모델을 사용하는 경우, [풀 리퀘스트](https://huggingface.co/docs/hub/repositories-pull-requests-discussions)를 모델 리포지토리에 열어 이 속성을 올바르게 설정할 수 있도록 하세요!

속성을 설정하면 끝입니다! `tokenizer.apply_chat_template`가 이제 해당 모델에 대해 올바르게 작동하므로, `TextGenerationPipeline`과 같은 곳에서도 자동으로 지원됩니다!

모델에 이 속성을 설정함으로써, 오픈 소스 모델의 전체 기능을 커뮤니티가 사용할 수 있도록 할 수 있습니다. 형식 불일치는 이 분야에서 오랫동안 성능을 저하시키는 문제였으므로, 이제 이를 끝낼 때입니다!

## 고급: 템플릿 작성 팁[[advanced-template-writing-tips]]

Jinja에 익숙하지 않은 경우, 채팅 템플릿을 작성하는 가장 쉬운 방법은 먼저 메시지를 원하는 방식으로 형식화하는 짧은 파이썬 스크립트를 작성한 다음, 해당 스크립트를 템플릿으로 변환하는 것입니다.

템플릿 핸들러는 `messages`라는 변수로 대화 기록을 받습니다. 파이썬에서와 마찬가지로 템플릿 내의 `messages`에 접근할 수 있으며, `{% for message in messages %}`로 반복하거나 `{{ messages[0] }}`와 같이 개별 메시지에 접근할 수 있습니다.

다음 팁을 사용하여 코드를 Jinja로 변환할 수도 있습니다:

### 공백 제거[[trimming-whitespace]]

기본적으로 Jinja는 블록 전후의 공백을 출력합니다. 이는 일반적으로 공백을 매우 정확하게 다루고자 하는 채팅 템플릿에서는 문제가 될 수 있습니다! 이를 피하기 위해 템플릿을 다음과 같이 작성하는 것이 좋습니다:

```
{%- for message in messages %}
    {{- message['role'] + message['content'] }}
{%- endfor %}
```

아래와 같이 작성하지 마세요:

```
{% for message in messages %}
    {{ message['role'] + message['content'] }}
{% endfor %}
```

`-`를 추가하면 블록 전후의 공백이 제거됩니다. 두 번째 예제는 무해해 보이지만, 줄바꿈과 들여쓰기가 출력에 포함될 수 있으며, 이는 원하지 않는 결과일 수 있습니다!

### 반복문[[for-loops]]

Jinja에서 반복문은 다음과 같습니다:

```
{%- for message in messages %}
    {{- message['content'] }}
{%- endfor %}
```

{{ 표현식 블록 }} 내부에 있는 모든 것이 출력으로 인쇄됩니다. `+`와 같은 연산자를 사용하여 표현식 블록 내부에서 문자열을 결합할 수 있습니다.

### 조건문[[if-statements]]

Jinja에서 조건문은 다음과 같습니다:

```
{%- if message['role'] == 'user' %}
    {{- message['content'] }}
{%- endif %}
```

파이썬이 공백을 사용하여 `for` 및 `if` 블록의 시작과 끝을 표시하는 반면, Jinja는 `{% endfor %}` 및 `{% endif %}`로 명시적으로 끝을 표시해야 합니다.

### 특수 변수[[special-variables]]

템플릿 내부에서는 `messages` 목록에 접근할 수 있을 뿐만 아니라 여러 다른 특수 변수에도 접근할 수 있습니다. 여기에는 `bos_token` 및 `eos_token`과 같은 특별 토큰과 앞서 논의한 `add_generation_prompt` 변수가 포함됩니다. 또한 `loop` 변수를 사용하여 현재 반복에 대한 정보를 얻을 수 있으며, 예를 들어 `{% if loop.last %}`를 사용하여 현재 메시지가 대화의 마지막 메시지인지 확인할 수 있습니다. `add_generation_prompt`가 `True`인 경우 대화 끝에 생성 프롬프트를 추가하는 예제는 다음과 같습니다:

```
{%- if loop.last and add_generation_prompt %}
    {{- bos_token + 'Assistant:\n' }}
{%- endif %}
```

### 비파이썬 Jinja와의 호환성[[compatibility-with-non-python-jinja]]

Jinja의 여러 구현은 다양한 언어로 제공됩니다. 일반적으로 동일한 구문을 사용하지만, 주요 차이점은 파이썬에서 템플릿을 작성할 때 파이썬 메소드를 사용할 수 있다는 점입니다. 예를 들어, 문자열에 `.lower()`를 사용하거나 딕셔너리에 `.items()`를 사용하는 것입니다. 이는 비파이썬 Jinja 구현에서 템플릿을 사용하려고 할 때 문제가 발생할 수 있습니다. 특히 JS와 Rust가 인기 있는 배포 환경에서는 비파이썬 구현이 흔합니다.

하지만 걱정하지 마세요! 모든 Jinja 구현에서 호환성을 보장하기 위해 템플릿을 쉽게 변경할 수 있는 몇 가지 방법이 있습니다:

- 파이썬 메소드를 Jinja 필터로 대체하세요. 일반적으로 같은 이름을 가지며, 예를 들어 `string.lower()`는 `string|lower`로, `dict.items()`는 `dict|items`로 대체할 수 있습니다. 주목할 만한 변경 사항은 `string.strip()`이 `string|trim`으로 바뀌는 것입니다. 더 자세한 내용은 Jinja 문서의 [내장 필터 목록](https://jinja.palletsprojects.com/en/3.1.x/templates/#builtin-filters)을 참조하세요.
- 파이썬에 특화된 `True`, `False`, `None`을 각각 `true`, `false`, `none`으로 대체하세요.
- 딕셔너리나 리스트를 직접 렌더링할 때 다른 구현에서는 결과가 다를 수 있습니다(예: 문자열 항목이 단일 따옴표에서 이중 따옴표로 변경될 수 있습니다). `tojson` 필터를 추가하면 일관성을 유지하는 데 도움이 됩니다.