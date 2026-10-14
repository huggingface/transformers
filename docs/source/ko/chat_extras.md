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

# 도구와 RAG[[Tools-and-RAG]]

[`~PreTrainedTokenizerBase.apply_chat_template`] 메소드는 채팅 메시지 외에도 문자열, 리스트, 딕셔너리 등 거의 모든 종류의 추가 인수 타입을 지원합니다. 이를 통해 다양한 사용 상황에서 채팅 템플릿을 활용할 수 있습니다.

이 가이드에서는 도구 및 검색 증강 생성(RAG)과 함께 채팅 템플릿을 사용하는 방법을 보여드립니다.

## 도구[[Tools]]

도구는 대규모 언어 모델(LLM)이 특정 작업을 수행하기 위해 호출할 수 있는 함수입니다. 이는 실시간 정보, 계산 도구 또는 대규모 데이터베이스 접근 등을 통해 대화형 에이전트의 기능을 확장하는 강력한 방법입니다.

도구를 만들 때는 아래 규칙을 따르세요.

1. 함수는 기능을 잘 설명하는 이름을 가져야 합니다.
2. 함수의 인수는 함수 헤더에 타입 힌트를 포함해야 합니다(`Args` 블록에는 포함하지 마세요).
3. 함수에는 [Google 스타일](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) 의 독스트링(docstring)이 포함되어야 합니다.
4. 함수에 반환 타입과 `Returns` 블록을 포함할 수 있지만, 도구를 활용하는 대부분의 모델에서 이를 사용하지 않기 때문에 무시할 수 있습니다.

주어진 위치의 현재 온도와 풍속을 가져오는 도구의 예시는 아래와 같습니다.

```py
def get_current_temperature(location: str, unit: str) -> float:
    """
    주어진 위치의 현재 온도를 가져옵니다.
    
    Args:
        location: 온도를 가져올 위치, "도시, 국가" 형식
        unit: 온도를 반환할 단위. (선택지: ["celsius(섭씨)", "fahrenheit(화씨)"])
    Returns:
        주어진 위치의 지정된 단위로 표시된 현재 온도(float 자료형).
    """
    return 22.  # 실제 함수라면 아마 진짜로 기온을 가져와야겠죠!

def get_current_wind_speed(location: str) -> float:
    """
    주어진 위치의 현재 풍속을 km/h 단위로 가져옵니다.
    
    Args:
        location: 온도를 가져올 위치, "도시, 국가" 형식
    Returns:
        주어진 위치의 현재 풍속(km/h, float 자료형).
    """
    return 6.  # 실제 함수라면 아마 진짜로 풍속을 가져와야겠죠!

tools = [get_current_temperature, get_current_wind_speed]
```

[NousResearch/Hermes-2-Pro-Llama-3-8B](https://hf.co/NousResearch/Hermes-2-Pro-Llama-3-8B)와 같이 도구 사용을 지원하는 모델과 토크나이저를 가져오세요. 하드웨어가 지원된다면 [Command-R](./model_doc/cohere)이나 [Mixtral-8x22B](./model_doc/mixtral)와 같은 더 큰 모델도 고려할 수 있습니다.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained( "NousResearch/Hermes-2-Pro-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained( "NousResearch/Hermes-2-Pro-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained( "NousResearch/Hermes-2-Pro-Llama-3-8B", torch_dtype=torch.bfloat16, device_map="auto")
```

채팅 메시지를 생성합니다.

```py
messages = [
  {"role": "system", "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location."},
  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
]
```

`messages`와 도구 목록 `tools`를 [`~PreTrainedTokenizerBase.apply_chat_template`]에 전달한 뒤, 이를 모델의 입력으로 사용하여 텍스트를 생성할 수 있습니다.

```py
inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):]))
```

```txt
<tool_call>
{"arguments": {"location": "Paris, France", "unit": "celsius"}, "name": "get_current_temperature"}
</tool_call><|im_end|>
```

채팅 모델은 독스트링(docstring)에 정의된 형식에 따라 `get_current_temperature` 함수에 올바른 매개변수를 전달해 호출했습니다. 파리를 기준으로 위치를 프랑스로 추론했으며, 온도 단위는 섭씨를 사용해야 한다고 판단했습니다.

이제 `get_current_temperature` 함수와 해당 인수들을 `tool_call` 딕셔너리에 담아 채팅 메시지에 추가합니다. `tool_call` 딕셔너리는 `system`이나 `user`가 아닌 `assistant` 역할로 제공되어야 합니다.

> [!WARNING]
> OpenAI API는 `tool_call` 형식으로 JSON 문자열을 사용합니다. Transformers에서 사용할 경우 딕셔너리를 요구하기 때문에, 오류가 발생하거나 모델이 이상하게 동작할 수 있습니다.

<hfoptions id="tool-call">
<hfoption id="Llama">

```py
tool_call = {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})
```

어시스턴트가 함수 출력을 읽고 사용자와 채팅할 수 있도록 합니다.

```py
inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

```txt
The temperature in Paris, France right now is approximately 12°C (53.6°F).<|im_end|>
```

</hfoption>
<hfoption id="Mistral/Mixtral">

[Mistral](./model_doc/mistral) 및 [Mixtral](./model_doc/mixtral) 모델의 경우 추가적으로 `tool_call_id`가 필요합니다. `tool_call_id`는 9자리 영숫자 문자열로 생성되어 `tool_call` 딕셔너리의 `id` 키에 할당됩니다.

```py
tool_call_id = "9Ae3bDc2F"
tool_call = {"name": "get_current_temperature", "arguments": {"location": "Paris, France", "unit": "celsius"}}
messages.append({"role": "assistant", "tool_calls": [{"type": "function", "id": tool_call_id, "function": tool_call}]})
```

```py
inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):]))
```

</hfoption>
</hfoptions>

## 스키마[[Schema]]

[`~PreTrainedTokenizerBase.apply_chat_template`]은 함수를 [JSON 스키마](https://json-schema.org/learn/getting-started-step-by-step)로 변환하여 채팅 템플릿에 전달합니다. LLM은 함수 내부의 코드를 보지 못합니다. 다시 말해, LLM은 함수가 기술적으로 어떻게 작동하는지는 신경 쓰지 않고, 함수의 **정의**와 **인수**만 참조합니다.

함수가 앞서 나열된 규칙을 따르면, 내부에서 JSON 스키마가 자동으로 생성됩니다. 하지만 더 나은 가독성이나 디버깅을 위해 [get_json_schema](https://github.com/huggingface/transformers/blob/14561209291255e51c55260306c7d00c159381a5/src/transformers/utils/chat_template_utils.py#L205)를 사용하여 스키마를 수동으로 변환할 수 있습니다.

```py
from transformers.utils import get_json_schema

def multiply(a: float, b: float):
    """
    두 숫자를 곱하는 함수
    
    Args:
        a: 곱할 첫 번째 숫자
        b: 곱할 두 번째 숫자
    """
    return a * b

schema = get_json_schema(multiply)
print(schema)
```

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

스키마를 편집하거나 처음부터 직접 작성할 수 있습니다. 이를 통해 더 복잡한 함수에 대한 정확한 스키마를 유연하게 정의할 수 있습니다.

> [!WARNING]
> 함수 시그니처를 단순하게 유지하고 인수를 최소한으로 유지하세요. 이러한 함수는 중첩된 인수를 가진 복잡한 함수에 비해 모델이 더 쉽게 이해하고 사용할 수 있습니다.

아래 예시는 스키마를 수동으로 작성한 다음 [`~PreTrainedTokenizerBase.apply_chat_template`]에 전달하는 방법을 보여줍니다.

```py
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

## RAG[[RAG]]

검색 증강 생성(Retrieval-augmented generation, RAG) 모델은 쿼리를 반환하기 전에 문서를 검색해 추가 정보를 얻어 모델이 기존에 가지고 있던 지식을 확장시킵니다. RAG 모델의 경우, [`~PreTrainedTokenizerBase.apply_chat_template`]에 `documents` 매개변수를 추가하세요. 이 `documents` 매개변수는 문서 목록이어야 하며, 각 문서는 `title`과 `content` 키를 가진 단일 딕셔너리여야 합니다.

> [!TIP]
> RAG를 위한 `documents` 매개변수는 폭넓게 지원되지 않으며 많은 모델들이 `documents`를 무시하는 채팅 템플릿을 가지고 있습니다. 모델이 `documents`를 지원하는지 확인하려면 모델 카드를 읽거나 `print(tokenizer.chat_template)`를 실행하여 `documents` 키가 있는지 확인하세요. [Command-R](https://hf.co/CohereForAI/c4ai-command-r-08-2024)과 [Command-R+](https://hf.co/CohereForAI/c4ai-command-r-plus-08-2024)는 모두 RAG 채팅 템플릿에서 `documents`를 지원합니다.

모델에 전달할 문서 목록을 생성하세요.

```py
documents = [
    {
        "title": "The Moon: Our Age-Old Foe", 
        "text": "Man has always dreamed of destroying the moon. In this essay, I shall..."
    },
    {
        "title": "The Sun: Our Age-Old Friend",
        "text": "Although often underappreciated, the sun provides several notable benefits..."
    }
]
```

[`~PreTrainedTokenizerBase.apply_chat_template`]에서 `chat_template="rag"`를 설정하고 응답을 생성하세요.

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-v01-4bit")
model = AutoModelForCausalLM.from_pretrained("CohereForAI/c4ai-command-r-v01-4bit", device_map="auto")
device = model.device # 모델을 가져온 장치 확인

# 대화 입력 정의
conversation = [
    {"role": "user", "content": "What has Man always dreamed of?"}
]

input_ids = tokenizer.apply_chat_template(
    conversation=conversation,
    documents=documents,
    chat_template="rag",
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt").to(device)

# 응답 생성
generated_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    )

# 생성된 텍스트를 디코딩하고 생성 프롬프트와 함께 출력
generated_text = tokenizer.decode(generated_tokens[0])
print(generated_text)
```
