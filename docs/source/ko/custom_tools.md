<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 사용자 정의 도구와 프롬프트[[custom-tools-and-prompts]]

<Tip>

Transformers와 관련하여 어떤 도구와 에이전트가 있는지 잘 모르신다면 [Transformers Agents](transformers_agents) 페이지를 먼저 읽어보시기 바랍니다. 

</Tip>

<Tip warning={true}>

Transformers Agents는 실험 중인 API로 언제든지 변경될 수 있습니다. 
API 또는 기반 모델이 변경되기 쉽기 때문에 에이전트가 반환하는 결과도 달라질 수 있습니다.

</Tip>

에이전트에게 권한을 부여하고 새로운 작업을 수행하게 하려면 사용자 정의 도구와 프롬프트를 만들고 사용하는 것이 무엇보다 중요합니다.
이 가이드에서는 다음과 같은 내용을 살펴보겠습니다:

- 프롬프트를 사용자 정의하는 방법
- 사용자 정의 도구를 사용하는 방법
- 사용자 정의 도구를 만드는 방법

## 프롬프트를 사용자 정의하기[[customizing-the-prompt]]

[Transformers Agents](transformers_agents)에서 설명한 것처럼 에이전트는 [`~Agent.run`] 및 [`~Agent.chat`] 모드에서 실행할 수 있습니다.
`run`(실행) 모드와 `chat`(채팅) 모드 모두 동일한 로직을 기반으로 합니다. 
에이전트를 구동하는 언어 모델은 긴 프롬프트에 따라 조건이 지정되고, 중지 토큰에 도달할 때까지 다음 토큰을 생성하여 프롬프트를 완수합니다.
`chat` 모드에서는 프롬프트가 이전 사용자 입력 및 모델 생성으로 연장된다는 점이 두 모드의 유일한 차이점입니다.
이를 통해 에이전트가 과거 상호작용에 접근할 수 있게 되므로 에이전트에게 일종의 메모리를 제공하는 셈입니다.

### 프롬프트의 구조[[structure-of-the-prompt]]

어떻게 프롬프트 사용자 정의를 잘 할 수 있는지 이해하기 위해 프롬프트의 구조를 자세히 살펴봅시다.
프롬프트는 크게 네 부분으로 구성되어 있습니다.

- 1. 도입: 에이전트가 어떻게 행동해야 하는지, 도구의 개념에 대한 설명.
- 2. 모든 도구에 대한 설명. 이는 런타임에 사용자가 정의/선택한 도구로 동적으로 대체되는 `<<all_tools>>` 토큰으로 정의됩니다.
- 3. 작업 예제 및 해당 솔루션 세트.
- 4. 현재 예제 및 해결 요청.

각 부분을 더 잘 이해할 수 있도록 짧은 버전을 통해 `run` 프롬프트가 어떻게 보이는지 살펴보겠습니다:

````text
I will ask you to perform a task, your job is to come up with a series of simple commands in Python that will perform the task.
[...]
You can print intermediate results if it makes sense to do so.

Tools:
- document_qa: This is a tool that answers a question about a document (pdf). It takes an input named `document` which should be the document containing the information, as well as a `question` that is the question about the document. It returns a text that contains the answer to the question.
- image_captioner: This is a tool that generates a description of an image. It takes an input named `image` which should be the image to the caption and returns a text that contains the description in English.
[...]

Task: "Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French."

I will use the following tools: `translator` to translate the question into English and then `image_qa` to answer the question on the input image.

Answer:
```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
answer = image_qa(image=image, question=translated_question)
print(f"The answer is {answer}")
```

Task: "Identify the oldest person in the `document` and create an image showcasing the result as a banner."

I will use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.

Answer:
```py
answer = document_qa(document, question="What is the oldest person?")
print(f"The answer is {answer}.")
image = image_generator("A banner showing " + answer)
```

[...]

Task: "Draw me a picture of rivers and lakes"

I will use the following
````

도입(*"도구:"* 앞의 텍스트)에서는 모델이 어떻게 작동하고 무엇을 해야 하는지 정확하게 설명합니다.
에이전트는 항상 같은 방식으로 작동해야 하므로 이 부분은 사용자 정의할 필요가 없을 가능성이 높습니다.

두 번째 부분(*"도구"* 아래의 글머리 기호)은 `run` 또는 `chat`을 호출할 때 동적으로 추가됩니다. 
정확히 `agent.toolbox`에 있는 도구 수만큼 글머리 기호가 있고, 각 글머리 기호는 도구의 이름과 설명으로 구성됩니다:

```text
- <tool.name>: <tool.description>
```

문서 질의응답 도구를 가져오고 이름과 설명을 출력해서 빠르게 확인해 보겠습니다.

```py
from transformers import load_tool

document_qa = load_tool("document-question-answering")
print(f"- {document_qa.name}: {document_qa.description}")
```

그러면 다음 결과가 출력됩니다:
```text
- document_qa: This is a tool that answers a question about a document (pdf). It takes an input named `document` which should be the document containing the information, as well as a `question` that is the question about the document. It returns a text that contains the answer to the question.
```

여기서 도구 이름이 짧고 정확하다는 것을 알 수 있습니다. 
설명은 두 부분으로 구성되어 있는데, 첫 번째 부분에서는 도구의 기능을 설명하고 두 번째 부분에서는 예상되는 입력 인수와 반환 값을 명시합니다.

에이전트가 도구를 올바르게 사용하려면 좋은 도구 이름과 도구 설명이 매우 중요합니다. 
에이전트가 도구에 대해 알 수 있는 유일한 정보는 이름과 설명뿐이므로, 이 두 가지를 정확하게 작성하고 도구 상자에 있는 기존 도구의 스타일과 일치하는지 확인해야 합니다. 
특히 이름에 따라 예상되는 모든 인수가 설명에 코드 스타일로 언급되어 있는지, 예상되는 유형과 그 유형이 무엇인지에 대한 설명이 포함되어 있는지 확인하세요.

<Tip>

도구에 어떤 이름과 설명이 있어야 하는지 이해하려면 엄선된 Transformers 도구의 이름과 설명을 확인하세요. 
[`Agent.toolbox`] 속성을 가진 모든 도구를 볼 수 있습니다.

</Tip>

세 번째 부분에는 에이전트가 어떤 종류의 사용자 요청에 대해 어떤 코드를 생성해야 하는지 정확하게 보여주는 엄선된 예제 세트가 포함되어 있습니다. 
에이전트를 지원하는 대규모 언어 모델은 프롬프트에서 패턴을 인식하고 새로운 데이터로 패턴을 반복하는 데 매우 능숙합니다. 
따라서 에이전트가 실제로 올바른 실행 가능한 코드를 생성할 가능성을 극대화하는 방식으로 예제를 작성하는 것이 매우 중요합니다. 

한 가지 예를 살펴보겠습니다:

````text
Task: "Identify the oldest person in the `document` and create an image showcasing the result as a banner."

I will use the following tools: `document_qa` to find the oldest person in the document, then `image_generator` to generate an image according to the answer.

Answer:
```py
answer = document_qa(document, question="What is the oldest person?")
print(f"The answer is {answer}.")
image = image_generator("A banner showing " + answer)
```

````
작업 설명, 에이전트가 수행하려는 작업에 대한 설명, 마지막으로 생성된 코드, 이 세 부분으로 구성된 프롬프트는 모델에 반복하여 제공됩니다. 
프롬프트의 일부인 모든 예제는 이러한 정확한 패턴으로 되어 있으므로, 에이전트가 새 토큰을 생성할 때 정확히 동일한 패턴을 재현할 수 있습니다.

프롬프트 예제는 Transformers 팀이 선별하고 일련의 [problem statements](https://github.com/huggingface/transformers/blob/main/src/transformers/tools/evaluate_agent.py)에 따라 엄격하게 평가하여 
에이전트의 프롬프트가 에이전트의 실제 사용 사례를 최대한 잘 해결할 수 있도록 보장합니다.

프롬프트의 마지막 부분은 다음에 해당합니다:
```text
Task: "Draw me a picture of rivers and lakes"

I will use the following
```

이는 에이전트가 완료해야 할 최종적인 미완성 예제입니다. 미완성 예제는 실제 사용자 입력에 따라 동적으로 만들어집니다. 
위 예시의 경우 사용자가 다음과 같이 실행했습니다:

```py
agent.run("Draw me a picture of rivers and lakes")
```

사용자 입력 - *즉* Task: *"Draw me a picture of rivers and lakes"*가 프롬프트 템플릿에 맞춰 "Task: <task> \n\n I will use the following"로 캐스팅됩니다. 
이 문장은 에이전트에게 조건이 적용되는 프롬프트의 마지막 줄을 구성하므로 에이전트가 이전 예제에서 수행한 것과 정확히 동일한 방식으로 예제를 완료하도록 강력하게 영향을 미칩니다.

너무 자세히 설명하지 않더라도 채팅 템플릿의 프롬프트 구조는 동일하지만 예제의 스타일이 약간 다릅니다. *예를 들면*:

````text
[...]

=====

Human: Answer the question in the variable `question` about the image stored in the variable `image`.

Assistant: I will use the tool `image_qa` to answer the question on the input image.

```py
answer = image_qa(text=question, image=image)
print(f"The answer is {answer}")
```

Human: I tried this code, it worked but didn't give me a good result. The question is in French

Assistant: In this case, the question needs to be translated first. I will use the tool `translator` to do this.

```py
translated_question = translator(question=question, src_lang="French", tgt_lang="English")
print(f"The translated question is {translated_question}.")
answer = image_qa(text=translated_question, image=image)
print(f"The answer is {answer}")
```

=====

[...]
````

`run` 프롬프트의 예와는 반대로, 각 `chat` 프롬프트의 예에는 *Human(사람)*과 *Assistant(어시스턴트)* 간에 하나 이상의 교환이 있습니다. 모든 교환은 `run` 프롬프트의 예와 유사한 구조로 되어 있습니다.
사용자의 입력이 *Human:* 뒤에 추가되며, 에이전트에게 코드를 생성하기 전에 수행해야 할 작업을 먼저 생성하라는 메시지가 표시됩니다. 
교환은 이전 교환을 기반으로 할 수 있으므로 위와 같이 사용자가 "**이** 코드를 시도했습니다"라고 입력하면 이전에 생성된 에이전트의 코드를 참조하여 과거 교환을 참조할 수 있습니다.

`.chat`을 실행하면 사용자의 입력 또는 *작업*이 미완성된 양식의 예시로 캐스팅됩니다:
```text
Human: <user-input>\n\nAssistant:
```
그러면 에이전트가 이를 완성합니다. `run` 명령과 달리 `chat` 명령은 완료된 예제를 프롬프트에 추가하여 에이전트에게 다음 `chat` 차례에 대한 더 많은 문맥을 제공합니다.

이제 프롬프트가 어떻게 구성되어 있는지 알았으니 어떻게 사용자 정의할 수 있는지 살펴봅시다!

### 좋은 사용자 입력 작성하기[[writing-good-user-inputs]]

대규모 언어 모델이 사용자의 의도를 이해하는 능력이 점점 더 향상되고 있지만, 에이전트가 올바른 작업을 선택할 수 있도록 최대한 정확성을 유지하는 것은 큰 도움이 됩니다. 
최대한 정확하다는 것은 무엇을 의미할까요?

에이전트는 프롬프트에서 도구 이름 목록과 해당 설명을 볼 수 있습니다. 
더 많은 도구가 추가될수록 에이전트가 올바른 도구를 선택하기가 더 어려워지고 실행할 도구의 올바른 순서를 선택하는 것은 더욱 어려워집니다. 
일반적인 실패 사례를 살펴보겠습니다. 여기서는 분석할 코드만 반환하겠습니다.

```py
from transformers import HfAgent

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")

agent.run("Show me a tree", return_code=True)
```

그러면 다음 결과가 출력됩니다:

```text
==Explanation from the agent==
I will use the following tool: `image_segmenter` to create a segmentation mask for the image.


==Code generated by the agent==
mask = image_segmenter(image, prompt="tree")
```

우리가 원했던 결과가 아닐 수도 있습니다. 대신 나무 이미지가 생성되기를 원할 가능성이 더 높습니다.
따라서 에이전트가 특정 도구를 사용하도록 유도하려면 도구의 이름과 설명에 있는 중요한 키워드를 사용하는 것이 매우 유용할 수 있습니다. 한번 살펴보겠습니다.
```py
agent.toolbox["image_generator"].description
```

```text
'This is a tool that creates an image according to a prompt, which is a text description. It takes an input named `prompt` which contains the image description and outputs an image.
```

이름과 설명은 "image", "prompt", "create" 및 "generate" 키워드를 사용합니다. 이 단어들을 사용하면 더 잘 작동할 가능성이 높습니다. 프롬프트를 조금 더 구체화해 보겠습니다.

```py
agent.run("Create an image of a tree", return_code=True)
```

이 코드는 다음 프롬프트를 만들어냅니다:
```text
==Explanation from the agent==
I will use the following tool `image_generator` to generate an image of a tree.


==Code generated by the agent==
image = image_generator(prompt="tree")
```

훨씬 낫네요! 저희가 원했던 것과 비슷해 보입니다. 
즉, 에이전트가 작업을 올바른 도구에 올바르게 매핑하는 데 어려움을 겪고 있다면 도구 이름과 설명에서 가장 관련성이 높은 키워드를 찾아보고 이를 통해 작업 요청을 구체화해 보세요.

### 도구 설명 사용자 정의하기[[customizing-the-tool-descriptions]]

앞서 살펴본 것처럼 에이전트는 각 도구의 이름과 설명에 액세스할 수 있습니다. 
기본 도구에는 매우 정확한 이름과 설명이 있어야 하지만 특정 사용 사례에 맞게 도구의 설명이나 이름을 변경하는 것이 도움이 될 수도 있습니다. 
이는 매우 유사한 여러 도구를 추가했거나 특정 도메인(*예*: 이미지 생성 및 변환)에만 에이전트를 사용하려는 경우에 특히 중요해질 수 있습니다.

일반적인 문제는 이미지 생성 작업에 많이 사용되는 경우 에이전트가 이미지 생성과 이미지 변환/수정을 혼동하는 것입니다. *예를 들어,*
```py
agent.run("Make an image of a house and a car", return_code=True)
```
그러면 다음 결과가 출력됩니다:
```text
==Explanation from the agent== 
I will use the following tools `image_generator` to generate an image of a house and `image_transformer` to transform the image of a car into the image of a house.

==Code generated by the agent==
house_image = image_generator(prompt="A house")
car_image = image_generator(prompt="A car")
house_car_image = image_transformer(image=car_image, prompt="A house")
```

결과물이 우리가 여기서 원하는 것과 정확히 일치하지 않을 수 있습니다. 에이전트가 `image_generator`와 `image_transformer`의 차이점을 이해하기 어려워서 두 가지를 함께 사용하는 경우가 많은 것 같습니다.

여기서 `image_transformer`의 도구 이름과 설명을 변경하여 에이전트가 도울 수 있습니다. 
"image" 및 "prompt"와 약간 분리하기 위해 `modifier`라고 대신 부르겠습니다:
```py
agent.toolbox["modifier"] = agent.toolbox.pop("image_transformer")
agent.toolbox["modifier"].description = agent.toolbox["modifier"].description.replace(
    "transforms an image according to a prompt", "modifies an image"
)
```

이제 "modify"은 새 이미지 프로세서를 사용하라는 강력한 신호이므로 위의 프롬프트에 도움이 될 것입니다. 다시 실행해 봅시다.

```py
agent.run("Make an image of a house and a car", return_code=True)
```

여기서 다음과 같은 결과를 얻게 됩니다:
```text
==Explanation from the agent==
I will use the following tools: `image_generator` to generate an image of a house, then `image_generator` to generate an image of a car.


==Code generated by the agent==
house_image = image_generator(prompt="A house")
car_image = image_generator(prompt="A car")
```

우리가 염두에 두었던 것과 확실히 더 가까워졌습니다! 하지만 집과 자동차가 모두 같은 이미지에 포함되면 좋겠습니다. 작업을 단일 이미지 생성에 더 집중하면 도움이 될 것입니다:

```py
agent.run("Create image: 'A house and car'", return_code=True)
```

```text
==Explanation from the agent==
I will use the following tool: `image_generator` to generate an image.


==Code generated by the agent==
image = image_generator(prompt="A house and car")
```

<Tip warning={true}>

에이전트는 여전히 특히 여러 개체의 이미지를 생성하는 것과 같이 약간 더 복잡한 사용 사례에서 취약한 경우가 많습니다.
앞으로 몇 달 안에 에이전트 자체와 기본 프롬프트가 더욱 개선되어 에이전트가 다양한 사용자 입력에 더욱 강력하게 대응할 수 있도록 할 예정입니다.

</Tip>

### 전체 프롬프트 사용자 정의하기[[customizing-the-whole-prompt]]

사용자에게 최대한의 유연성을 제공하기 위해 [위](#structure-of-the-prompt)에 설명된 전체 프롬프트 템플릿을 사용자가 덮어쓸 수 있습니다. 
이 경우 사용자 정의 프롬프트에 소개 섹션, 도구 섹션, 예제 섹션 및 미완성 예제 섹션이 포함되어 있는지 확인하세요. 
`run` 프롬프트 템플릿을 덮어쓰려면 다음과 같이 하면 됩니다:

```py
template = """ [...] """

agent = HfAgent(your_endpoint, run_prompt_template=template)
```

<Tip warning={true}>

에이전트가 사용 가능한 도구를 인식하고 사용자의 프롬프트를 올바르게 삽입할 수 있도록 `<<all_tools>>` 문자열과 `<<prompt>>`를 `template` 어딘가에 정의해야 합니다.

</Tip>

마찬가지로 `chat` 프롬프트 템플릿을 덮어쓸 수 있습니다. `chat` 모드에서는 항상 다음과 같은 교환 형식을 사용한다는 점에 유의하세요:

```text
Human: <<task>>

Assistant:
```

따라서 사용자 정의 `chat` 프롬프트 템플릿의 예제에서도 이 형식을 사용하는 것이 중요합니다. 
다음과 같이 인스턴스화 할 때 `chat` 템플릿을 덮어쓸 수 있습니다.

```
template = """ [...] """

agent = HfAgent(url_endpoint=your_endpoint, chat_prompt_template=template)
```

<Tip warning={true}>

에이전트가 사용 가능한 도구를 인식할 수 있도록 `<<all_tools>>` 문자열을 `template` 어딘가에 정의해야 합니다.

</Tip>

두 경우 모두 커뮤니티의 누군가가 호스팅하는 템플릿을 사용하려는 경우 프롬프트 템플릿 대신 저장소 ID를 전달할 수 있습니다. 
기본 프롬프트는 [이 저장소](https://huggingface.co/datasets/huggingface-tools/default-prompts)를 예로 들 수 있습니다.

Hub의 저장소에 사용자 정의 프롬프트를 업로드하여 커뮤니티와 공유하려면 다음을 확인하세요:
- 데이터 세트 저장소를 사용하세요.
- `run` 명령에 대한 프롬프트 템플릿을 `run_prompt_template.txt`라는 파일에 넣으세요.
- `chat` 명령에 대한 프롬프트 템플릿을 `chat_prompt_template.txt`라는 파일에 넣으세요.

## 사용자 정의 도구 사용하기[[using-custom-tools]]

이 섹션에서는 이미지 생성에 특화된 두 가지 기존 사용자 정의 도구를 활용하겠습니다:

- 더 많은 이미지 수정을 허용하기 위해 [huggingface-tools/image-transformation](https://huggingface.co/spaces/huggingface-tools/image-transformation)을 
  [diffusers/controlnet-canny-tool](https://huggingface.co/spaces/diffusers/controlnet-canny-tool)로 대체합니다.
- 기본 도구 상자에 이미지 업스케일링을 위한 새로운 도구가 추가되었습니다: 
  [diffusers/latent-upscaler-tool](https://huggingface.co/spaces/diffusers/latent-upscaler-tool)가 기존 이미지 변환 도구를 대체합니다.

편리한 [`load_tool`] 함수를 사용하여 사용자 정의 도구를 가져오는 것으로 시작하겠습니다:

```py
from transformers import load_tool

controlnet_transformer = load_tool("diffusers/controlnet-canny-tool")
upscaler = load_tool("diffusers/latent-upscaler-tool")
```

에이전트에게 사용자 정의 도구를 추가하면 도구의 설명과 이름이 에이전트의 프롬프트에 자동으로 포함됩니다. 
따라서 에이전트가 사용 방법을 이해할 수 있도록 사용자 정의 도구의 설명과 이름을 잘 작성해야 합니다.
`controlnet_transformer`의 설명과 이름을 살펴보겠습니다:

```py
print(f"Description: '{controlnet_transformer.description}'")
print(f"Name: '{controlnet_transformer.name}'")
```

그러면 다음 결과가 출력됩니다:
```text
Description: 'This is a tool that transforms an image with ControlNet according to a prompt. 
It takes two inputs: `image`, which should be the image to transform, and `prompt`, which should be the prompt to use to change it. It returns the modified image.'
Name: 'image_transformer'
```

이름과 설명이 정확하고 [큐레이팅 된 도구 세트(curated set of tools)](./transformers_agents#a-curated-set-of-tools)의 스타일에 맞습니다.
다음으로, `controlnet_transformer`와 `upscaler`로 에이전트를 인스턴스화해 봅시다:
```py
tools = [controlnet_transformer, upscaler]
agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder", additional_tools=tools)
```

이 명령을 실행하면 다음 정보가 표시됩니다:

```text
image_transformer has been replaced by <transformers_modules.diffusers.controlnet-canny-tool.bd76182c7777eba9612fc03c0
8718a60c0aa6312.image_transformation.ControlNetTransformationTool object at 0x7f1d3bfa3a00> as provided in `additional_tools`
```

큐레이팅된 도구 세트에는 이미 'image_transformer' 도구가 있으며, 이 도구는 사용자 정의 도구로 대체됩니다.

<Tip>

기존 도구와 똑같은 작업에 사용자 정의 도구를 사용하려는 경우 기존 도구를 덮어쓰는 것이 유용할 수 있습니다. 
에이전트가 해당 작업에 능숙하기 때문입니다.
이 경우 사용자 정의 도구가 덮어쓴 도구와 정확히 동일한 API를 따라야 하며, 그렇지 않으면 해당 도구를 사용하는 모든 예제가 업데이트되도록 프롬프트 템플릿을 조정해야 한다는 점에 유의하세요.

</Tip>

업스케일러 도구에 지정된 'image_upscaler'라는 이름 아직 기본 도구 상자에는 존재하지 않기 때문에, 도구 목록에 해당 이름이 간단히 추가되었습니다.
에이전트가 현재 사용할 수 있는 도구 상자는 언제든지 `agent.toolbox` 속성을 통해 확인할 수 있습니다:

```py
print("\n".join([f"- {a}" for a in agent.toolbox.keys()]))
```

```text
- document_qa
- image_captioner
- image_qa
- image_segmenter
- transcriber
- summarizer
- text_classifier
- text_qa
- text_reader
- translator
- image_transformer
- text_downloader
- image_generator
- video_generator
- image_upscaler
```

에이전트의 도구 상자에 `image_upscaler`가 추가된 점을 주목하세요.

이제 새로운 도구를 사용해봅시다! [Transformers Agents Quickstart](./transformers_agents#single-execution-run)에서 생성한 이미지를 다시 사용하겠습니다.

```py
from diffusers.utils import load_image

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png"
)
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png" width=200> 

이미지를 아름다운 겨울 풍경으로 바꿔 봅시다:

```py
image = agent.run("Transform the image: 'A frozen lake and snowy forest'", image=image)
```

```text
==Explanation from the agent==
I will use the following tool: `image_transformer` to transform the image.


==Code generated by the agent==
image = image_transformer(image, prompt="A frozen lake and snowy forest")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes_winter.png" width=200> 

새로운 이미지 처리 도구는 이미지를 매우 강력하게 수정할 수 있는 ControlNet을 기반으로 합니다.
기본적으로 이미지 처리 도구는 512x512 픽셀 크기의 이미지를 반환합니다. 이를 업스케일링할 수 있는지 살펴봅시다.

```py
image = agent.run("Upscale the image", image)
```

```text
==Explanation from the agent==
I will use the following tool: `image_upscaler` to upscale the image.


==Code generated by the agent==
upscaled_image = image_upscaler(image)
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes_winter_upscale.png" width=400> 

에이전트는 업스케일러 도구의 설명과 이름만 보고 방금 추가한 업스케일러 도구에 "이미지 업스케일링"이라는 프롬프트를 자동으로 매핑하여 올바르게 실행했습니다.

다음으로 새 사용자 정의 도구를 만드는 방법을 살펴보겠습니다.

### 새 도구 추가하기[[adding-new-tools]]

이 섹션에서는 에이전트에게 추가할 수 있는 새 도구를 만드는 방법을 보여 드립니다.

#### 새 도구 만들기[[creating-a-new-tool]]

먼저 도구를 만드는 것부터 시작하겠습니다. 
특정 작업에 대해 가장 많은 다운로드를 받은 Hugging Face Hub의 모델을 가져오는, 그다지 유용하지는 않지만 재미있는 작업을 추가하겠습니다.

다음 코드를 사용하면 됩니다:

```python
from huggingface_hub import list_models

task = "text-classification"

model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
print(model.id)
```
`text-classification`(텍스트 분류) 작업의 경우 `'facebook/bart-large-mnli'`를 반환하고, `translation`(번역) 작업의 경우 `'t5-base'`를 반환합니다.

이를 에이전트가 활용할 수 있는 도구로 변환하려면 어떻게 해야 할까요? 
모든 도구는 필요한 주요 속성을 보유하는 슈퍼클래스 `Tool`에 의존합니다. 이를 상속하는 클래스를 만들어 보겠습니다:

```python
from transformers import Tool


class HFModelDownloadsTool(Tool):
    pass
```

이 클래스에는 몇 가지 요구사항이 있습니다:
- 도구 자체의 이름에 해당하는 `name` 속성. 수행명이 있는 다른 도구와 호환되도록 `model_download_counter`로 이름을 지정하겠습니다.
- 에이전트의 프롬프트를 채우는 데 사용되는 속성 `description`.
- `inputs` 및 `outputs` 속성. 이를 정의하면 Python 인터프리터가 유형에 대한 정보에 입각한 선택을 하는 데 도움이 되며, 
  도구를 허브에 푸시할 때 gradio 데모를 생성할 수 있습니다. 
  두 속성 모두 값은 '텍스트', '이미지' 또는 '오디오'가 될 수 있는 예상 값의 리스트입니다.
- 추론 코드가 포함된 `__call__` 메소드. 이것이 우리가 위에서 다루었던 코드입니다!

이제 클래스의 모습은 다음과 같습니다:

```python
from transformers import Tool
from huggingface_hub import list_models


class HFModelDownloadsTool(Tool):
    name = "model_download_counter"
    description = (
        "This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub. "
        "It takes the name of the category (such as text-classification, depth-estimation, etc), and "
        "returns the name of the checkpoint."
    )

    inputs = ["text"]
    outputs = ["text"]

    def __call__(self, task: str):
        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id
```

이제 도구를 손쉽게 사용할 수 있게 되었습니다. 
도구를 파일에 저장하고 메인 스크립트에서 가져옵니다. 이 파일의 이름을 `model_downloads.py`로 지정하면 결과적으로 가져오기 코드는 다음과 같습니다:

```python
from model_downloads import HFModelDownloadsTool

tool = HFModelDownloadsTool()
```

다른 사람들이 이 기능을 활용할 수 있도록 하고 초기화를 더 간단하게 하려면 네임스페이스 아래의 Hub로 푸시하는 것이 좋습니다. 
그렇게 하려면 `tool` 변수에서 `push_to_hub`를 호출하면 됩니다:

```python
tool.push_to_hub("hf-model-downloads")
```

이제 허브에 코드가 생겼습니다! 마지막 단계인 에이전트가 코드를 사용하도록 하는 단계를 살펴보겠습니다.

#### 에이전트가 도구를 사용하게 하기[[Having-the-agent-use-the-tool]]

이제 이런 식으로 허브에 존재하는 도구를 인스턴스화할 수 있습니다(도구의 사용자 이름은 변경하세요):
We now have our tool that lives on the Hub which can be instantiated as such (change the user name for your tool):

```python
from transformers import load_tool

tool = load_tool("lysandre/hf-model-downloads")
```

이 도구를 에이전트에서 사용하려면 에이전트 초기화 메소드의 `additional_tools` 매개변수에 전달하기만 하면 됩니다:

```python
from transformers import HfAgent

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder", additional_tools=[tool])

agent.run(
    "Can you read out loud the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
)
```
그러면 다음과 같은 결과가 출력됩니다:
```text
==Code generated by the agent==
model = model_download_counter(task="text-to-video")
print(f"The model with the most downloads is {model}.")
audio_model = text_reader(model)


==Result==
The model with the most downloads is damo-vilab/text-to-video-ms-1.7b.
```

and generates the following audio.

| **Audio**                                                                                                                                            |
|------------------------------------------------------------------------------------------------------------------------------------------------------|
| <audio controls><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/damo.wav" type="audio/wav"/> |


<Tip>

LLM에 따라 일부는 매우 취약하기 때문에 제대로 작동하려면 매우 정확한 프롬프트가 필요합니다. 
에이전트가 도구를 잘 활용하기 위해서는 도구의 이름과 설명을 잘 정의하는 것이 무엇보다 중요합니다.

</Tip>

### 기존 도구 대체하기[[replacing-existing-tools]]

에이전트의 도구 상자에 새 항목을 배정하기만 하면 기존 도구를 대체할 수 있습니다. 방법은 다음과 같습니다:

```python
from transformers import HfAgent, load_tool

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
agent.toolbox["image-transformation"] = load_tool("diffusers/controlnet-canny-tool")
```

<Tip>

다른 도구로 교체할 때는 주의하세요! 이 작업으로 에이전트의 프롬프트도 조정됩니다. 
작업에 더 적합한 프롬프트가 있으면 좋을 수 있지만, 
다른 도구보다 더 많이 선택되거나 정의한 도구 대신 다른 도구가 선택될 수도 있습니다.

</Tip>

## gradio-tools 사용하기[[leveraging-gradio-tools]]

[gradio-tools](https://github.com/freddyaboulton/gradio-tools)는 Hugging Face Spaces를 도구로 사용할 수 있는 강력한 라이브러리입니다. 
기존의 많은 Spaces뿐만 아니라 사용자 정의 Spaces를 사용하여 디자인할 수 있도록 지원합니다.

우리는 `Tool.from_gradio` 메소드를 사용하여 `gradio_tools`에 대한 지원을 제공합니다. 
예를 들어, 프롬프트를 개선하고 더 나은 이미지를 생성하기 위해 `gradio-tools` 툴킷에서 제공되는 `StableDiffusionPromptGeneratorTool` 도구를 활용하고자 합니다.

먼저 `gradio_tools`에서 도구를 가져와서 인스턴스화합니다:

```python
from gradio_tools import StableDiffusionPromptGeneratorTool

gradio_tool = StableDiffusionPromptGeneratorTool()
```

해당 인스턴스를 `Tool.from_gradio` 메소드에 전달합니다:

```python
from transformers import Tool

tool = Tool.from_gradio(gradio_tool)
```

이제 일반적인 사용자 정의 도구와 똑같이 관리할 수 있습니다. 
이를 활용하여 `a rabbit wearing a space suit'(우주복을 입은 토끼)라는 프롬프트를 개선했습니다:

```python
from transformers import HfAgent

agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder", additional_tools=[tool])

agent.run("Generate an image of the `prompt` after improving it.", prompt="A rabbit wearing a space suit")
```

모델이 도구를 적절히 활용합니다:
```text
==Explanation from the agent==
I will use the following  tools: `StableDiffusionPromptGenerator` to improve the prompt, then `image_generator` to generate an image according to the improved prompt.


==Code generated by the agent==
improved_prompt = StableDiffusionPromptGenerator(prompt)
print(f"The improved prompt is {improved_prompt}.")
image = image_generator(improved_prompt)
```

마지막으로 이미지를 생성하기 전에:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png">

<Tip warning={true}>

gradio-tools는 다른 모달리티로 작업할 때에도 *텍스트* 입력 및 출력을 필요로 합니다. 
이 구현은 이미지 및 오디오 객체에서 작동합니다. 
현재는 이 두 가지가 호환되지 않지만 지원 개선을 위해 노력하면서 빠르게 호환될 것입니다.

</Tip>

## 향후 Langchain과의 호환성[[future-compatibility-with-langchain]]

저희는 Langchain을 좋아하며 매우 매력적인 도구 모음을 가지고 있다고 생각합니다. 
이러한 도구를 처리하기 위해 Langchain은 다른 모달리티와 작업할 때에도 *텍스트* 입력과 출력을 필요로 합니다.
이는 종종 객체의 직렬화된(즉, 디스크에 저장된) 버전입니다.

이 차이로 인해 transformers-agents와 Langchain 간에는 멀티 모달리티가 처리되지 않습니다. 
향후 버전에서 이 제한이 해결되기를 바라며, 이 호환성을 달성할 수 있도록 열렬한 Langchain 사용자의 도움을 환영합니다.

저희는 더 나은 지원을 제공하고자 합니다. 도움을 주고 싶으시다면, [이슈를 열어](https://github.com/huggingface/transformers/issues/new) 의견을 공유해 주세요.
