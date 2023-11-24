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

# Transformers Agent [[transformers-agent]]

<Tip warning={true}>

Transformers Agent는 실험 중인 API로 언제든지 변경될 수 있습니다. 
API 또는 기반 모델이 변경되기 쉽기 때문에 에이전트가 반환하는 결과도 달라질 수 있습니다.

</Tip>

Transformers 버전 4.29.0.에서 *도구*와 *에이전트*라는 컨셉을 도입했습니다. [이 colab](https://colab.research.google.com/drive/1c7MHD-T1forUPGcC_jlwsIptOzpG3hSj)에서 사용해볼 수 있습니다.

간단히 말하면, Agent는 트랜스포머 위에 자연어 API를 제공합니다. 
엄선된 도구 세트를 정의하고, 자연어를 해석하여 이러한 도구를 사용할 수 있는 에이전트를 설계했습니다. 
이 API는 확장이 가능하도록 설계 되었습니다. 
주요 도구를 선별해두었지만, 커뮤니티에서 개발한 모든 도구를 사용할 수 있도록 시스템을 쉽게 확장할 수 있는 방법도 보여드리겠습니다.

몇 가지 예를 통해 새로운 API로 무엇을 할 수 있는지 살펴보겠습니다. 
이 API는 특히 멀티모달 작업에서 강력하므로 이미지를 생성하고 텍스트를 소리내어 읽어보겠습니다.

```py
agent.run("Caption the following image", image=image)
```

| **Input**                                                                                                                   | **Output**                        |
|-----------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/beaver.png" width=200> | A beaver is swimming in the water |

---

```py
agent.run("Read the following text out loud", text=text)
```
| **Input**                                                                                                               | **Output**                                   |
|-------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| A beaver is swimming in the water | <audio controls><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tts_example.wav" type="audio/wav"> your browser does not support the audio element. </audio>

---

```py
agent.run(
    "In the following `document`, where will the TRRF Scientific Advisory Council Meeting take place?",
    document=document,
)
```
| **Input**                                                                                                                   | **Output**     |
|-----------------------------------------------------------------------------------------------------------------------------|----------------|
| <img src="https://datasets-server.huggingface.co/assets/hf-internal-testing/example-documents/--/hf-internal-testing--example-documents/test/0/image/image.jpg" width=200> | ballroom foyer |

## 바로 시작하기 [[quickstart]]

`agent.run`을 사용하려면 먼저 대규모 언어 모델(LLM)인 에이전트를 인스턴스화해야 합니다. 
저희는 openAI 모델뿐만 아니라 BigCode 및 OpenAssistant의 오픈소스 대체 모델도 지원합니다. 
openAI 모델의 성능이 더 우수하지만(단, openAI API 키가 필요하므로 무료로 사용할 수 없음), 
Hugging Face는 BigCode와 OpenAssistant 모델의 엔드포인트에 대한 무료 액세스를 제공하고 있습니다.

우선 모든 기본 종속성을 설치하려면 `agents`를 추가로 설치하세요.
```bash
pip install transformers[agents]
```

openAI 모델을 사용하려면 `openai` 종속성을 설치한 후 [`OpenAiAgent`]를 인스턴스화합니다:

```bash
pip install openai
```


```py
from transformers import OpenAiAgent

agent = OpenAiAgent(model="text-davinci-003", api_key="<your_api_key>")
```

BigCode 또는 OpenAssistant를 사용하려면 먼저 로그인하여 Inference API에 액세스하세요:

```py
from huggingface_hub import login

login("<YOUR_TOKEN>")
```

그런 다음 에이전트를 인스턴스화합니다.

```py
from transformers import HfAgent

# Starcoder
agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
# StarcoderBase
# agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoderbase")
# OpenAssistant
# agent = HfAgent(url_endpoint="https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")
```

현재 Hugging Face에서 무료로 제공하는 추론 API를 사용하고 있습니다. 
이 모델에 대한 자체 추론 엔드포인트가 있는 경우(또는 다른 엔드포인트가 있는 경우) 위의 URL을 해당 URL 엔드포인트로 바꿀 수 있습니다.

<Tip>

StarCoder와 OpenAssistant는 무료로 사용할 수 있으며 간단한 작업에서 놀라울 정도로 잘 작동합니다. 
그러나 더 복잡한 프롬프트를 처리할 때는 체크포인트가 잘 작동하지 않습니다. 
이러한 문제가 발생하면 OpenAI 모델을 사용해 보시기 바랍니다. 아쉽게도 오픈소스는 아니지만 현재로서는 더 나은 성능을 제공합니다.

</Tip>

이제 준비가 완료되었습니다! 이제 자유롭게 사용할 수 있는 두 가지 API에 대해 자세히 알아보겠습니다.

### 단일 실행 (run) [[single-execution-(run)]] 

단일 실행 방법은 에이전트의 [`~Agent.run`] 메소드를 사용하는 경우입니다:

```py
agent.run("Draw me a picture of rivers and lakes.")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png" width=200>

수행하려는 작업에 적합한 도구를 자동으로 선택하여 적절하게 실행합니다. 
동일한 명령어에서 하나 또는 여러 개의 작업을 수행할 수 있습니다
(다만, 명령어가 복잡할수록 에이전트가 실패할 가능성이 높아집니다).

```py
agent.run("Draw me a picture of the sea then transform the picture to add an island")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/sea_and_island.png" width=200>

<br/>


모든 [`~Agent.run`] 작업은 독립적이므로 다른 작업으로 여러 번 연속해서 실행할 수 있습니다.

`agent`는 큰 언어 모델일 뿐이므로 프롬프트에 약간의 변화를 주면 완전히 다른 결과가 나올 수 있다는 점에 유의하세요. 
수행하려는 작업을 최대한 명확하게 설명하는 것이 중요합니다. 
좋은 프롬프트를 작성하는 방법은 [여기](custom_tools#writing-good-user-inputs)에서 자세히 확인할 수 있습니다.

여러 실행에 걸쳐 상태를 유지하거나 텍스트가 아닌 개체를 에이전트에게 전달하려는 경우에는 에이전트가 사용할 변수를 지정할 수 있습니다. 
예를 들어 강과 호수의 첫 번째 이미지를 생성한 뒤, 
모델이 해당 그림에 섬을 추가하도록 다음과 같이 요청할 수 있습니다:

```python
picture = agent.run("Generate a picture of rivers and lakes.")
updated_picture = agent.run("Transform the image in `picture` to add an island to it.", picture=picture)
```

<Tip>

이 방법은 모델이 요청을 이해하지 못하고 도구를 혼합할 때 유용할 수 있습니다. 예를 들면 다음과 같습니다:

```py
agent.run("Draw me the picture of a capybara swimming in the sea")
```

여기서 모델은 두 가지 방식으로 해석할 수 있습니다:
- `text-to-image`이 바다에서 헤엄치는 카피바라를 생성하도록 합니다.
- 또는 `text-to-image`이 카피바라를 생성한 다음 `image-transformation` 도구를 사용하여 바다에서 헤엄치도록 합니다.

첫 번째 시나리오를 강제로 실행하려면 프롬프트를 인수로 전달하여 실행할 수 있습니다:

```py
agent.run("Draw me a picture of the `prompt`", prompt="a capybara swimming in the sea")
```

</Tip>


### 대화 기반 실행 (chat) [[chat-based-execution-(chat)]]

에이전트는 [`~Agent.chat`] 메소드를 사용하는 대화 기반 접근 방식도 있습니다:

```py
agent.chat("Generate a picture of rivers and lakes")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png" width=200> 

```py
agent.chat("Transform the picture so that there is a rock in there")
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes_and_beaver.png" width=200>

<br/>

이 방식은 여러 명령어에 걸쳐 상태를 유지하고자 할 때 흥미로운 접근 방식입니다. 
실험용으로 더 좋지만 복잡한 명령어보다는 
단일 명령어([`~Agent.run`] 메소드가 더 잘 처리하는 명령어)에 훨씬 더 잘 작동하는 경향이 있습니다.

이 메소드는 텍스트가 아닌 유형이나 특정 프롬프트를 전달하려는 경우 인수를 받을 수도 있습니다.

### ⚠️ 원격 실행 [[remote-execution]]

데모 목적과 모든 설정에서 사용할 수 있도록 
에이전트가 접근할 수 있는 몇 가지 기본 도구에 대한 원격 실행기를 만들었습니다. 
이러한 도구는 [inference endpoints](https://huggingface.co/inference-endpoints)를 사용하여 만들어졌습니다. 
원격 실행기 도구를 직접 설정하는 방법을 보려면 [사용자 정의 도구 가이드](./custom_tools)를 읽어보시기 바랍니다.

원격 도구로 실행하려면 [`~Agent.run`] 또는 [`~Agent.chat`] 중 하나에 `remote=True`를 지정하기만 하면 됩니다.

예를 들어 다음 명령은 많은 RAM이나 GPU 없이도 모든 장치에서 효율적으로 실행할 수 있습니다:

```py
agent.run("Draw me a picture of rivers and lakes", remote=True)
```

[`~Agent.chat`]도 마찬가지입니다:

```py
agent.chat("Draw me a picture of rivers and lakes", remote=True)
```

### 여기서 무슨 일이 일어나는 거죠? 도구란 무엇이고, 에이전트란 무엇인가요? [[whats-happening-here-what-are-tools-and-what-are-agents]]

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/diagram.png">

#### 에이전트 [[agents]]

여기서 "에이전트"는 대규모 언어 모델이며, 특정 도구 모음에 접근할 수 있도록 프롬프트하고 있습니다.

LLM은 작은 코드 샘플을 생성하는 데 상당히 능숙하므로,
이 장점을 활용해 도구 모음을 사용하여 작업을 수행하는 작은 코드 샘플을 제공하라는 메시지를 표시합니다. 
그런 다음 에이전트에게 제공하는 작업과 제공하는 도구에 대한 설명으로 이 프롬프트가 완료됩니다. 
이렇게 하면 사용 중인 도구들의 문서에 접근할 수 있으며, 해당 도구들의 입력과 출력을 예상하고, 관련된 코드를 생성할 수 있습니다.

#### 도구 [[tools]]

도구는 매우 간단합니다. 이름과 설명이 있는 단일 기능으로 구성되어 있습니다. 
그런 다음 이러한 도구의 설명을 사용하여 상담원에게 프롬프트를 표시합니다. 
이 프롬프트를 통해 상담원에게 쿼리에서 요청된 작업을 수행하기 위해 도구를 활용하는 방법을 보여줍니다.

에이전트가 매우 원자적인 도구를 사용하여 더 나은 코드를 작성하기 때문에 파이프라인이 아닌 완전히 새로운 도구를 사용합니다. 
파이프라인은 더 많이 리팩터링되며 종종 여러 작업을 하나로 결합합니다. 
도구는 하나의 매우 간단한 작업에만 집중하도록 되어 있습니다.

#### 코드 실행?! [[code-execution]]

그런 다음 이 코드는 도구와 함께 전달된 입력 세트에 대해 작은 Python 인터프리터를 사용하여 실행됩니다. 
"임의 코드 실행이라니!"이라고 비명을 지르는 소리가 들리겠지만, 그렇지 않은 이유를 설명하겠습니다.

호출할 수 있는 함수는 제공한 도구와 인쇄 기능뿐이므로 이미 실행할 수 있는 기능이 제한되어 있습니다. 
Hugging Face 도구로 제한되어 있다면 안전할 것입니다. 

그리고 어트리뷰트 조회나 가져오기를 허용하지 않으므로
(어차피 작은 함수 집합에 입/출력을 전달할 때는 필요하지 않아야 합니다) 
가장 명백한 공격(어차피 LLM에 출력하라는 메시지를 표시해야 합니다)은 문제가 되지 않습니다. 
매우 안전하게 하고 싶다면 추가 인수 return_code=True를 사용하여 run() 메소드를 실행하면 됩니다.
이 경우 에이전트가 실행할 코드를 반환하고 실행할지 여부를 결정할 수 있습니다.

불법적인 연산을 수행하려고 하거나 에이전트가 생성한 코드에 일반적인 파이썬 오류가 있는 경우 
실행이 중지됩니다.

### 엄선된 도구 모음 [[a-curated-set-of-tools]]

저희는 이러한 에이전트들의 역량을 강화할 수 있는 일련의 도구를 확인하고 있습니다. 
다음은 연동된 도구의 최신 목록입니다:

- **문서 질문 답변**: 이미지 형식의 문서(예: PDF)가 주어지면 이 문서에 대한 질문에 답변합니다. ([Donut](./model_doc/donut))
- **텍스트 질문 답변**: 긴 텍스트와 질문이 주어지면 텍스트에서 질문에 답변합니다. ([Flan-T5](./model_doc/flan-t5))
- **무조건 이미지 캡셔닝**: 이미지에 캡션을 답니다! ([BLIP](./model_doc/blip))
- **이미지 질문 답변**: 이미지가 주어지면 이 이미지에 대한 질문에 답변하기. ([VILT](./model_doc/vilt))
- **이미지 분할**: 이미지와 프롬프트가 주어지면 해당 프롬프트의 분할 마스크를 출력합니다. ([CLIPSeg](./model_doc/clipseg))
- **음성을 텍스트로 변환**: 사람이 말하는 오디오 녹음이 주어지면 음성을 텍스트로 변환합니다. ([Whisper](./model_doc/whisper))
- **텍스트 음성 변환**: 텍스트를 음성으로 변환합니다. ([SpeechT5](./model_doc/speecht5))
- **제로 샷(zero-shot) 텍스트 분류**: 텍스트와 레이블 목록이 주어지면 텍스트와 가장 관련 있는 레이블을 식별합니다. ([BART](./model_doc/bart))
- **텍스트 요약**: 긴 텍스트를 한 문장 또는 몇 문장으로 요약합니다. ([BART](./model_doc/bart))
- **번역**: 텍스트를 지정된 언어로 번역합니다. ([NLLB](./model_doc/nllb))

이러한 도구는 트랜스포머에 통합되어 있으며, 예를 들어 수동으로도 사용할 수 있습니다:

```py
from transformers import load_tool

tool = load_tool("text-to-speech")
audio = tool("This is a text to speech tool")
```

### 사용자 정의 도구 [[custom-tools]]

엄선된 도구 세트도 있지만, 이 구현이 제공하는 가장 큰 가치는 사용자 지정 도구를 빠르게 만들고 공유할 수 있다는 점입니다.

도구의 코드를 Hugging Face Space나 모델 저장소에 푸시하면 에이전트에게 직접 도구를 활용할 수 있습니다.  [`huggingface-tools` organization](https://huggingface.co/huggingface-tools)에 몇 가지 **트랜스포머에 구애받지 않는** 툴을 추가했습니다:

- **텍스트 다운로더**: 웹 URL에서 텍스트를 다운로드합니다.
- **텍스트 이미지 변환**: 프롬프트에 따라 이미지를 생성하여 안정적인 확산을 활용합니다.
- **이미지 변환**: 초기 이미지와 프롬프트가 주어진 이미지를 수정하고, 안정적인 확산을 활용하는 지시 픽셀 2 픽셀을 활용합니다.
- **텍스트 비디오 변환**: 프롬프트에 따라 작은 비디오를 생성하며, damo-vilab을 활용합니다.

저희가 처음부터 사용하고 있는 텍스트-이미지 변환 도구는 [*huggingface-tools/text-to-image*](https://huggingface.co/spaces/huggingface-tools/text-to-image)에 있는 원격 도구입니다! 저희는 이 도구와 다른 조직에 이러한 도구를 계속 출시하여 이 구현을 더욱 강화할 것입니다.

에이전트는 기본적으로 [`huggingface-tools`](https://huggingface.co/huggingface-tools)에 있는 도구에 접근할 수 있습니다.
[다음 가이드](custom_tools)에서 도구를 작성하고 공유하는 방법과 Hub에 있는 사용자 지정 도구를 활용하는 방법에 대해 설명합니다.

### 코드 생성[[code-generation]]

지금까지 에이전트를 사용하여 작업을 수행하는 방법을 보여드렸습니다. 하지만 에이전트는 매우 제한된 Python 인터프리터를 사용하여 실행할 코드만 생성하고 있습니다. 다른 설정에서 생성된 코드를 사용하려는 경우 에이전트에게 도구 정의 및 정확한 가져오기와 함께 코드를 반환하라는 메시지를 표시할 수 있습니다.

예를 들어 다음 명령어는 
```python
agent.run("Draw me a picture of rivers and lakes", return_code=True)
```

다음 코드를 반환합니다.

```python
from transformers import load_tool

image_generator = load_tool("huggingface-tools/text-to-image")

image = image_generator(prompt="rivers and lakes")
```

이 코드는 직접 수정하고 실행할 수 있습니다.