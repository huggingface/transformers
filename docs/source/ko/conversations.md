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

# Transformers로 채팅하기[[chatting-with-transformers]]

이 글을 보고 있다면 **채팅 모델**에 대해 어느 정도 알고 계실 것입니다.
채팅 모델이란 메세지를 주고받을 수 있는 대화형 인공지능입니다. 
대표적으로 ChatGPT가 있고, 이와 비슷하거나 더 뛰어난 오픈소스 채팅 모델이 많이 존재합니다.  
이러한 모델들은 무료 다운로드할 수 있으며, 로컬에서 실행할 수 있습니다. 
크고 무거운 모델은 고성능 하드웨어와 메모리가 필요하지만, 
저사양 GPU 혹은 일반 데스크탑이나 노트북 CPU에서도 잘 작동하는 소형 모델들도 있습니다.

이 가이드는 채팅 모델을 처음 사용하는 분들에게 유용할 것입니다.
우리는 간편한 고수준(High-Level) "pipeline"을 통해 빠른 시작 가이드를 진행할 것입니다.
가이드에는 채팅 모델을 바로 시작할 때 필요한 모든 정보가 담겨 있습니다.
빠른 시작 가이드 이후에는 채팅 모델이 정확히 무엇인지, 적절한 모델을 선택하는 방법과, 
채팅 모델을 사용하는 각 단계의 저수준(Low-Level) 분석 등 더 자세한 정보를 다룰 것입니다. 
또한 채팅 모델의 성능과 메모리 사용을 최적화하는 방법에 대한 팁도 제공할 것입니다.


## 빠른 시작[[quickstart]]

자세히 볼 여유가 없는 분들을 위해 간단히 요약해 보겠습니다: 
채팅 모델은 대화 메세지를 계속해서 생성해 나갑니다.
즉, 짤막한 채팅 메세지를 모델에게 전달하면, 모델은 이를 바탕으로 응답을 추가하며 대화를 이어 나갑니다.
이제 실제로 어떻게 작동하는지 살펴보겠습니다. 
먼저, 채팅을 만들어 보겠습니다:


```python
chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]
```

주목하세요, 대화를 처음 시작할 때 유저 메세지 이외의도, 별도의 **시스템** 메세지가 필요할 수 있습니다.
모든 채팅 모델이 시스템 메세지를 지원하는 것은 아니지만,
지원하는 경우에는 시스템 메세지는 대화에서 모델이 어떻게 행동해야 하는지를 지시할 수 있습니다.
예를 들어, 유쾌하거나 진지하고자 할 때, 짧은 답변이나 긴 답변을 원할 때 등을 설정할 수 있습니다.
시스템 메세지를 생략하고
"You are a helpful and intelligent AI assistant who responds to user queries."
와 같은 간단한 프롬프트를 사용하는 것도 가능합니다.

채팅을 시작했다면 대화를 이어 나가는 가장 빠른 방법은 [`TextGenerationPipeline`]를 사용하는 것입니다. 
한번 `LLaMA-3`를 사용하여 이를 시연해 보겠습니다. 
우선 `LLaMA-3`를 사용하기 위해서는 승인이 필요합니다. [권한 신청](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)을 하고 Hugging Face 계정으로 로그인한 후에 사용할 수 있습니다. 
또한 우리는 `device_map="auto"`를 사용합니다. GPU 메모리가 충분하다면 로드될 것입니다. 
그리고 메모리 절약을 위해 dtype을 `torch.bfloat16`으로 설정할 것입니다.

```python
import torch
from transformers import pipeline

pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
response = pipe(chat, max_new_tokens=512)
print(response[0]['generated_text'][-1]['content'])
```

이후 실행을 하면 아래와 같이 출력됩니다:

```text
(sigh) Oh boy, you're asking me for advice? You're gonna need a map, pal! Alright, 
alright, I'll give you the lowdown. But don't say I didn't warn you, I'm a robot, not a tour guide!

So, you wanna know what's fun to do in the Big Apple? Well, let me tell you, there's a million 
things to do, but I'll give you the highlights. First off, you gotta see the sights: the Statue of 
Liberty, Central Park, Times Square... you know, the usual tourist traps. But if you're lookin' for 
something a little more... unusual, I'd recommend checkin' out the Museum of Modern Art. It's got 
some wild stuff, like that Warhol guy's soup cans and all that jazz.

And if you're feelin' adventurous, take a walk across the Brooklyn Bridge. Just watch out for 
those pesky pigeons, they're like little feathered thieves! (laughs) Get it? Thieves? Ah, never mind.

Now, if you're lookin' for some serious fun, hit up the comedy clubs in Greenwich Village. You might 
even catch a glimpse of some up-and-coming comedians... or a bunch of wannabes tryin' to make it big. (winks)

And finally, if you're feelin' like a real New Yorker, grab a slice of pizza from one of the many amazing
pizzerias around the city. Just don't try to order a "robot-sized" slice, trust me, it won't end well. (laughs)

So, there you have it, pal! That's my expert advice on what to do in New York. Now, if you'll
excuse me, I've got some oil changes to attend to. (winks)
```

채팅을 계속하려면, 자신의 답장을 추가하면 됩니다. 
파이프라인에서 반환된 `response` 객체에는 현재까지 모든 채팅을 포함하고 있으므로 
메세지를 추가하고 다시 전달하기만 하면 됩니다.

```python
chat = response[0]['generated_text']
chat.append(
    {"role": "user", "content": "Wait, what's so wild about soup cans?"}
)
response = pipe(chat, max_new_tokens=512)
print(response[0]['generated_text'][-1]['content'])
```

이후 실행을 하면 아래와 같이 출력됩니다:

```text
(laughs) Oh, you're killin' me, pal! You don't get it, do you? Warhol's soup cans are like, art, man! 
It's like, he took something totally mundane, like a can of soup, and turned it into a masterpiece. It's 
like, "Hey, look at me, I'm a can of soup, but I'm also a work of art!" 
(sarcastically) Oh, yeah, real original, Andy.

But, you know, back in the '60s, it was like, a big deal. People were all about challenging the
status quo, and Warhol was like, the king of that. He took the ordinary and made it extraordinary.
And, let me tell you, it was like, a real game-changer. I mean, who would've thought that a can of soup could be art? (laughs)

But, hey, you're not alone, pal. I mean, I'm a robot, and even I don't get it. (winks)
But, hey, that's what makes art, art, right? (laughs)
```

이 튜토리얼의 후반부에서는 성능과 메모리 관리, 
그리고 사용자의 필요에 맞는 채팅 모델 선택과 같은 구체적인 주제들을 다룰 것입니다.

## 채팅 모델 고르기[[choosing-a-chat-model]]

[Hugging Face Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending)는 채팅 모델을 다양하게 제공하고 있습니다.
처음 사용하는 사람에게는 모델을 선택하기가 어려울지 모릅니다.
하지만 걱정하지 마세요! 두 가지만 명심하면 됩니다:

- 모델의 크기는 실행 속도와 메모리에 올라올 수 있는지 여부를 결정.
- 모델이 생성한 출력의 품질.

일반적으로 이러한 요소들은 상관관계가 있습니다. 더 큰 모델일수록 더 뛰어난 성능을 보이는 경향이 있지만, 동일한 크기의 모델이라도 유의미한 차이가 날 수 있습니다!

### 모델의 명칭과 크기[[size-and-model-naming]]

모델의 크기는 모델 이름에 있는 숫자로 쉽게 알 수 있습니다. 
예를 들어, "8B" 또는 "70B"와 같은 숫자는 모델의 **파라미터** 수를 나타냅니다. 
양자화된 경우가 아니라면, 파라미터 하나당 약 2바이트의 메모리가 필요하다고 예상 가능합니다. 
따라서 80억 개의 파라미터를 가진 "8B" 모델은 16GB의 메모리를 차지하며, 추가적인 오버헤드를 위한 약간의 여유가 필요합니다. 
이는 3090이나 4090와 같은 24GB의 메모리를 갖춘 하이엔드 GPU에 적합합니다.

일부 채팅 모델은 "Mixture of Experts" 모델입니다. 
이러한 모델은 크기를 "8x7B" 또는 "141B-A35B"와 같이 다르게 표시하곤 합니다. 
숫자가 다소 모호하다 느껴질 수 있지만, 첫 번째 경우에는 약 56억(8x7) 개의 파라미터가 있고, 
두 번째 경우에는 약 141억 개의 파라미터가 있다고 해석할 수 있습니다.

양자화는 파라미터당 메모리 사용량을 8비트, 4비트, 또는 그 이하로 줄이는 데 사용됩니다. 
이 주제에 대해서는 아래의 [메모리 고려사항](#memory-considerations) 챕터에서 더 자세히 다룰 예정입니다.

### 그렇다면 어떤 채팅 모델이 가장 좋을까요?[[but-which-chat-model-is-best]]
모델의 크기 외에도 고려할 점이 많습니다. 
이를 한눈에 살펴보려면 **리더보드**를 참고하는 것이 좋습니다. 
가장 인기 있는 리더보드 두 가지는 [OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)와 [LMSys Chatbot Arena Leaderboard](https://chat.lmsys.org/?leaderboard)입니다. 
LMSys 리더보드에는 독점 모델도 포함되어 있으니,
`license` 열에서 접근 가능한 모델을 선택한 후
[Hugging Face Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending)에서 검색해 보세요.

### 전문 분야[[specialist-domains]]
일부 모델은 의료 또는 법률 텍스트와 같은 특정 도메인이나 비영어권 언어에 특화되어 있기도 합니다. 
이러한 도메인에서 작업할 경우 특화된 모델이 좋은 성능을 보일 수 있습니다. 
하지만 항상 그럴 것이라 단정하기는 힘듭니다. 
특히 모델의 크기가 작거나 오래된 모델인 경우, 
최신 범용 모델이 더 뛰어날 수 있습니다. 
다행히도 [domain-specific leaderboards](https://huggingface.co/blog/leaderboard-medicalllm)가 점차 등장하고 있어, 특정 도메인에 최고의 모델을 쉽게 찾을 수 있을 것입니다. 


## 파이프라인 내부는 어떻게 되어있는가?[[what-happens-inside-the-pipeline]]
위의 빠른 시작에서는 고수준(High-Level) 파이프라인을 사용하였습니다.
이는 간편한 방법이지만, 유연성은 떨어집니다.
이제 더 저수준(Low-Level) 접근 방식을 통해 대화에 포함된 각 단계를 살펴보겠습니다. 
코드 샘플로 시작한 후 이를 분석해 보겠습니다:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Prepare the input as before
chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

# 1: Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# 2: Apply the chat template
formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print("Formatted chat:\n", formatted_chat)

# 3: Tokenize the chat (This can be combined with the previous step using tokenize=True)
inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
# Move the tokenized inputs to the same device the model is on (GPU/CPU)
inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
print("Tokenized inputs:\n", inputs)

# 4: Generate text from the model
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
print("Generated tokens:\n", outputs)

# 5: Decode the output back to a string
decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
print("Decoded output:\n", decoded_output)
```
여기에는 각 부분이 자체 문서가 될 수 있을 만큼 많은 내용이 담겨 있습니다! 
너무 자세히 설명하기보다는 넓은 개념을 다루고, 세부 사항은 링크된 문서에서 다루겠습니다. 
주요 단계는 다음과 같습니다:

1. [모델](https://huggingface.co/learn/nlp-course/en/chapter2/3)과 [토크나이저](https://huggingface.co/learn/nlp-course/en/chapter2/4?fw=pt)를 Hugging Face Hub에서 로드합니다.
2. 대화는 토크나이저의 [채팅 템플릿](https://huggingface.co/docs/transformers/main/en/chat_templating)을 사용하여 양식을 구성합니다.
3. 구성된 채팅은 토크나이저를 사용하여 [토큰화](https://huggingface.co/learn/nlp-course/en/chapter2/4)됩니다.
4. 모델에서 응답을 [생성](https://huggingface.co/docs/transformers/en/llm_tutorial)합니다.
5. 모델이 출력한 토큰을 다시 문자열로 디코딩합니다.

## 성능, 메모리와 하드웨어[[performance-memory-and-hardware]]
이제 대부분의 머신 러닝 작업이 GPU에서 실행된다는 것을 아실 겁니다. 
다소 느리기는 해도 CPU에서 채팅 모델이나 언어 모델로부터 텍스트를 생성하는 것도 가능합니다. 
하지만 모델을 GPU 메모리에 올려놓을 수만 있다면, GPU를 사용하는 것이 일반적으로 더 선호되는 방식입니다.

### 메모리 고려사항[[memory-considerations]]

기본적으로, [`TextGenerationPipeline`]이나 [`AutoModelForCausalLM`]과 같은 
Hugging Face 클래스는 모델을 `float32` 정밀도(Precision)로 로드합니다. 
이는 파라미터당 4바이트(32비트)를 필요로 하므로, 
80억 개의 파라미터를 가진 "8B" 모델은 약 32GB의 메모리를 필요로 한다는 것을 의미합니다. 
하지만 이는 낭비일 수 있습니다! 
대부분의 최신 언어 모델은 파라미터당 2바이트를 사용하는 "bfloat16" 정밀도(Precision)로 학습됩니다. 
하드웨어가 이를 지원하는 경우(Nvidia 30xx/Axxx 이상), 
`torch_dtype` 파라미터로 위와 같이 `bfloat16` 정밀도(Precision)로 모델을 로드할 수 있습니다.

또한, 16비트보다 더 낮은 정밀도(Precision)로 모델을 압축하는 
"양자화(quantization)" 방법을 사용할 수도 있습니다. 
이 방법은 모델의 가중치를 손실 압축하여 각 파라미터를 8비트, 
4비트 또는 그 이하로 줄일 수 있습니다. 
특히 4비트에서 모델의 출력이 부정적인 영향을 받을 수 있지만, 
더 크고 강력한 채팅 모델을 메모리에 올리기 위해 이 같은 트레이드오프를 감수할 가치가 있습니다. 
이제 `bitsandbytes`를 사용하여 이를 실제로 확인해 보겠습니다:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # You can also try load_in_4bit
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto", quantization_config=quantization_config)
```

위의 작업은 `pipeline` API에도 적용 가능합니다:

```python
from transformers import pipeline, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # You can also try load_in_4bit
pipe = pipeline("text-generation", "meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto", model_kwargs={"quantization_config": quantization_config})
```

`bitsandbytes` 외에도 모델을 양자화하는 다양한 방법이 있습니다. 
자세한 내용은 [Quantization guide](./quantization)를 참조해 주세요.


### 성능 고려사항[[performance-considerations]]

<Tip>

언어 모델 성능과 최적화에 대한 보다 자세한 가이드는 [LLM Inference Optimization](./llm_optims)을 참고하세요.

</Tip>


일반적으로 더 큰 채팅 모델은 메모리를 더 많이 요구하고, 
속도도 느려지는 경향이 있습니다. 구체적으로 말하자면, 
채팅 모델에서 텍스트를 생성할 때는 컴퓨팅 파워보다 **메모리 대역폭**이 병목 현상을 일으키는 경우가 많습니다. 
이는 모델이 토큰을 하나씩 생성할 때마다 파라미터를 메모리에서 읽어야 하기 때문입니다. 
따라서 채팅 모델에서 초당 생성할 수 있는 토큰 수는 모델이 위치한 메모리의 대역폭을 모델의 크기로 나눈 값에 비례합니다.

위의 예제에서는 모델이 bfloat16 정밀도(Precision)로 로드될 때 용량이 약 16GB였습니다. 
이 경우, 모델이 생성하는 각 토큰마다 16GB를 메모리에서 읽어야 한다는 의미입니다. 
총 메모리 대역폭은 소비자용 CPU에서는 20-100GB/sec, 
소비자용 GPU나 Intel Xeon, AMD Threadripper/Epyc, 
애플 실리콘과 같은 특수 CPU에서는 200-900GB/sec, 
데이터 센터 GPU인 Nvidia A100이나 H100에서는 최대 2-3TB/sec에 이를 수 있습니다. 
이러한 정보는 각자 하드웨어에서 생성 속도를 예상하는 데 도움이 될 것입니다.

따라서 텍스트 생성 속도를 개선하려면 가장 간단한 방법은 모델의 크기를 줄이거나(주로 양자화를 사용), 
메모리 대역폭이 더 높은 하드웨어를 사용하는 것입니다. 
이 대역폭 병목 현상을 피할 수 있는 고급 기술도 여러 가지 있습니다. 
가장 일반적인 방법은 [보조 생성](https://huggingface.co/blog/assisted-generation), "추측 샘플링"이라고 불리는 기술입니다. 
이 기술은 종종 더 작은 "초안 모델"을 사용하여 여러 개의 미래 토큰을 한 번에 추측한 후, 
채팅 모델로 생성 결과를 확인합니다.
만약 채팅 모델이 추측을 확인하면, 한 번의 순전파에서 여러 개의 토큰을 생성할 수 있어 
병목 현상이 크게 줄어들고 생성 속도가 빨라집니다.

마지막으로, "Mixture of Experts" (MoE) 모델에 대해서도 짚고 넘어가 보도록 합니다. 
Mixtral, Qwen-MoE, DBRX와 같은 인기 있는 채팅 모델이 바로 MoE 모델입니다. 
이 모델들은 토큰을 생성할 때 모든 파라미터가 사용되지 않습니다. 
이로 인해 MoE 모델은 전체 크기가 상당히 클 수 있지만, 
차지하는 메모리 대역폭은 낮은 편입니다. 
따라서 동일한 크기의 일반 "조밀한(Dense)" 모델보다 몇 배 빠를 수 있습니다. 
하지만 보조 생성과 같은 기술은 MoE 모델에서 비효율적일 수 있습니다. 
새로운 추측된 토큰이 추가되면서 더 많은 파라미터가 활성화되기 때문에, 
MoE 아키텍처가 제공하는 속도 이점이 상쇄될 수 있습니다.