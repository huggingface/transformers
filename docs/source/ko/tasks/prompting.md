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


# 대규모 언어 모델(LLM) 프롬프팅 가이드 [[llm-prompting-guide]]

[[open-in-colab]]

Falcon, LLaMA 등의 대규모 언어 모델은 사전 훈련된 트랜스포머 모델로, 초기에는 주어진 입력 텍스트에 대해 다음 토큰을 예측하도록 훈련됩니다. 이들은 보통 수십억 개의 매개변수를 가지고 있으며, 장기간에 걸쳐 수조 개의 토큰으로 훈련됩니다. 그 결과, 이 모델들은 매우 강력하고 다재다능해져서, 자연어 프롬프트로 모델에 지시하여 다양한 자연어 처리 작업을 즉시 수행할 수 있습니다.

최적의 출력을 보장하기 위해 이러한 프롬프트를 설계하는 것을 흔히 "프롬프트 엔지니어링"이라고 합니다. 프롬프트 엔지니어링은 상당한 실험이 필요한 반복적인 과정입니다. 자연어는 프로그래밍 언어보다 훨씬 유연하고 표현력이 풍부하지만, 동시에 모호성을 초래할 수 있습니다. 또한, 자연어 프롬프트는 변화에 매우 민감합니다. 프롬프트의 사소한 수정만으로도 완전히 다른 출력이 나올 수 있습니다.

모든 경우에 적용할 수 있는 정확한 프롬프트 생성 공식은 없지만, 연구자들은 더 일관되게 최적의 결과를 얻는 데 도움이 되는 여러 가지 모범 사례를 개발했습니다.

이 가이드에서는 더 나은 대규모 언어 모델 프롬프트를 작성하고 다양한 자연어 처리 작업을 해결하는 데 도움이 되는 프롬프트 엔지니어링 모범 사례를 다룹니다:

- [프롬프팅의 기초](#basics-of-prompting)
- [대규모 언어 모델 프롬프팅의 모범 사례](#best-practices-of-llm-prompting)
- [고급 프롬프팅 기법: 퓨샷(Few-shot) 프롬프팅과 생각의 사슬(Chain-of-thought, CoT) 기법](#advanced-prompting-techniques)
- [프롬프팅 대신 미세 조정을 해야 하는 경우](#prompting-vs-fine-tuning)

<Tip>

프롬프트 엔지니어링은 대규모 언어 모델 출력 최적화 과정의 일부일 뿐입니다. 또 다른 중요한 구성 요소는 최적의 텍스트 생성 전략을 선택하는 것입니다. 학습 가능한 매개변수를 수정하지 않고도 대규모 언어 모델이 텍스트를 생성하리 때 각각의 후속 토큰을 선택하는 방식을 사용자가 직접 정의할 수 있습니다. 텍스트 생성 매개변수를 조정함으로써 생성된 텍스트의 반복을 줄이고 더 일관되고 사람이 말하는 것 같은 텍스트를 만들 수 있습니다. 텍스트 생성 전략과 매개변수는 이 가이드의 범위를 벗어나지만, 다음 가이드에서 이러한 주제에 대해 자세히 알아볼 수 있습니다:
 
* [대규모 언어 모델을 이용한 생성](../llm_tutorial)
* [텍스트 생성 전략](../generation_strategies)

</Tip>

## 프롬프팅의 기초 [[basics-of-prompting]]

### 모델의 유형 [[types-of-models]]

현대의 대부분의 대규모 언어 모델은 디코더만을 이용한 트랜스포머입니다. 예를 들어 [LLaMA](../model_doc/llama), 
[Llama2](../model_doc/llama2), [Falcon](../model_doc/falcon), [GPT2](../model_doc/gpt2) 등이 있습니다. 그러나 [Flan-T5](../model_doc/flan-t5)와 [BART](../model_doc/bart)와 같은 인코더-디코더 기반의 트랜스포머 대규모 언어 모델을 접할 수도 있습니다.

인코더-디코더 기반의 모델은 일반적으로 출력이 입력에 **크게** 의존하는 생성 작업에 사용됩니다. 예를 들어, 번역과 요약 작업에 사용됩니다. 디코더 전용 모델은 다른 모든 유형의 생성 작업에 사용됩니다.

파이프라인을 사용하여 대규모 언어 모델으로 텍스트를 생성할 때, 어떤 유형의 대규모 언어 모델을 사용하고 있는지 아는 것이 중요합니다. 왜냐하면 이들은 서로 다른 파이프라인을 사용하기 때문입니다.

디코더 전용 모델로 추론을 실행하려면 `text-generation` 파이프라인을 사용하세요:

```python
>>> from transformers import pipeline
>>> import torch

>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT

>>> generator = pipeline('text-generation', model = 'openai-community/gpt2')
>>> prompt = "Hello, I'm a language model"

>>> generator(prompt, max_length = 30)
[{'generated_text': "Hello, I'm a language model programmer so you can use some of my stuff. But you also need some sort of a C program to run."}]
```

인코더-디코더로 추론을 실행하려면 `text2text-generation` 파이프라인을 사용하세요:

```python
>>> text2text_generator = pipeline("text2text-generation", model = 'google/flan-t5-base')
>>> prompt = "Translate from English to French: I'm very happy to see you"

>>> text2text_generator(prompt)
[{'generated_text': 'Je suis très heureuse de vous rencontrer.'}]
```

### 기본 모델 vs 지시/채팅 모델 [[base-vs-instructchat-models]]

🤗 Hub에서 최근 사용 가능한 대부분의 대규모 언어 모델 체크포인트는 기본 버전과 지시(또는 채팅) 두 가지 버전이 제공됩니다. 예를 들어, [`tiiuae/falcon-7b`](https://huggingface.co/tiiuae/falcon-7b)와 [`tiiuae/falcon-7b-instruct`](https://huggingface.co/tiiuae/falcon-7b-instruct)가 있습니다.

기본 모델은 초기 프롬프트가 주어졌을 때 텍스트를 완성하는 데 탁월하지만, 지시를 따라야 하거나 대화형 사용이 필요한 자연어 처리작업에는 이상적이지 않습니다. 이때 지시(채팅) 버전이 필요합니다. 이러한 체크포인트는 사전 훈련된 기본 버전을 지시사항과 대화 데이터로 추가 미세 조정한 결과입니다. 이 추가적인 미세 조정으로 인해 많은 자연어 처리 작업에 더 적합한 선택이 됩니다.  

[`tiiuae/falcon-7b-instruct`](https://huggingface.co/tiiuae/falcon-7b-instruct)를 사용하여 일반적인 자연어 처리 작업을 해결하는 데 사용할 수 있는 몇 가지 간단한 프롬프트를 살펴보겠습니다.

### 자연어 처리 작업 [[nlp-tasks]]

먼저, 환경을 설정해 보겠습니다:

```bash
pip install -q transformers accelerate
```

다음으로, 적절한 파이프라인("text-generation")을 사용하여 모델을 로드하겠습니다:

```python
>>> from transformers import pipeline, AutoTokenizer
>>> import torch

>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT
>>> model = "tiiuae/falcon-7b-instruct"

>>> tokenizer = AutoTokenizer.from_pretrained(model)
>>> pipe = pipeline(
...     "text-generation",
...     model=model,
...     tokenizer=tokenizer,
...     dtype=torch.bfloat16,
...     device_map="auto",
... )
```

<Tip>

Falcon 모델은 bfloat16 데이터 타입을 사용하여 훈련되었으므로, 같은 타입을 사용하는 것을 권장합니다. 이를 위해서는 최신 버전의 CUDA가 필요하며, 최신 그래픽 카드에서 가장 잘 작동합니다.

</Tip>

이제 파이프라인을 통해 모델을 로드했으니, 프롬프트를 사용하여 자연어 처리 작업을 해결하는 방법을 살펴보겠습니다.

#### 텍스트 분류 [[text-classification]]

텍스트 분류의 가장 일반적인 형태 중 하나는 감정 분석입니다. 이는 텍스트 시퀀스에 "긍정적", "부정적" 또는 "중립적"과 같은 레이블을 할당합니다. 주어진 텍스트(영화 리뷰)를 분류하도록 모델에 지시하는 프롬프트를 작성해 보겠습니다. 먼저 지시사항을 제공한 다음, 분류할 텍스트를 지정하겠습니다. 여기서 주목할 점은 단순히 거기서 끝내지 않고, 응답의 시작 부분인 `"Sentiment: "`을 추가한다는 것입니다:

```python
>>> torch.manual_seed(0)
>>> prompt = """Classify the text into neutral, negative or positive. 
... Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
... Sentiment:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=10,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: Classify the text into neutral, negative or positive. 
Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
Sentiment:
Positive
```

결과적으로, 우리가 지시사항에서 제공한 목록에서 선택된 분류 레이블이 정확하게 포함되어 생성된 것을 확인할 수 있습니다!
<Tip>

프롬프트 외에도 `max_new_tokens` 매개변수를 전달하는 것을 볼 수 있습니다. 이 매개변수는 모델이 생성할 토큰의 수를 제어하며, [텍스트 생성 전략](../generation_strategies) 가이드에서 배울 수 있는 여러 텍스트 생성 매개변수 중 하나입니다.

</Tip>

#### 개체명 인식 [[named-entity-recognition]]

개체명 인식(Named Entity Recognition, NER)은 텍스트에서 인물, 장소, 조직과 같은 명명된 개체를 찾는 작업입니다. 프롬프트의 지시사항을 수정하여 대규모 언어 모델이 이 작업을 수행하도록 해보겠습니다. 여기서는 `return_full_text = False`로 설정하여 출력에 프롬프트가 포함되지 않도록 하겠습니다:

```python
>>> torch.manual_seed(1) # doctest: +IGNORE_RESULT
>>> prompt = """Return a list of named entities in the text.
... Text: The Golden State Warriors are an American professional basketball team based in San Francisco.
... Named entities:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=15,
...     return_full_text = False,    
... )

>>> for seq in sequences:
...     print(f"{seq['generated_text']}")
- Golden State Warriors
- San Francisco
```

보시다시피, 모델이 주어진 텍스트에서 두 개의 명명된 개체를 정확하게 식별했습니다.

#### 번역 [[translation]]

대규모 언어 모델이 수행할 수 있는 또 다른 작업은 번역입니다. 이 작업을 위해 인코더-디코더 모델을 사용할 수 있지만, 여기서는 예시의 단순성을 위해 꽤 좋은 성능을 보이는 Falcon-7b-instruct를 계속 사용하겠습니다. 다시 한 번, 모델에게 영어에서 이탈리아어로 텍스트를 번역하도록 지시하는 기본적인 프롬프트를 작성하는 방법은 다음과 같습니다:

```python
>>> torch.manual_seed(2) # doctest: +IGNORE_RESULT
>>> prompt = """Translate the English text to Italian.
... Text: Sometimes, I've believed as many as six impossible things before breakfast.
... Translation:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=20,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"{seq['generated_text']}")
A volte, ho creduto a sei impossibili cose prima di colazione.
```

여기서는 모델이 출력을 생성할 때 조금 더 유연해질 수 있도록 `do_sample=True`와 `top_k=10`을 추가했습니다.

#### 텍스트 요약 [[text-summarization]]

번역과 마찬가지로, 텍스트 요약은 출력이 입력에 크게 의존하는 또 다른 생성 작업이며, 인코더-디코더 기반 모델이 더 나은 선택일 수 있습니다. 그러나 디코더 기반의 모델도 이 작업에 사용될 수 있습니다. 이전에는 프롬프트의 맨 처음에 지시사항을 배치했습니다. 하지만 프롬프트의 맨 끝도 지시사항을 넣을 적절한 위치가 될 수 있습니다. 일반적으로 지시사항을 양 극단 중 하나에 배치하는 것이 더 좋습니다.

```python
>>> torch.manual_seed(3) # doctest: +IGNORE_RESULT
>>> prompt = """Permaculture is a design process mimicking the diversity, functionality and resilience of natural ecosystems. The principles and practices are drawn from traditional ecological knowledge of indigenous cultures combined with modern scientific understanding and technological innovations. Permaculture design provides a framework helping individuals and communities develop innovative, creative and effective strategies for meeting basic needs while preparing for and mitigating the projected impacts of climate change.
... Write a summary of the above text.
... Summary:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=30,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"{seq['generated_text']}")
Permaculture is an ecological design mimicking natural ecosystems to meet basic needs and prepare for climate change. It is based on traditional knowledge and scientific understanding.
```

#### 질의 응답 [[question-answering]]

질의 응답 작업을 위해 프롬프트를 다음과 같은 논리적 구성요소로 구조화할 수 있습니다. 지시사항, 맥락, 질문, 그리고 모델이 답변 생성을 시작하도록 유도하는 선도 단어나 구문(`"Answer:"`) 을 사용할 수 있습니다:

```python
>>> torch.manual_seed(4) # doctest: +IGNORE_RESULT
>>> prompt = """Answer the question using the context below.
... Context: Gazpacho is a cold soup and drink made of raw, blended vegetables. Most gazpacho includes stale bread, tomato, cucumbers, onion, bell peppers, garlic, olive oil, wine vinegar, water, and salt. Northern recipes often include cumin and/or pimentón (smoked sweet paprika). Traditionally, gazpacho was made by pounding the vegetables in a mortar with a pestle; this more laborious method is still sometimes used as it helps keep the gazpacho cool and avoids the foam and silky consistency of smoothie versions made in blenders or food processors.
... Question: What modern tool is used to make gazpacho?
... Answer:
... """

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=10,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: Modern tools often used to make gazpacho include
```

#### 추론 [[reasoning]]

추론은 대규모 언어 모델(LLM)에게 가장 어려운 작업 중 하나이며, 좋은 결과를 얻기 위해서는 종종 [생각의 사슬(Chain-of-thought, CoT)](#chain-of-thought)과 같은 고급 프롬프팅 기법을 적용해야 합니다. 간단한 산술 작업에 대해 기본적인 프롬프트로 모델이 추론할 수 있는지 시도해 보겠습니다:

```python
>>> torch.manual_seed(5) # doctest: +IGNORE_RESULT
>>> prompt = """There are 5 groups of students in the class. Each group has 4 students. How many students are there in the class?"""

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=30,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: 
There are a total of 5 groups, so there are 5 x 4=20 students in the class.
```

정확한 답변이 생성되었습니다! 복잡성을 조금 높여보고 기본적인 프롬프트로도 여전히 해결할 수 있는지 확인해 보겠습니다:

```python
>>> torch.manual_seed(6)
>>> prompt = """I baked 15 muffins. I ate 2 muffins and gave 5 muffins to a neighbor. My partner then bought 6 more muffins and ate 2. How many muffins do we now have?"""

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=10,
...     do_sample=True,
...     top_k=10,
...     return_full_text = False,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: 
The total number of muffins now is 21
```

정답은 12여야 하는데 21이라는 잘못된 답변이 나왔습니다. 이 경우, 프롬프트가 너무 기본적이거나 모델의 크기가 작아서 생긴 문제일 수 있습니다. 우리는 Falcon의 가장 작은 버전을 선택했습니다. 추론은 큰 모델에게도 어려운 작업이지만, 더 큰 모델들이 더 나은 성능을 보일 가능성이 높습니다.

## 대규모 언어 모델 프롬프트 작성의 모범 사례 [[best-practices-of-llm-prompting]]

이 섹션에서는 프롬프트 결과를 향상시킬 수 있는 모범 사례 목록을 작성했습니다:

* 작업할 모델을 선택할 때 최신 및 가장 강력한 모델이 더 나은 성능을 발휘할 가능성이 높습니다.
* 간단하고 짧은 프롬프트로 시작하여 점진적으로 개선해 나가세요.
* 프롬프트의 시작 부분이나 맨 끝에 지시사항을 배치하세요. 대규모 컨텍스트를 다룰 때, 모델들은 어텐션 복잡도가 2차적으로 증가하는 것을 방지하기 위해 다양한 최적화를 적용합니다. 이렇게 함으로써 모델이 프롬프트의 중간보다 시작이나 끝 부분에 더 주의를 기울일 수 있습니다.
* 지시사항을 적용할 텍스트와 명확하게 분리해보세요. (이에 대해서는 다음 섹션에서 더 자세히 다룹니다.)
* 작업과 원하는 결과에 대해 구체적이고 풍부한 설명을 제공하세요.  형식, 길이, 스타일, 언어 등을 명확하게 작성해야 합니다.
* 모호한 설명과 지시사항을 피하세요.
* "하지 말라"는 지시보다는 "무엇을 해야 하는지"를 말하는 지시를 사용하는 것이 좋습니다.
* 첫 번째 단어를 쓰거나 첫 번째 문장을 시작하여 출력을 올바른 방향으로 "유도"하세요.
* [퓨샷(Few-shot) 프롬프팅](#few-shot-prompting) 및 [생각의 사슬(Chain-of-thought, CoT)](#chain-of-thought) 같은 고급 기술을 사용해보세요.
* 프롬프트의 견고성을 평가하기 위해 다른 모델로도 테스트하세요.
* 프롬프트의 버전을 관리하고 성능을 추적하세요.

## 고급 프롬프트 기법 [[advanced-prompting-techniques]]

### 퓨샷(Few-shot) 프롬프팅 [[few-shot-prompting]]

위 섹션의 기본 프롬프트들은 "제로샷(Zero-shot)" 프롬프트의 예시입니다. 이는 모델에 지시사항과 맥락은 주어졌지만, 해결책이 포함된 예시는 제공되지 않았다는 의미입니다. 지시 데이터셋으로 미세 조정된 대규모 언어 모델은 일반적으로 이러한 "제로샷" 작업에서 좋은 성능을 보입니다. 하지만 여러분의 작업이 더 복잡하거나 미묘한 차이가 있을 수 있고, 아마도 지시사항만으로는 모델이 포착하지 못하는 출력에 대한 요구사항이 있을 수 있습니다. 이런 경우에는 퓨샷(Few-shot) 프롬프팅이라는 기법을 시도해 볼 수 있습니다.

퓨샷 프롬프팅에서는 프롬프트에 예시를 제공하여 모델에 더 많은 맥락을 주고 성능을 향상시킵니다. 이 예시들은 모델이 예시의 패턴을 따라 출력을 생성하도록 조건화합니다.

다음은 예시입니다:

```python
>>> torch.manual_seed(0) # doctest: +IGNORE_RESULT
>>> prompt = """Text: The first human went into space and orbited the Earth on April 12, 1961.
... Date: 04/12/1961
... Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon. 
... Date:"""

>>> sequences = pipe(
...     prompt,
...     max_new_tokens=8,
...     do_sample=True,
...     top_k=10,
... )

>>> for seq in sequences:
...     print(f"Result: {seq['generated_text']}")
Result: Text: The first human went into space and orbited the Earth on April 12, 1961.
Date: 04/12/1961
Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon. 
Date: 09/28/1960
```

위의 코드 스니펫에서는 모델에 원하는 출력을 보여주기 위해 단일 예시를 사용했으므로, 이를 "원샷(One-shot)" 프롬프팅이라고 부를 수 있습니다. 그러나 작업의 복잡성에 따라 하나 이상의 예시를 사용해야 할 수도 있습니다.

퓨샷 프롬프팅 기법의 한계:
- 대규모 언어 모델이 예시의 패턴을 파악할 수 있지만, 이 기법은 복잡한 추론 작업에는 잘 작동하지 않습니다.
- 퓨샷 프롬프팅을 적용하면 프롬프트의 길이가 길어집니다. 토큰 수가 많은 프롬프트는 계산량과 지연 시간을 증가시킬 수 있으며 프롬프트 길이에도 제한이 있습니다.
- 때로는 여러 예시가 주어질 때, 모델은 의도하지 않은 패턴을 학습할 수 있습니다. 예를 들어, 세 번째 영화 리뷰가 항상 부정적이라고 학습할 수 있습니다.

### 생각의 사슬(Chain-of-thought, CoT) [[chain-of-thought]]

생각의 사슬(Chain-of-thought, CoT) 프롬프팅은 모델이 중간 추론 단계를 생성하도록 유도하는 기법으로, 복잡한 추론 작업의 결과를 개선합니다.

모델이 추론 단계를 생성하도록 유도하는 두 가지 방법이 있습니다:
- 질문에 대한 상세한 답변을 예시로 제시하는 퓨샷 프롬프팅을 통해 모델에게 문제를 어떻게 해결해 나가는지 보여줍니다.
- "단계별로 생각해 봅시다" 또는 "깊게 숨을 쉬고 문제를 단계별로 해결해 봅시다"와 같은 문구를 추가하여 모델에게 추론하도록 지시합니다.

[reasoning section](#reasoning)의 머핀 예시에 생각의 사슬(Chain-of-thought, CoT) 기법을 적용하고 [HuggingChat](https://huggingface.co/chat/)에서 사용할 수 있는 (`tiiuae/falcon-180B-chat`)과 같은 더 큰 모델을 사용하면, 추론 결과가 크게 개선됩니다:

```text
단계별로 살펴봅시다:
1. 처음에 15개의 머핀이 있습니다.
2. 2개의 머핀을 먹으면 13개의 머핀이 남습니다.
3. 이웃에게 5개의 머핀을 주면 8개의 머핀이 남습니다.
4. 파트너가 6개의 머핀을 더 사오면 총 머핀 수는 14개가 됩니다.
5. 파트너가 2개의 머핀을 먹으면 12개의 머핀이 남습니다.
따라서, 현재 12개의 머핀이 있습니다.
```

## 프롬프팅 vs 미세 조정 [[prompting-vs-fine-tuning]]

프롬프트를 최적화하여 훌륭한 결과를 얻을 수 있지만, 여전히 모델을 미세 조정하는 것이 더 좋을지 고민할 수 있습니다. 다음은 더 작은 모델을 미세 조정하는 것이 선호되는 시나리오입니다:

- 도메인이 대규모 언어 모델이 사전 훈련된 것과 크게 다르고 광범위한 프롬프트 최적화로도 충분한 결과를 얻지 못한 경우. 
- 저자원 언어에서 모델이 잘 작동해야 하는 경우.
- 엄격한 규제 하에 있는 민감한 데이터로 모델을 훈련해야 하는 경우.
- 비용, 개인정보 보호, 인프라 또는 기타 제한으로 인해 작은 모델을 사용해야 하는 경우. 

위의 모든 예시에서, 모델을 미세 조정하기 위해 충분히 큰 도메인별 데이터셋을 이미 가지고 있거나 합리적인 비용으로 쉽게 얻을 수 있는지 확인해야 합니다. 또한 모델을 미세 조정할 충분한 시간과 자원이 필요합니다.

만약 위의 예시들이 여러분의 경우에 해당하지 않는다면, 프롬프트를 최적화하는 것이 더 유익할 수 있습니다.
