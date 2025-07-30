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

# IDEFICS를 이용한 이미지 작업[[image-tasks-with-idefics]]

[[open-in-colab]]

개별 작업은 특화된 모델을 미세 조정하여 처리할 수 있지만, 최근 등장하여 인기를 얻고 있는 방식은 대규모 모델을 미세 조정 없이 다양한 작업에 사용하는 것입니다. 예를 들어, 대규모 언어 모델은 요약, 번역, 분류 등과 같은 자연어처리 (NLP) 작업을 처리할 수 있습니다. 이 접근 방식은 텍스트와 같은 단일 모달리티에 국한되지 않으며, 이 가이드에서는 IDEFICS라는 대규모 멀티모달 모델을 사용하여 이미지-텍스트 작업을 다루는 방법을 설명합니다.

[IDEFICS](../model_doc/idefics)는 [Flamingo](https://huggingface.co/papers/2204.14198)를 기반으로 하는 오픈 액세스 비전 및 언어 모델로, DeepMind에서 처음 개발한 최신 시각 언어 모델입니다. 이 모델은 임의의 이미지 및 텍스트 입력 시퀀스를 받아 일관성 있는 텍스트를 출력으로 생성합니다. 이미지에 대한 질문에 답변하고, 시각적인 내용을 설명하며, 여러 이미지에 기반한 이야기를 생성하는 등 다양한 작업을 수행할 수 있습니다. IDEFICS는 [800억 파라미터](https://huggingface.co/HuggingFaceM4/idefics-80b)와 [90억 파라미터](https://huggingface.co/HuggingFaceM4/idefics-9b) 두 가지 버전을 제공하며, 두 버전 모두 🤗 Hub에서 이용할 수 있습니다. 각 버전에는 대화형 사용 사례에 맞게 미세 조정된 버전도 있습니다.

이 모델은 매우 다재다능하며 광범위한 이미지 및 멀티모달 작업에 사용될 수 있습니다. 그러나 대규모 모델이기 때문에 상당한 컴퓨팅 자원과 인프라가 필요합니다. 각 개별 작업에 특화된 모델을 미세 조정하는 것보다 모델을 그대로 사용하는 것이 더 적합한지는 사용자가 판단해야 합니다.

이 가이드에서는 다음을 배우게 됩니다:
- [IDEFICS 로드하기](#loading-the-model) 및 [양자화된 버전의 모델 로드하기](#quantized-model)
- IDEFICS를 사용하여:
  - [이미지 캡셔닝](#image-captioning)
  - [프롬프트 이미지 캡셔닝](#prompted-image-captioning)
  - [퓨샷 프롬프트](#few-shot-prompting)
  - [시각적 질의 응답](#visual-question-answering)
  - [이미지 분류](#image-classification)
  - [이미지 기반 텍스트 생성](#image-guided-text-generation)
- [배치 모드에서 추론 실행](#running-inference-in-batch-mode)
- [대화형 사용을 위한 IDEFICS 인스트럭트 실행](#idefics-instruct-for-conversational-use)

시작하기 전에 필요한 모든 라이브러리가 설치되어 있는지 확인하세요.

```bash
pip install -q bitsandbytes sentencepiece accelerate transformers
```

<Tip>
다음 예제를 비양자화된 버전의 모델 체크포인트로 실행하려면 최소 20GB의 GPU 메모리가 필요합니다.
</Tip>

## 모델 로드[[loading-the-model]]

모델을 90억 파라미터 버전의 체크포인트로 로드해 봅시다:

```py
>>> checkpoint = "HuggingFaceM4/idefics-9b"
```

다른 Transformers 모델과 마찬가지로, 체크포인트에서 프로세서와 모델 자체를 로드해야 합니다.
IDEFICS 프로세서는 [`LlamaTokenizer`]와 IDEFICS 이미지 프로세서를 하나의 프로세서로 감싸서 텍스트와 이미지 입력을 모델에 맞게 준비합니다.

```py
>>> import torch

>>> from transformers import IdeficsForVisionText2Text, AutoProcessor

>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> model = IdeficsForVisionText2Text.from_pretrained(checkpoint, dtype=torch.bfloat16, device_map="auto")
```

`device_map`을 `"auto"`로 설정하면 사용 중인 장치를 고려하여 모델 가중치를 가장 최적화된 방식으로 로드하고 저장하는 방법을 자동으로 결정합니다.

### 양자화된 모델[[quantized-model]]

고용량 GPU 사용이 어려운 경우, 모델의 양자화된 버전을 로드할 수 있습니다. 모델과 프로세서를 4비트 정밀도로 로드하기 위해서, `from_pretrained` 메소드에 `BitsAndBytesConfig`를 전달하면 모델이 로드되는 동안 실시간으로 압축됩니다.

```py
>>> import torch
>>> from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig

>>> quantization_config = BitsAndBytesConfig(
...     load_in_4bit=True,
...     bnb_4bit_compute_dtype=torch.float16,
... )

>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> model = IdeficsForVisionText2Text.from_pretrained(
...     checkpoint,
...     quantization_config=quantization_config,
...     device_map="auto"
... )
```

이제 모델을 제안된 방법 중 하나로 로드했으니, IDEFICS를 사용할 수 있는 작업들을 탐구해봅시다.

## 이미지 캡셔닝[[image-captioning]]
이미지 캡셔닝은 주어진 이미지에 대한 캡션을 예측하는 작업입니다. 일반적인 응용 분야는 시각 장애인이 다양한 상황을 탐색할 수 있도록 돕는 것입니다. 예를 들어, 온라인에서 이미지 콘텐츠를 탐색하는 데 도움을 줄 수 있습니다.

작업을 설명하기 위해 캡션을 달 이미지 예시를 가져옵니다. 예시:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-im-captioning.jpg" alt="Image of a puppy in a flower bed"/>
</div>

사진 제공: [Hendo Wang](https://unsplash.com/@hendoo).

IDEFICS는 텍스트 및 이미지 프롬프트를 모두 수용합니다. 그러나 이미지를 캡션하기 위해 모델에 텍스트 프롬프트를 제공할 필요는 없습니다. 전처리된 입력 이미지만 제공하면 됩니다. 텍스트 프롬프트 없이 모델은 BOS(시퀀스 시작) 토큰부터 텍스트 생성을 시작하여 캡션을 만듭니다.

모델에 이미지 입력으로는 이미지 객체(`PIL.Image`) 또는 이미지를 가져올 수 있는 URL을 사용할 수 있습니다.

```py
>>> prompt = [
...     "https://images.unsplash.com/photo-1583160247711-2191776b4b91?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3542&q=80",
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
A puppy in a flower bed
```

<Tip>

`max_new_tokens`의 크기를 증가시킬 때 발생할 수 있는 오류를 피하기 위해 `generate` 호출 시 `bad_words_ids`를 포함하는 것이 좋습니다. 모델로부터 생성된 이미지가 없을 때 새로운 `<image>` 또는 `<fake_token_around_image>` 토큰을 생성하려고 하기 때문입니다.
이 가이드에서처럼 `bad_words_ids`를 함수 호출 시에 매개변수로 설정하거나, [텍스트 생성 전략](../generation_strategies) 가이드에 설명된 대로 `GenerationConfig`에 저장할 수도 있습니다.
</Tip>

## 프롬프트 이미지 캡셔닝[[prompted-image-captioning]]

텍스트 프롬프트를 이용하여 이미지 캡셔닝을 확장할 수 있으며, 모델은 주어진 이미지를 바탕으로 텍스트를 계속 생성합니다. 다음 이미지를 예시로 들어보겠습니다:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-prompted-im-captioning.jpg" alt="Image of the Eiffel Tower at night"/>
</div>

사진 제공: [Denys Nevozhai](https://unsplash.com/@dnevozhai).

텍스트 및 이미지 프롬프트는 적절한 입력을 생성하기 위해 모델의 프로세서에 하나의 목록으로 전달될 수 있습니다.

```py
>>> prompt = [
...     "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3501&q=80",
...     "This is an image of ",
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
This is an image of the Eiffel Tower in Paris, France.
```

## 퓨샷 프롬프트[[few-shot-prompting]]

IDEFICS는 훌륭한 제로샷 결과를 보여주지만, 작업에 특정 형식의 캡션이 필요하거나 작업의 복잡성을 높이는 다른 제한 사항이나 요구 사항이 있을 수 있습니다. 이럴 때 퓨샷 프롬프트를 사용하여 맥락 내 학습(In-Context Learning)을 가능하게 할 수 있습니다.
프롬프트에 예시를 제공함으로써 모델이 주어진 예시의 형식을 모방한 결과를 생성하도록 유도할 수 있습니다.

이전의 에펠탑 이미지를 모델에 예시로 사용하고, 모델에게 이미지의 객체를 학습하는 것 외에도 흥미로운 정보를 얻고 싶다는 것을 보여주는 프롬프트를 작성해 봅시다.
그런 다음 자유의 여신상 이미지에 대해 동일한 응답 형식을 얻을 수 있는지 확인해 봅시다:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg" alt="Image of the Statue of Liberty"/>
</div>

사진 제공: [Juan Mayobre](https://unsplash.com/@jmayobres).
  
```py
>>> prompt = ["User:",
...            "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3501&q=80",
...            "Describe this image.\nAssistant: An image of the Eiffel Tower at night. Fun fact: the Eiffel Tower is the same height as an 81-storey building.\n",
...            "User:",
...            "https://images.unsplash.com/photo-1524099163253-32b7f0256868?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3387&q=80",
...            "Describe this image.\nAssistant:"
...            ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=30, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
User: Describe this image.
Assistant: An image of the Eiffel Tower at night. Fun fact: the Eiffel Tower is the same height as an 81-storey building. 
User: Describe this image.
Assistant: An image of the Statue of Liberty. Fun fact: the Statue of Liberty is 151 feet tall.
```

단 하나의 예시만으로도(즉, 1-shot) 모델이 작업 수행 방법을 학습했다는 점이 주목할 만합니다. 더 복잡한 작업의 경우, 더 많은 예시(예: 3-shot, 5-shot 등)를 사용하여 실험해 보는 것도 좋은 방법입니다. 

## 시각적 질의 응답[[visual-question-answering]]

시각적 질의 응답(VQA)은 이미지를 기반으로 개방형 질문에 답하는 작업입니다. 이미지 캡셔닝과 마찬가지로 접근성 애플리케이션에서 사용할 수 있지만, 교육(시각 자료에 대한 추론), 고객 서비스(이미지를 기반으로 한 제품 질문), 이미지 검색 등에서도 사용할 수 있습니다.

이 작업을 위해 새로운 이미지를 가져옵니다:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-vqa.jpg" alt="Image of a couple having a picnic"/>
</div>

사진 제공: [Jarritos Mexican Soda](https://unsplash.com/@jarritos).

적절한 지시문을 사용하면 이미지 캡셔닝에서 시각적 질의 응답으로 모델을 유도할 수 있습니다:

```py
>>> prompt = [
...     "Instruction: Provide an answer to the question. Use the image to answer.\n",
...     "https://images.unsplash.com/photo-1623944889288-cd147dbb517c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",
...     "Question: Where are these people and what's the weather like? Answer:"
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=20, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
Instruction: Provide an answer to the question. Use the image to answer.
 Question: Where are these people and what's the weather like? Answer: They're in a park in New York City, and it's a beautiful day.
```

## 이미지 분류[[image-classification]]

IDEFICS는 특정 카테고리의 라벨이 포함된 데이터로 명시적으로 학습되지 않아도 이미지를 다양한 카테고리로 분류할 수 있습니다. 카테고리 목록이 주어지면, 모델은 이미지와 텍스트 이해 능력을 사용하여 이미지가 속할 가능성이 높은 카테고리를 추론할 수 있습니다.

여기에 야채 가판대 이미지가 있습니다.

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-classification.jpg" alt="Image of a vegetable stand"/>
</div>

사진 제공: [Peter Wendt](https://unsplash.com/@peterwendt).

우리는 모델에게 우리가 가진 카테고리 중 하나로 이미지를 분류하도록 지시할 수 있습니다:

```py
>>> categories = ['animals','vegetables', 'city landscape', 'cars', 'office']
>>> prompt = [f"Instruction: Classify the following image into a single category from the following list: {categories}.\n",
...     "https://images.unsplash.com/photo-1471193945509-9ad0617afabf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",    
...     "Category: "
... ]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=6, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0])
Instruction: Classify the following image into a single category from the following list: ['animals', 'vegetables', 'city landscape', 'cars', 'office'].
Category: Vegetables
```  

위 예제에서는 모델에게 이미지를 단일 카테고리로 분류하도록 지시했지만, 순위 분류를 하도록 모델에 프롬프트를 제공할 수도 있습니다.

## 이미지 기반 텍스트 생성[[image-guided-text-generation]]

이미지를 활용한 텍스트 생성 기술을 사용하면 더욱 창의적인 작업이 가능합니다. 이 기술은 이미지를 바탕으로 텍스트를 만들어내며, 제품 설명, 광고 문구, 장면 묘사 등 다양한 용도로 활용할 수 있습니다.

간단한 예로, 빨간 문 이미지를 IDEFICS에 입력하여 이야기를 만들어보겠습니다:

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-story-generation.jpg" alt="Image of a red door with a pumpkin on the steps"/>
</div>

사진 제공: [Craig Tidball](https://unsplash.com/@devonshiremedia).
  
```py
>>> prompt = ["Instruction: Use the image to write a story. \n",
...     "https://images.unsplash.com/photo-1517086822157-2b0358e7684a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2203&q=80",
...     "Story: \n"]

>>> inputs = processor(prompt, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, num_beams=2, max_new_tokens=200, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> print(generated_text[0]) 
Instruction: Use the image to write a story. 
 Story: 
Once upon a time, there was a little girl who lived in a house with a red door.  She loved her red door.  It was the prettiest door in the whole world.

One day, the little girl was playing in her yard when she noticed a man standing on her doorstep.  He was wearing a long black coat and a top hat.

The little girl ran inside and told her mother about the man.

Her mother said, “Don’t worry, honey.  He’s just a friendly ghost.”

The little girl wasn’t sure if she believed her mother, but she went outside anyway.

When she got to the door, the man was gone.

The next day, the little girl was playing in her yard again when she noticed the man standing on her doorstep.

He was wearing a long black coat and a top hat.

The little girl ran
```

IDEFICS가 문 앞에 있는 호박을 보고 유령에 대한 으스스한 할로윈 이야기를 만든 것 같습니다.

<Tip>

이처럼 긴 텍스트를 생성할 때는 텍스트 생성 전략을 조정하는 것이 좋습니다. 이렇게 하면 생성된 결과물의 품질을 크게 향상시킬 수 있습니다. 자세한 내용은 [텍스트 생성 전략](../generation_strategies)을 참조하세요.
</Tip>

## 배치 모드에서 추론 실행[[running-inference-in-batch-mode]]

앞선 모든 섹션에서는 단일 예시에 대해 IDEFICS를 설명했습니다. 이와 매우 유사한 방식으로, 프롬프트 목록을 전달하여 여러 예시에 대한 추론을 실행할 수 있습니다:

```py
>>> prompts = [
...     [   "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3501&q=80",
...         "This is an image of ",
...     ],
...     [   "https://images.unsplash.com/photo-1623944889288-cd147dbb517c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",
...         "This is an image of ",
...     ],
...     [   "https://images.unsplash.com/photo-1471193945509-9ad0617afabf?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=3540&q=80",
...         "This is an image of ",
...     ],
... ]

>>> inputs = processor(prompts, return_tensors="pt").to("cuda")
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, max_new_tokens=10, bad_words_ids=bad_words_ids)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> for i,t in enumerate(generated_text):
...     print(f"{i}:\n{t}\n") 
0:
This is an image of the Eiffel Tower in Paris, France.

1:
This is an image of a couple on a picnic blanket.

2:
This is an image of a vegetable stand.
```

## 대화형 사용을 위한 IDEFICS 인스트럭트 실행[[idefics-instruct-for-conversational-use]]

대화형 사용 사례를 위해, 🤗 Hub에서 명령어 수행에 최적화된 버전의 모델을 찾을 수 있습니다. 이곳에는 `HuggingFaceM4/idefics-80b-instruct`와 `HuggingFaceM4/idefics-9b-instruct`가 있습니다.

이 체크포인트는 지도 학습 및 명령어 미세 조정 데이터셋의 혼합으로 각각의 기본 모델을 미세 조정한 결과입니다. 이를 통해 모델의 하위 작업 성능을 향상시키는 동시에 대화형 환경에서 모델을 더 사용하기 쉽게 합니다.

대화형 사용을 위한 사용법 및 프롬프트는 기본 모델을 사용하는 것과 매우 유사합니다.

```py
>>> import torch
>>> from transformers import IdeficsForVisionText2Text, AutoProcessor

>>> device = "cuda" if torch.cuda.is_available() else "cpu"

>>> checkpoint = "HuggingFaceM4/idefics-9b-instruct"
>>> model = IdeficsForVisionText2Text.from_pretrained(checkpoint, dtype=torch.bfloat16).to(device)
>>> processor = AutoProcessor.from_pretrained(checkpoint)

>>> prompts = [
...     [
...         "User: What is in this image?",
...         "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
...         "<end_of_utterance>",

...         "\nAssistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground.<end_of_utterance>",

...         "\nUser:",
...         "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
...         "And who is that?<end_of_utterance>",

...         "\nAssistant:",
...     ],
... ]

>>> # --batched mode
>>> inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
>>> # --single sample mode
>>> # inputs = processor(prompts[0], return_tensors="pt").to(device)

>>> # args 생성
>>> exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
>>> bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

>>> generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> for i, t in enumerate(generated_text):
...     print(f"{i}:\n{t}\n")
```
