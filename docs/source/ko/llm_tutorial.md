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


# 대규모 언어 모델로 생성하기 [[generation-with-llms]]

[[open-in-colab]]

LLM 또는 대규모 언어 모델은 텍스트 생성의 핵심 구성 요소입니다. 간단히 말하면, 주어진 입력 텍스트에 대한 다음 단어(정확하게는 토큰)를 예측하기 위해 훈련된 대규모 사전 훈련 변환기 모델로 구성됩니다. 토큰을 한 번에 하나씩 예측하기 때문에 새로운 문장을 생성하려면 모델을 호출하는 것 외에 더 복잡한 작업을 수행해야 합니다. 즉, 자기회귀 생성을 수행해야 합니다.

자기회귀 생성은 몇 개의 초기 입력값을 제공한 후, 그 출력을 다시 모델에 입력으로 사용하여 반복적으로 호출하는 추론 과정입니다. 🤗 Transformers에서는 [`~generation.GenerationMixin.generate`] 메소드가 이 역할을 하며, 이는 생성 기능을 가진 모든 모델에서 사용 가능합니다.

이 튜토리얼에서는 다음 내용을 다루게 됩니다:

* LLM으로 텍스트 생성
* 일반적으로 발생하는 문제 해결
* LLM을 최대한 활용하기 위한 다음 단계

시작하기 전에 필요한 모든 라이브러리가 설치되어 있는지 확인하세요:

```bash
pip install transformers bitsandbytes>=0.39.0 -q
```


## 텍스트 생성 [[generate-text]]

[인과적 언어 모델링(causal language modeling)](tasks/language_modeling)을 목적으로 학습된 언어 모델은 일련의 텍스트 토큰을 입력으로 사용하고, 그 결과로 다음 토큰이 나올 확률 분포를 제공합니다.

<!-- [GIF 1 -- FWD PASS] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_1_1080p.mov"
    ></video>
    <figcaption>"LLM의 전방 패스"</figcaption>
</figure>

LLM과 자기회귀 생성을 함께 사용할 때 핵심적인 부분은 이 확률 분포로부터 다음 토큰을 어떻게 고를 것인지입니다. 다음 반복 과정에 사용될 토큰을 결정하는 한, 어떠한 방법도 가능합니다. 확률 분포에서 가장 가능성이 높은 토큰을 선택하는 것처럼 간단할 수도 있고, 결과 분포에서 샘플링하기 전에 수십 가지 변환을 적용하는 것처럼 복잡할 수도 있습니다.

<!-- [GIF 2 -- TEXT GENERATION] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_2_1080p.mov"
    ></video>
    <figcaption>"자기회귀 생성은 확률 분포에서 다음 토큰을 반복적으로 선택하여 텍스트를 생성합니다."</figcaption>
</figure>

위에서 설명한 과정은 어떤 종료 조건이 충족될 때까지 반복적으로 수행됩니다. 모델이 시퀀스의 끝(EOS 토큰)을 출력할 때까지를 종료 조건으로 하는 것이 이상적입니다. 그렇지 않은 경우에는 미리 정의된 최대 길이에 도달했을 때 생성이 중단됩니다.

모델이 예상대로 동작하기 위해선 토큰 선택 단계와 정지 조건을 올바르게 설정하는 것이 중요합니다. 이러한 이유로, 각 모델에는 기본 생성 설정이 잘 정의된 [`~generation.GenerationConfig`] 파일이 함께 제공됩니다.

코드를 확인해봅시다!

<Tip>

기본 LLM 사용에 관심이 있다면, 우리의 [`Pipeline`](pipeline_tutorial) 인터페이스로 시작하는 것을 추천합니다. 그러나 LLM은 양자화나 토큰 선택 단계에서의 미세한 제어와 같은 고급 기능들을 종종 필요로 합니다. 이러한 작업은 [`~generation.GenerationMixin.generate`]를 통해 가장 잘 수행될 수 있습니다. LLM을 이용한 자기회귀 생성은 자원을 많이 소모하므로, 적절한 처리량을 위해 GPU에서 실행되어야 합니다.

</Tip>

먼저, 모델을 불러오세요.

```python
>>> from transformers import AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained(
...     "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
... )
```

`from_pretrained` 함수를 호출할 때 2개의 플래그를 주목하세요:

- `device_map`은 모델이 GPU로 이동되도록 합니다.
- `load_in_4bit`는 리소스 요구 사항을 크게 줄이기 위해 [4비트 동적 양자화](main_classes/quantization)를 적용합니다.

이 외에도 모델을 초기화하는 다양한 방법이 있지만, LLM을 처음 시작할 때 이 설정을 추천합니다.

이어서 텍스트 입력을 [토크나이저](tokenizer_summary)으로 전처리하세요.

```python
>>> from transformers import AutoTokenizer
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(device)
```

`model_inputs` 변수에는 토큰화된 텍스트 입력과 함께 어텐션 마스크가 들어 있습니다. [`~generation.GenerationMixin.generate`]는 어텐션 마스크가 제공되지 않았을 경우에도 이를 추론하려고 노력하지만, 최상의 성능을 위해서는 가능하면 어텐션 마스크를 전달하는 것을 권장합니다. 

마지막으로 [`~generation.GenerationMixin.generate`] 메소드를 호출해 생성된 토큰을 얻은 후, 이를 출력하기 전에 텍스트 형태로 변환하세요.

```python
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A list of colors: red, blue, green, yellow, black, white, and brown'
```

이게 전부입니다! 몇 줄의 코드만으로 LLM의 능력을 활용할 수 있게 되었습니다.


## 일반적으로 발생하는 문제 [[common-pitfalls]]

[생성 전략](generation_strategies)이 많고, 기본값이 항상 사용 사례에 적합하지 않을 수 있습니다. 출력이 예상과 다를 때 흔히 발생하는 문제와 이를 해결하는 방법에 대한 목록을 만들었습니다.

```py
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
>>> tokenizer.pad_token = tokenizer.eos_token  # Mistral has no pad token by default
>>> model = AutoModelForCausalLM.from_pretrained(
...     "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
... )
```

### 생성된 출력이 너무 짧거나 길다 [[generated-output-is-too-shortlong]]

[`~generation.GenerationConfig`] 파일에서 별도로 지정하지 않으면, `generate`는 기본적으로 최대 20개의 토큰을 반환합니다. `generate` 호출에서 `max_new_tokens`을 수동으로 설정하여 반환할 수 있는 새 토큰의 최대 수를 설정하는 것이 좋습니다. LLM(정확하게는 [디코더 전용 모델](https://huggingface.co/learn/nlp-course/chapter1/6?fw=pt))은 입력 프롬프트도 출력의 일부로 반환합니다.


```py
>>> model_inputs = tokenizer(["A sequence of numbers: 1, 2"], return_tensors="pt").to("cuda")

>>> # By default, the output will contain up to 20 tokens
>>> generated_ids = model.generate(**model_inputs, pad_token_id=tokenizer.eos_token_id)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5'

>>> # Setting `max_new_tokens` allows you to control the maximum length
>>> generated_ids = model.generate(**model_inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=50)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,'
```

### 잘못된 생성 모드 [[incorrect-generation-mode]]

기본적으로 [`~generation.GenerationConfig`] 파일에서 별도로 지정하지 않으면, `generate`는 각 반복에서 가장 확률이 높은 토큰을 선택합니다(그리디 디코딩). 하려는 작업에 따라 이 방법은 바람직하지 않을 수 있습니다. 예를 들어, 챗봇이나 에세이 작성과 같은 창의적인 작업은 샘플링이 적합할 수 있습니다. 반면, 오디오를 텍스트로 변환하거나 번역과 같은 입력 기반 작업은 그리디 디코딩이 더 적합할 수 있습니다. `do_sample=True`로 샘플링을 활성화할 수 있으며, 이 주제에 대한 자세한 내용은 이 [블로그 포스트](https://huggingface.co/blog/how-to-generate)에서 볼 수 있습니다.

```python
>>> # Set seed or reproducibility -- you don't need this unless you want full reproducibility
>>> from transformers import set_seed
>>> set_seed(0)

>>> model_inputs = tokenizer(["I am a cat."], return_tensors="pt").to("cuda")

>>> # LLM + greedy decoding = repetitive, boring output
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'I am a cat. I am a cat. I am a cat. I am a cat'

>>> # With sampling, the output becomes more creative!
>>> generated_ids = model.generate(**model_inputs, do_sample=True)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'I am a cat.\nI just need to be. I am always.\nEvery time'
```

### 잘못된 패딩 [[wrong-padding-side]]

LLM은 [디코더 전용](https://huggingface.co/learn/nlp-course/chapter1/6?fw=pt) 구조를 가지고 있어, 입력 프롬프트에 대해 지속적으로 반복 처리를 합니다. 입력 데이터의 길이가 다르면 패딩 작업이 필요합니다. LLM은 패딩 토큰에서 작동을 이어가도록 설계되지 않았기 때문에, 입력 왼쪽에 패딩이 추가 되어야 합니다. 그리고 어텐션 마스크도 꼭 `generate` 함수에 전달되어야 합니다!

```python
>>> # The tokenizer initialized above has right-padding active by default: the 1st sequence,
>>> # which is shorter, has padding on the right side. Generation fails.
>>> model_inputs = tokenizer(
...     ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
... ).to("cuda")
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids[0], skip_special_tokens=True)[0]
''

>>> # With left-padding, it works as expected!
>>> tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b", padding_side="left")
>>> tokenizer.pad_token = tokenizer.eos_token  # Llama has no pad token by default
>>> model_inputs = tokenizer(
...     ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
... ).to("cuda")
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 3, 4, 5, 6,'
```

<!-- TODO: when the prompting guide is ready, mention the importance of setting the right prompt in this section -->

## 추가 자료 [[further-resources]]

자기회귀 생성 프로세스는 상대적으로 단순한 편이지만, LLM을 최대한 활용하려면 여러 가지 요소를 고려해야 하므로 쉽지 않을 수 있습니다. LLM에 대한 더 깊은 이해와 활용을 위한 다음 단계는 아래와 같습니다:

<!-- TODO: complete with new guides -->
### 고급 생성 사용 [[advanced-generate-usage]]

1. [가이드](generation_strategies)는 다양한 생성 방법을 제어하는 방법, 생성 설정 파일을 설정하는 방법, 출력을 스트리밍하는 방법에 대해 설명합니다.
2. [`~generation.GenerationConfig`]와 [`~generation.GenerationMixin.generate`], [generate-related classes](internal/generation_utils)를 참조해보세요.

### LLM 리더보드 [[llm-leaderboards]]

1. [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)는 오픈 소스 모델의 품질에 중점을 둡니다.
2. [Open LLM-Perf Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)는 LLM 처리량에 중점을 둡니다.

### 지연 시간 및 처리량 [[latency-and-throughput]]

1. 메모리 요구 사항을 줄이려면, 동적 양자화에 대한 [가이드](main_classes/quantization)를 참조하세요.

### 관련 라이브러리 [[related-libraries]]

1. [`text-generation-inference`](https://github.com/huggingface/text-generation-inference)는 LLM을 위한 실제 운영 환경에 적합한 서버입니다.
2. [`optimum`](https://github.com/huggingface/optimum)은 특정 하드웨어 장치에서 LLM을 최적화하기 위해 🤗 Transformers를 확장한 것입니다.
