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


# LLM을 사용한 생성 [[generation-with-llms]]

[[open-in-colab]]

LLM 또는 대형 언어 모델은 텍스트 생성의 핵심 구성 요소입니다. 간단히 말하면, 이것은 주어진 입력 텍스트에 대한 다음 단어(또는 정확하게는 토큰)를 예측하기 위해 훈련된 대형 사전 훈련 변환기 모델로 구성됩니다. 토큰을 한 번에 하나씩 예측하기 때문에 모델을 호출하는 것 외에 새로운 문장을 생성하기 위해 더 복잡한 작업을 수행해야 합니다. -- 자동 회귀 생성을 수행해야 합니다.

자동 회귀 생성은 초기 입력을 주고 모델을 반복적으로 호출하는 추론 시간 절차입니다. 🤗 Transformers에서 이것은 모든 생성 능력이 있는 모델에 사용 가능한 [`~generation.GenerationMixin.generate`] 메서드에 의해 처리됩니다.

이 튜토리얼에서는 다음을 보여줍니다:

* LLM으로 텍스트 생성하기
* 흔한 함정 피하기
* LLM에서 최대한 활용하기 위한 다음 단계

시작하기 전에 필요한 모든 라이브러리가 설치되어 있는지 확인하세요.

```bash
pip install transformers bitsandbytes>=0.39.0 -q
```


## 텍스트 생성 [[generate-text]]

[인과 언어 모델링](tasks/language_modeling)을 위해 훈련된 언어 모델은 입력으로 텍스트 토큰의 시퀀스를 받아들이고 다음 토큰에 대한 확률 분포를 반환합니다.

<!-- [GIF 1 -- FWD PASS] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_1_1080p.mov"
    ></video>
    <figcaption>"LLM의 전방 패스"</figcaption>
</figure>

LLM과 함께 자동 회귀 생성의 중요한 측면은 이 확률 분포에서 다음 토큰을 어떻게 선택할 것인지입니다. 다음 반복을 위해 토큰을 얻는 한 이 단계에서는 무엇이든 가능합니다. 이것은 확률 분포에서 가장 가능성이 높은 토큰을 선택하는 것처럼 간단할 수도 있고, 결과 분포에서 샘플링하기 전에 수십 가지 변환을 적용하는 것처럼 복잡할 수도 있습니다.

<!-- [GIF 2 -- TEXT GENERATION] -->
<figure class="image table text-center m-0 w-full">
    <video
        style="max-width: 90%; margin: auto;"
        autoplay loop muted playsinline
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_2_1080p.mov"
    ></video>
    <figcaption>"자동 회귀 생성은 확률 분포에서 다음 토큰을 반복적으로 선택하여 텍스트를 생성합니다."</figcaption>
</figure>

위에 표시된 프로세스는 일부 정지 조건에 도달할 때까지 반복적으로 반복됩니다. 이상적으로는 정지 조건이 모델에 의해 지정되어야 하며, 이는 시퀀스 종료 (`EOS`) 토큰을 출력할 때 언제인지 학습해야 합니다. 이 경우가 아니라면, 생성은 미리 정의된 최대 길이에 도달할 때 중지됩니다.

토큰 선택 단계와 정지 조건을 올바르게 설정하는 것은 작업에서 모델이 예상대로 작동하게 만드는 데 필수적입니다. 그래서 우리는 각 모델과 함께 로드되는 좋은 기본 생성 매개 변수화를 포함하는 [`~generation.GenerationConfig`] 파일을 가지고 있습니다.

코드에 대해 이야기해봅시다!

<Tip>

기본 LLM 사용에 관심이 있다면, 우리의 고수준 [`Pipeline`](pipeline_tutorial) 인터페이스는 좋은 시작점입니다. 그러나 LLM은 종종 양자화와 토큰 선택 단계의 세밀한 제어와 같은 고급 기능을 필요로 합니다. 이는 [`~generation.GenerationMixin.generate`]를 통해 가장 잘 수행됩니다. LLM과 함께 자동 회귀 생성은 또한 자원 집약적이므로 적절한 처리량을 위해 GPU에서 실행해야 합니다.

</Tip>

<!-- TODO: update example to llama 2 (or a newer popular baseline) when it becomes ungated -->
먼저 모델을 로드해야 합니다.

```py
>>> from transformers import AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained(
...     "openlm-research/open_llama_7b", device_map="auto", load_in_4bit=True
... )
```

`from_pretrained` 호출에서 두 플래그를 주목하십시오:

- `device_map`은 모델이 GPU로 이동되도록 합니다.
- `load_in_4bit`는 리소스 요구 사항을 크게 줄이기 위해 [4비트 동적 양자화](main_classes/quantization)를 적용합니다.

모델을 초기화하는 다른 방법이 있지만, LLM으로 시작하기에 좋은 기준선입니다.

다음으로, [토크나이저](tokenizer_summary)로 텍스트 입력을 전처리해야 합니다.

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b")
>>> model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")
```

`model_inputs` 변수는 토크나이즈된 텍스트 입력과 주의 마스크를 보유하고 있습니다. [`~generation.GenerationMixin.generate`]는 주의 마스크가 전달되지 않을 때 최선을 다해 주의 마스크를 추론하려고 하지만, 최적의 결과를 위해 가능한 경우에는 항상 전달하는 것이 좋습니다.

마지막으로, [`~generation.GenerationMixin.generate`] 메서드를 호출하여 생성된 토큰을 반환하고, 출력하기 전에 텍스트로 변환해야 합니다.

```py
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A list of colors: red, blue, green, yellow, black, white, and brown'
```

그게 다에요! 몇 줄의 코드로 LLM의 힘을 활용할 수 있습니다.


## 흔한 함정 [[common-pitfalls]]

[생성 전략](generation_strategies)이 많고, 때로는 기본 값이 사용 사례에 적합하지 않을 수 있습니다. 출력이 예상대로 정렬되지 않는 경우, 가장 흔한 함정과 이를 피하는 방법에 대한 목록을 만들었습니다.

```py
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b")
>>> tokenizer.pad_token = tokenizer.eos_token  # Llama has no pad token by default
>>> model = AutoModelForCausalLM.from_pretrained(
...     "openlm-research/open_llama_7b", device_map="auto", load_in_4bit=True
... )
```

### 생성된 출력이 너무 짧거나 길다 [[generated-output-is-too-shortlong]]

[`~generation.GenerationConfig`] 파일에서 지정되지 않은 경우, `generate`는 기본적으로 최대 20개의 토큰을 반환합니다. `generate` 호출에서 `max_new_tokens`을 수동으로 설정하여 반환할 수 있는 새 토큰의 최대 수를 제어하는 것이 좋습니다. LLM(정확하게는 [디코더 전용 모델](https://huggingface.co/learn/nlp-course/chapter1/6?fw=pt))은 출력의 일부로 입력 프롬프트도 반환합니다.


```py
>>> model_inputs = tokenizer(["A sequence of numbers: 1, 2"], return_tensors="pt").to("cuda")

>>> # By default, the output will contain up to 20 tokens
>>> generated_ids = model.generate(**model_inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5'

>>> # Setting `max_new_tokens` allows you to control the maximum length
>>> generated_ids = model.generate(**model_inputs, max_new_tokens=50)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,'
```

### 잘못된 생성 모드 [[incorrect-generation-mode]]

기본적으로, [`~generation.GenerationConfig`] 파일에서 지정되지 않은 경우, `generate`는 각 반복에서 가장 가능성이 높은 토큰을 선택합니다(탐욕스러운 디코딩). 작업에 따라 이것은 바람직하지 않을 수 있습니다. 챗봇이나 에세이 작성과 같은 창의적인 작업은 샘플링에서 이익을 얻습니다. 반면, 오디오 전사나 번역과 같은 입력 기반 작업은 탐욕스러운 디코딩에서 이익을 얻습니다. `do_sample=True`로 샘플링을 활성화하고, 이 주제에 대해 더 알아보려면 이 [블로그 게시물](https://huggingface.co/blog/how-to-generate)을 참조하세요.

```py
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

### 잘못된 패딩 측면 [[wrong-padding-side]]

LLM은 [디코더 전용](https://huggingface.co/learn/nlp-course/chapter1/6?fw=pt) 아키텍처로, 입력 프롬프트에서 계속 반복합니다. 입력이 동일한 길이를 갖지 않는 경우 패딩이 필요합니다. LLM은 패드 토큰에서 계속되도록 훈련되지 않았으므로 입력은 왼쪽 패딩되어야 합니다. 또한 생성에 주의 마스크를 전달하는 것을 잊지 않도록 주의하세요!

```py
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

자동 회귀 생성 프로세스는 상대적으로 간단하지만, LLM에서 최대한 활용하는 것은 많은 움직이는 부분이 있기 때문에 도전적인 과제일 수 있습니다. LLM 사용과 이해에 더 깊게 들어가기 위한 다음 단계:

<!-- TODO: complete with new guides -->
### 고급 생성 사용 [[advanced-generate-usage]]

1. 다른 생성 방법을 제어하는 방법, 생성 구성 파일을 설정하는 방법, 출력을 스트리밍하는 방법에 대한 [가이드](generation_strategies);
2. [`~generation.GenerationConfig`], [`~generation.GenerationMixin.generate`], [generate-related classes](internal/generation_utils)에 대한 API 참조.

### LLM 리더보드 [[llm-leaderboards]]

1. 오픈 소스 모델의 품질에 중점을 둔 [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard);
2. LLM 처리량에 중점을 둔 [Open LLM-Perf Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard).

### 지연 시간 및 처리량 [[latency-and-throughput]]

1. 동적 양자화에 대한 [가이드](main_classes/quantization), 이를 통해 메모리 요구 사항을 크게 줄일 수 있습니다.

### 관련 라이브러리 [[related-libraries]]

1. LLM을 위한 생산 준비 서버인 [`text-generation-inference`](https://github.com/huggingface/text-generation-inference);
2. 특정 하드웨어 장치를 위해 최적화하는 🤗 Transformers의 확장인 [`optimum`](https://github.com/huggingface/optimum).
