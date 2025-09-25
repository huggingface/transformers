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

# Text generation strategies[[text-generation-strategies]]

텍스트 생성은 개방형 텍스트 작성, 요약, 번역 등 다양한 자연어 처리(NLP) 작업에 필수적입니다. 이는 또한 음성-텍스트 변환, 시각-텍스트 변환과 같이 텍스트를 출력으로 하는 여러 혼합 모달리티 응용 프로그램에서도 중요한 역할을 합니다. 텍스트 생성을 가능하게 하는 몇몇 모델로는 GPT2, XLNet, OpenAI GPT, CTRL, TransformerXL, XLM, Bart, T5, GIT, Whisper 등이 있습니다.


[`~generation.GenerationMixin.generate`] 메서드를 활용하여 다음과 같은 다양한 작업들에 대해 텍스트 결과물을 생성하는 몇 가지 예시를 살펴보세요:
* [텍스트 요약](./tasks/summarization#inference)
* [이미지 캡셔닝](./model_doc/git#transformers.GitForCausalLM.forward.example)
* [오디오 전사](./model_doc/whisper#transformers.WhisperForConditionalGeneration.forward.example)

generate 메소드에 입력되는 값들은 모델의 데이터 형태에 따라 달라집니다. 이 값들은 AutoTokenizer나 AutoProcessor와 같은 모델의 전처리 클래스에 의해 반환됩니다. 모델의 전처리 장치가 하나 이상의 입력 유형을 생성하는 경우, 모든 입력을 generate()에 전달해야 합니다. 각 모델의 전처리 장치에 대해서는 해당 모델의 문서에서 자세히 알아볼 수 있습니다.

텍스트를 생성하기 위해 출력 토큰을 선택하는 과정을 디코딩이라고 하며, `generate()` 메소드가 사용할 디코딩 전략을 사용자가 커스터마이징할 수 있습니다. 디코딩 전략을 수정하는 것은 훈련 가능한 매개변수의 값들을 변경하지 않지만, 생성된 출력의 품질에 눈에 띄는 영향을 줄 수 있습니다. 이는 텍스트에서 반복을 줄이고, 더 일관성 있게 만드는 데 도움을 줄 수 있습니다.


이 가이드에서는 다음과 같은 내용을 다룹니다:
* 기본 생성 설정
* 일반적인 디코딩 전략과 주요 파라미터
* 🤗 Hub에서 미세 조정된 모델과 함께 사용자 정의 생성 설정을 저장하고 공유하는 방법

## 기본 텍스트 생성 설정[[default-text-generation-configuration]]

모델의 디코딩 전략은 생성 설정에서 정의됩니다. 사전 훈련된 모델을 [`pipeline`] 내에서 추론에 사용할 때, 모델은 내부적으로 기본 생성 설정을 적용하는 `PreTrainedModel.generate()` 메소드를 호출합니다. 사용자가 모델과 함께 사용자 정의 설정을 저장하지 않았을 경우에도 기본 설정이 사용됩니다.

모델을 명시적으로 로드할 때, `model.generation_config`을 통해 제공되는 생성 설정을 검사할 수 있습니다.

```python
>>> from transformers import AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
>>> model.generation_config
GenerationConfig {
    "bos_token_id": 50256,
    "eos_token_id": 50256,
}
```

 `model.generation_config`를 출력하면 기본 설정과 다른 값들만 표시되고, 기본값들은 나열되지 않습니다.

기본 생성 설정은 입력 프롬프트와 출력을 합친 최대 크기를 20 토큰으로 제한하여 리소스 부족을 방지합니다. 기본 디코딩 전략은 탐욕 탐색(greedy search)으로, 다음 토큰으로 가장 높은 확률을 가진 토큰을 선택하는 가장 단순한 디코딩 전략입니다. 많은 작업과 작은 출력 크기에 대해서는 이 방법이 잘 작동하지만, 더 긴 출력을 생성할 때 사용하면 매우 반복적인 결과를 생성하게 될 수 있습니다.

## 텍스트 생성 사용자 정의[[customize-text-generation]]

파라미터와 해당 값을 [`generate`] 메소드에 직접 전달하여 `generation_config`을 재정의할 수 있습니다:

```python
>>> my_model.generate(**inputs, num_beams=4, do_sample=True)  # doctest: +SKIP
```

기본 디코딩 전략이 대부분의 작업에 잘 작동한다 하더라도, 조정할 수 있는 몇 가지 파라미터가 있습니다. 일반적으로 조정되는 파라미터에는 다음과 같은 것들이 포함됩니다:

- `max_new_tokens`: 생성할 최대 토큰 수입니다. 즉, 프롬프트에 있는 토큰을 제외한 출력 시퀀스의 크기입니다. 출력의 길이를 중단 기준으로 사용하는 대신, 전체 생성물이 일정 시간을 초과할 때 생성을 중단하기로 선택할 수도 있습니다. 더 알아보려면 [`StoppingCriteria`]를 확인하세요.
- `num_beams`: 1보다 큰 수의 빔을 지정함으로써, 탐욕 탐색(greedy search)에서 빔 탐색(beam search)으로 전환하게 됩니다. 이 전략은 각 시간 단계에서 여러 가설을 평가하고 결국 전체 시퀀스에 대해 가장 높은 확률을 가진 가설을 선택합니다. 이는 초기 토큰의 확률이 낮아 탐욕 탐색에 의해 무시되었을 높은 확률의 시퀀스를 식별할 수 있는 장점을 가집니다.
- `do_sample`: 이 매개변수를 `True`로 설정하면, 다항 샘플링, 빔 탐색 다항 샘플링, Top-K 샘플링 및 Top-p 샘플링과 같은 디코딩 전략을 활성화합니다. 이러한 전략들은 전체 어휘에 대한 확률 분포에서 다음 토큰을 선택하며, 전략별로 특정 조정이 적용됩니다.
- `num_return_sequences`: 각 입력에 대해 반환할 시퀀스 후보의 수입니다. 이 옵션은 빔 탐색(beam search)의 변형과 샘플링과 같이 여러 시퀀스 후보를 지원하는 디코딩 전략에만 사용할 수 있습니다. 탐욕 탐색(greedy search) 같은 디코딩 전략은 단일 출력 시퀀스를 반환합니다.

## 모델에 사용자 정의 디코딩 전략 저장[[save-a-custom-decoding-strategy-with-your-model]]

특정 생성 설정을 가진 미세 조정된 모델을 공유하고자 할 때, 다음 단계를 따를 수 있습니다:
* [`GenerationConfig`] 클래스 인스턴스를 생성합니다.
* 디코딩 전략 파라미터를 설정합니다.
* 생성 설정을 [`GenerationConfig.save_pretrained`]를 사용하여 저장하며, `config_file_name` 인자는 비워둡니다.
* 모델의 저장소에 설정을 업로드하기 위해 `push_to_hub`를 `True`로 설정합니다.

```python
>>> from transformers import AutoModelForCausalLM, GenerationConfig

>>> model = AutoModelForCausalLM.from_pretrained("my_account/my_model")  # doctest: +SKIP
>>> generation_config = GenerationConfig(
...     max_new_tokens=50, do_sample=True, top_k=50, eos_token_id=model.config.eos_token_id
... )
>>> generation_config.save_pretrained("my_account/my_model", push_to_hub=True)  # doctest: +SKIP
```

단일 디렉토리에 여러 생성 설정을 저장할 수 있으며, 이때 [`GenerationConfig.save_pretrained`]의 `config_file_name` 인자를 사용합니다. 나중에 [`GenerationConfig.from_pretrained`]로 이들을 인스턴스화할 수 있습니다. 이는 단일 모델에 대해 여러 생성 설정을 저장하고 싶을 때 유용합니다(예: 샘플링을 이용한 창의적 텍스트 생성을 위한 하나, 빔 탐색을 이용한 요약을 위한 다른 하나 등). 모델에 설정 파일을 추가하기 위해 적절한 Hub 권한을 가지고 있어야 합니다.

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

>>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

>>> translation_generation_config = GenerationConfig(
...     num_beams=4,
...     early_stopping=True,
...     decoder_start_token_id=0,
...     eos_token_id=model.config.eos_token_id,
...     pad_token=model.config.pad_token_id,
... )

>>> # 팁: Hub에 push하려면 `push_to_hub=True`를 추가
>>> translation_generation_config.save_pretrained("/tmp", "translation_generation_config.json")

>>> # 명명된 생성 설정 파일을 사용하여 생성을 매개변수화할 수 있습니다.
>>> generation_config = GenerationConfig.from_pretrained("/tmp", "translation_generation_config.json")
>>> inputs = tokenizer("translate English to French: Configuration files are easy to use!", return_tensors="pt")
>>> outputs = model.generate(**inputs, generation_config=generation_config)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['Les fichiers de configuration sont faciles à utiliser!']
```

## 스트리밍[[streaming]]

`generate()` 메소드는 `streamer` 입력을 통해 스트리밍을 지원합니다. `streamer` 입력은 `put()`과 `end()` 메소드를 가진 클래스의 인스턴스와 호환됩니다. 내부적으로, `put()`은 새 토큰을 추가하는 데 사용되며, `end()`는 텍스트 생성의 끝을 표시하는 데 사용됩니다.

> [!WARNING]
> 스트리머 클래스의 API는 아직 개발 중이며, 향후 변경될 수 있습니다.

실제로 다양한 목적을 위해 자체 스트리밍 클래스를 만들 수 있습니다! 또한, 기본적인 스트리밍 클래스들도 준비되어 있어 바로 사용할 수 있습니다. 예를 들어, [`TextStreamer`] 클래스를 사용하여 `generate()`의 출력을 화면에 한 단어씩 스트리밍할 수 있습니다:

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

>>> tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
>>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
>>> streamer = TextStreamer(tok)

>>> # 스트리머는 평소와 같은 출력값을 반환할 뿐만 아니라 생성된 텍스트도 표준 출력(stdout)으로 출력합니다.
>>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
```

## 디코딩 전략[[decoding-strategies]]

`generate()` 매개변수와 궁극적으로 `generation_config`의 특정 조합을 사용하여 특정 디코딩 전략을 활성화할 수 있습니다. 이 개념이 처음이라면, 흔히 사용되는 디코딩 전략이 어떻게 작동하는지 설명하는 [이 블로그 포스트](https://huggingface.co/blog/how-to-generate)를 읽어보는 것을 추천합니다.

여기서는 디코딩 전략을 제어하는 몇 가지 매개변수를 보여주고, 이를 어떻게 사용할 수 있는지 설명하겠습니다.

### 탐욕 탐색(Greedy Search)[[greedy-search]]

[`generate`]는 기본적으로 탐욕 탐색 디코딩을 사용하므로 이를 활성화하기 위해 별도의 매개변수를 지정할 필요가 없습니다. 이는 `num_beams`가 1로 설정되고 `do_sample=False`로 되어 있다는 의미입니다."

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> prompt = "I look forward to"
>>> checkpoint = "distilbert/distilgpt2"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> outputs = model.generate(**inputs)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['I look forward to seeing you all again!\n\n\n\n\n\n\n\n\n\n\n']
```

### 다항 샘플링(Multinomial sampling)[[multinomial-sampling]]

탐욕 탐색(greedy search)이 항상 가장 높은 확률을 가진 토큰을 다음 토큰으로 선택하는 것과 달리, 다항 샘플링(multinomial sampling, 조상 샘플링(ancestral sampling)이라고도 함)은 모델이 제공하는 전체 어휘에 대한 확률 분포를 기반으로 다음 토큰을 무작위로 선택합니다. 0이 아닌 확률을 가진 모든 토큰은 선택될 기회가 있으므로, 반복의 위험을 줄일 수 있습니다.

다항 샘플링을 활성화하려면 `do_sample=True` 및 `num_beams=1`을 설정하세요.

```python
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
>>> set_seed(0)  # 재현성을 위해

>>> checkpoint = "openai-community/gpt2-large"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)

>>> prompt = "Today was an amazing day because"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> outputs = model.generate(**inputs, do_sample=True, num_beams=1, max_new_tokens=100)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Today was an amazing day because when you go to the World Cup and you don\'t, or when you don\'t get invited,
that\'s a terrible feeling."']
```

### 빔 탐색(Beam-search) 디코딩[[beam-search-decoding]]

탐욕 검색(greedy search)과 달리, 빔 탐색(beam search) 디코딩은 각 시간 단계에서 여러 가설을 유지하고 결국 전체 시퀀스에 대해 가장 높은 확률을 가진 가설을 선택합니다. 이는 낮은 확률의 초기 토큰으로 시작하고 그리디 검색에서 무시되었을 가능성이 높은 시퀀스를 식별하는 이점이 있습니다.

이 디코딩 전략을 활성화하려면 `num_beams` (추적할 가설 수라고도 함)를 1보다 크게 지정하세요.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> prompt = "It is astonishing how one can"
>>> checkpoint = "openai-community/gpt2-medium"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)

>>> outputs = model.generate(**inputs, num_beams=5, max_new_tokens=50)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['It is astonishing how one can have such a profound impact on the lives of so many people in such a short period of
time."\n\nHe added: "I am very proud of the work I have been able to do in the last few years.\n\n"I have']
```

### 빔 탐색 다항 샘플링(Beam-search multinomial sampling)[[beam-search-multinomial-sampling]]

이 디코딩 전략은 이름에서 알 수 있듯이 빔 탐색과 다항 샘플링을 결합한 것입니다. 이 디코딩 전략을 사용하기 위해서는 `num_beams`를 1보다 큰 값으로 설정하고, `do_sample=True`로 설정해야 합니다.

```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
>>> set_seed(0)  # 재현성을 위해

>>> prompt = "translate English to German: The house is wonderful."
>>> checkpoint = "google-t5/t5-small"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

>>> outputs = model.generate(**inputs, num_beams=5, do_sample=True)
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'Das Haus ist wunderbar.'
```

### 추론 디코딩(Speculative Decoding)[[speculative-decoding]]

추론 디코딩(보조 디코딩(assisted decoding)으로도 알려짐)은 동일한 토크나이저를 사용하는 훨씬 작은 보조 모델을 활용하여 몇 가지 후보 토큰을 생성하는 상위 모델의 디코딩 전략을 수정한 것입니다. 주 모델은 단일 전방 통과로 후보 토큰을 검증함으로써 디코딩 과정을 가속화합니다. `do_sample=True`일 경우, [추론 디코딩 논문](https://huggingface.co/papers/2211.17192)에 소개된 토큰 검증과 재샘플링 방식이 사용됩니다.

현재, 탐욕 검색(greedy search)과 샘플링만이 지원되는 보조 디코딩(assisted decoding) 기능을 통해, 보조 디코딩은 배치 입력을 지원하지 않습니다. 보조 디코딩에 대해 더 알고 싶다면, [이 블로그 포스트](https://huggingface.co/blog/assisted-generation)를 확인해 주세요.

보조 디코딩을 활성화하려면 모델과 함께 `assistant_model` 인수를 설정하세요.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> prompt = "Alice and Bob"
>>> checkpoint = "EleutherAI/pythia-1.4b-deduped"
>>> assistant_checkpoint = "EleutherAI/pythia-160m-deduped"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint)
>>> outputs = model.generate(**inputs, assistant_model=assistant_model)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Alice and Bob are sitting in a bar. Alice is drinking a beer and Bob is drinking a']
```

샘플링 방법과 함께 보조 디코딩을 사용하는 경우 다항 샘플링과 마찬가지로 `temperature` 인수를 사용하여 무작위성을 제어할 수 있습니다. 그러나 보조 디코딩에서는 `temperature`를 낮추면 대기 시간을 개선하는 데 도움이 될 수 있습니다.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
>>> set_seed(42)  # 재현성을 위해

>>> prompt = "Alice and Bob"
>>> checkpoint = "EleutherAI/pythia-1.4b-deduped"
>>> assistant_checkpoint = "EleutherAI/pythia-160m-deduped"

>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint)
>>> outputs = model.generate(**inputs, assistant_model=assistant_model, do_sample=True, temperature=0.5)
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['Alice and Bob, who were both in their early twenties, were both in the process of']
```
