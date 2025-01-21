<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# GPT-NeoX-Japanese [[gpt-neox-japanese]]

## 개요 [[overview]]


일본어를 위한 자동회귀 언어 모델인 GPT-NeoX-Japanese를 소개합니다. 이 모델은 [https://github.com/EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox)에서 학습되었습니다. 일본어는 많은 어휘와 히라가나, 가타카나, 한자의 조합으로 이루어진 독특한 언어입니다. 이러한 일본어의 독특한 구조를 해결하기 위해 [특수 서브워드 토크나이저](https://github.com/tanreinama/Japanese-BPEEncoder_V2)를 사용했습니다. 이 유용한 토크나이저를 오픈소스로 제공해 준 *tanreinama*에게 매우 감사드립니다.

이 모델은 Google의 [PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html) 연구 권장 사항을 따르며, 트랜스포머 블록에서 편향 파라미터를 제거하여 모델 성능을 향상시켰습니다. 자세한 내용은 [이 기사](https://medium.com/ml-abeja/training-a-better-gpt-2-93b157662ae4)를 참조하세요.

모델 개발은 [ABEJA, Inc.](https://www.abejainc.com/)의 [신야 오타니](https://github.com/SO0529), [타카요시 마카베](https://github.com/spider-man-tm), [안주 아로라](https://github.com/Anuj040), [쿄 하토리](https://github.com/go5paopao)에 의해 주도되었습니다. 이 모델 개발 활동에 대한 자세한 내용은 [여기](https://tech-blog.abeja.asia/entry/abeja-gpt-project-202207)를 참조하세요.



### 사용 예시 [[usage-example]]

`generate()` 메서드를 사용하면 GPT NeoX Japanese 모델을 통해 텍스트를 생성할 수 있습니다.

```python
>>> from transformers import GPTNeoXJapaneseForCausalLM, GPTNeoXJapaneseTokenizer

>>> model = GPTNeoXJapaneseForCausalLM.from_pretrained("abeja/gpt-neox-japanese-2.7b")
>>> tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")

>>> prompt = "人とAIが協調するためには、"

>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

>>> gen_tokens = model.generate(
...     input_ids,
...     do_sample=True,
...     temperature=0.9,
...     max_length=100,
... )
>>> gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

>>> print(gen_text)
人とAIが協調するためには、AIと人が共存し、AIを正しく理解する必要があります。
```

## 자료 [[resources]]

- [일상 언어 모델링 작업 가이드 ](../tasks/language_modeling)

## GPTNeoXJapanese 설정 (GPTNeoXJapaneseConfig) [[transformers.GPTNeoXJapaneseConfig]]

[[autodoc]] GPTNeoXJapaneseConfig

## GPTNeoXJapanese토큰화 (GPTNeoXJapaneseTokenizer) [[transformers.GPTNeoXJapaneseTokenizer]]

[[autodoc]] GPTNeoXJapaneseTokenizer

## GPTNeoXJapaneseModel [[transformers.GPTNeoXJapaneseModel]]

[[autodoc]] GPTNeoXJapaneseModel
    - forward

## 일상 LLM 을 위한 GPTNeoXJapanese(GPTNeoXJapaneseForCausalLM) [[transformers.GPTNeoXJapaneseForCausalLM]]

[[autodoc]] GPTNeoXJapaneseForCausalLM
    - forward
