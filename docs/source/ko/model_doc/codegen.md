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

# CodeGen[[Codegen]]

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## 개요[[Overview]]

CodeGen 모델은 Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, Caiming Xiong이 작성한 논문  [A Conversational Paradigm for Program Synthesis](https://huggingface.co/papers/2203.13474)에서 제안되었습니다.

CodeGen 모델은 프로그램 합성(program synthesis)을 위한 자기회귀(autoregressive) 언어 모델로, [The Pile](https://pile.eleuther.ai/), BigQuery, BigPython 데이터로 순차적으로 학습되었습니다.

논문의 초록은 다음과 같습니다:

*프로그램 합성(program synthesis)은 주어진 문제 명세에 대한 해답으로 프로그램을 생성하는 것을 목표로 합니다. 이 논문에서는 대규모 언어 모델(LLM)을 활용한 대화형 프로그램 합성(conversational program synthesis) 접근법을 제안하여, 기존 접근법에서의 방대한 프로그램 탐색 공간과 사용자의 의도를 명세화하는 과정에서의 어려움을 해결합니다. 제안된 방식에서는 프로그램 명세 작성과 실제 프로그램 작성을 사용자와 시스템 간 다회 대화(multi-turn conversation)로 바라봅니다. 즉, 프로그램 합성 과정 명세를 자연어로 표현하고, 기대하는 프로그램 합성을 조건부로 예측하여 생성하는 일종의 순차적 예측 문제(sequence prediction problem)로 접근했습니다. 이를 위해 자연어와 프로그래밍 언어 데이터를 기반으로 CodeGen이라는 대규모 언어 모델 그룹을 학습시켰으며, 데이터로부터 약한 지도(weak supervision)와 데이터 및 모델 규모의 확장만으로도 모델이 자연스럽게 대화 능력을 갖추게 된다는 점을 확인하였습니다. 더해서 모델의 대화형 프로그램 합성 능력을 평가하기 위해 다회 대화 기반 프로그래밍 벤치마크(MTPB)를 개발했습니다. 이 벤치마크는 각 문제를 해결하기 위해 사용자와 모델 간 여러 단계의 대화를 거쳐 프로그램이 점진적으로 합성되는 과정을 요구합니다. 연구 결과, CodeGen 모델은 대화형 능력을 성공적으로 발휘했으며 본 논문에서 제안한 대화형 합성 패러다임의 우수성과 효율성을 입증했습니다. 특히 16B 파라미터 규모로 TPU-v4에서 학습된 CodeGen 모델은 HumanEval 벤치마크에서 OpenAI의 Codex를 뛰어넘는 성능을 기록했습니다. 학습된 사용된 라이브러리인 JaxFormer와 모델 체크포인트는 오픈소스로 공개되었습니다: [이 https URL에서 확인하세요](https://github.com/salesforce/codegen).*

이 모델은[Hiroaki Hayashi](https://huggingface.co/rooa)가 기여했습니다.
모델의 원본 코드는 [여기](https://github.com/salesforce/codegen)에 있습니다.

## 체크포인트 명명 규칙[[Checkpoint Naming]]

* CodeGen 모델의 [체크포인트](https://huggingface.co/models?other=codegen)는 서로 다른 사전 학습 데이터와 다양한 크기로 제공됩니다.
* 체크포인트의 형식은 다음과 같습니다: `Salesforce/codegen-{size}-{data}`
  * `size`: `350M`, `2B`, `6B`, `16B`
  * `data`: 
    * `nl`: The Pile 데이터로 사전학습된 모델
    * `multi`: `nl` 모델에서 시작하여 다양한 프로그래밍 언어를 추가적으로 학습한 모델
    * `mono`: `multi` 모델에서 시작하여 추가로 Python 데이터에 대해 학습된 모델
* 예를 들어, `Salesforce/codegen-350M-mono`는 3억 5천만(350M) 개의 파라미터를 모델로, The Pile, 다양한 프로그래밍 언어, Python 데이터의 순서로 단계적으로 학습한 체크포인트를 의미합니다.

## 사용 예시[[Usage example]]

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> checkpoint = "Salesforce/codegen-350M-mono"
>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)

>>> text = "def hello_world():"

>>> completion = model.generate(**tokenizer(text, return_tensors="pt"))

>>> print(tokenizer.decode(completion[0]))
def hello_world():
    print("Hello World")

hello_world()
```

## 자료[[Resources]]

- [Causal language modeling task guide](../tasks/language_modeling)

## CodeGenConfig

[[autodoc]] CodeGenConfig
    - all

## CodeGenTokenizer

[[autodoc]] CodeGenTokenizer
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CodeGenTokenizerFast

[[autodoc]] CodeGenTokenizerFast

## CodeGenModel

[[autodoc]] CodeGenModel
    - forward

## CodeGenForCausalLM

[[autodoc]] CodeGenForCausalLM
    - forward
