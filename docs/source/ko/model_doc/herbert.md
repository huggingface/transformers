<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# HerBERT[[HerBERT]]

## 개요[[Overview]]

The HerBERT model was proposed in [KLEJ: Comprehensive Benchmark for Polish Language Understanding](https://www.aclweb.org/anthology/2020.acl-main.111.pdf) by Piotr Rybak, Robert Mroczkowski, Janusz Tracz, and
Ireneusz Gawlik. It is a BERT-based Language Model trained on Polish Corpora using only MLM objective with dynamic
masking of whole words.

HerBERT 모델은 Piotr Rybak, Robert Mroczkowski, Janusz Tracz, Ireneusz Gawlik이 제안한 [KLEJ: Comprehensive Benchmark for Polish Language Understanding](https://www.aclweb.org/anthology/2020.acl-main.111.pdf) 에서 소개되었습니다. HerBERT는 전체 단어를 동적으로 마스킹하는 MLM(마스킹 언어 모델) 목표만을 사용하여 폴란드어 코퍼스를 통해 학습된 BERT 기반의 언어 모델입니다.

The abstract from the paper is the following:
논문의 초록은 다음과 같습니다:

*In recent years, a series of Transformer-based models unlocked major improvements in general natural language
understanding (NLU) tasks. Such a fast pace of research would not be possible without general NLU benchmarks, which
allow for a fair comparison of the proposed methods. However, such benchmarks are available only for a handful of
languages. To alleviate this issue, we introduce a comprehensive multi-task benchmark for the Polish language
understanding, accompanied by an online leaderboard. It consists of a diverse set of tasks, adopted from existing
datasets for named entity recognition, question-answering, textual entailment, and others. We also introduce a new
sentiment analysis task for the e-commerce domain, named Allegro Reviews (AR). To ensure a common evaluation scheme and
promote models that generalize to different NLU tasks, the benchmark includes datasets from varying domains and
applications. Additionally, we release HerBERT, a Transformer-based model trained specifically for the Polish language,
which has the best average performance and obtains the best results for three out of nine tasks. Finally, we provide an
extensive evaluation, including several standard baselines and recently proposed, multilingual Transformer-based
models.*
*최근 몇 년간, Transformer 기반 모델들은 일반적인 자연어 이해(NLU) 작업에서 큰 발전을 이루어냈습니다. 이렇게 연구가 빠르게 발전할 수 있었던 것은  공정한 비교를 가능하게 하는 일반 NLU 벤치마크 없이는 불가능했을 것입니다. 그러나, 이러한 벤치마크는 소수의 언어에만 제공됩니다. 이 문제를 해결하기 위해 우리는 폴란드어 이해를 위한 종합적인 다중 작업 벤치마크를 소개하며, 온라인 리더보드를 제공합니다. 이 벤치마크는 기존 데이터셋에서 가져온 개체명 인식, 질문 응답, 텍스트 함의 등의 다양한 다양한 작업으로 구성되어 있습니다. 또한, 전자상거래 도메인에 대한 새로운 감정 분석 작업인 Allegro Reviews(AR)를 같이 소개합니다. 다양한 NLU 작업에 대한 공통된 평가 기준을 제공하고, 일반화 능력을 가진 모델을 지원하기 위해, 이 벤치마크는 여러 도메인과 응용 분야에서의 데이터셋을 포함합니다. 더해서, 우리는 폴란드어에 특하된 Transformer 기반 모델인 HerBERT를 공개했습니다. 이 모델은 가장 우수한 평균 성능을 보였고, 9개 작업 중 3개 작업에서 최고의 결과를 달성했습니다. 최종적으로, 우리는 여러 표준 기준선 및 최근 제안된 다중 언어 Transformer 모델들을 포함한 폭넓은 평가를 제공합니다.

이 모델은 [rmroczkowski](https://huggingface.co/rmroczkowski)에 의해 제공되었습니다. 원본 코드는[here](https://github.com/allegro/HerBERT)에서 확인할 수 있습니다.


## 사용 예시[[Usage example]]

```python
>>> from transformers import HerbertTokenizer, RobertaModel

>>> tokenizer = HerbertTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
>>> model = RobertaModel.from_pretrained("allegro/herbert-klej-cased-v1")

>>> encoded_input = tokenizer.encode("Kto ma lepszą sztukę, ma lepszy rząd – to jasne.", return_tensors="pt")
>>> outputs = model(encoded_input)

>>> # HerBERT can also be loaded using AutoTokenizer and AutoModel:
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
>>> model = AutoModel.from_pretrained("allegro/herbert-klej-cased-v1")
```

<Tip>
 
HerBERT의 구현은 `BERT`와 동일하지만 토큰화 방법에서 차이가 있습니다. API참조와 예시에 대한 부분은 [BERT 문서](bert)를 참고하세요. 

</Tip>

## HerbertTokenizer

[[autodoc]] HerbertTokenizer

## HerbertTokenizerFast

[[autodoc]] HerbertTokenizerFast
