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

# 토크나이저 요약[[summary-of-the-tokenizers]]

[[open-in-colab]]

이 페이지에서는 토큰화에 대해 자세히 살펴보겠습니다.

<Youtube id="VFp38yj8h3A"/>

[데이터 전처리하기 튜토리얼](preprocessing)에서 살펴본 것처럼, 텍스트를 토큰화하는 것은 텍스트를 단어 또는 서브워드로 분할하고 룩업 테이블을 통해 id로 변환하는 과정입니다.
단어 또는 서브워드를 id로 변환하는 것은 간단하기 때문에 이번 문서에서는 텍스트를 단어 또는 서브워드로 쪼개는 것(즉, 텍스트를 토큰화하는 것)에 중점을 두겠습니다.
구체적으로, 🤗 Transformers에서 사용되는 세 가지 주요 토큰화 유형인 [Byte-Pair Encoding (BPE)](#byte-pair-encoding), [WordPiece](#wordpiece), [SentencePiece](#sentencepiece)를 살펴보고 어떤 모델에서 어떤 토큰화 유형을 사용하는지 예시를 보여드리겠습니다.

각 모델 페이지에 연결된 토크나이저의 문서를 보면 사전 훈련 모델에서 어떤 토크나이저를 사용했는지 알 수 있습니다.
예를 들어, [`BertTokenizer`]를 보면 이 모델이 [WordPiece](#wordpiece)를 사용하는 것을 알 수 있습니다.

## 개요[[introduction]]

텍스트를 작은 묶음(chunk)으로 쪼개는 것은 보기보다 어려운 작업이며, 여러 가지 방법이 있습니다.
예를 들어, `"Don't you love 🤗 Transformers? We sure do."` 라는 문장을 살펴보도록 하겠습니다.

<Youtube id="nhJxYji1aho"/>

위 문장을 토큰화하는 간단한 방법은 공백을 기준으로 쪼개는 것입니다.
토큰화된 결과는 다음과 같습니다:

```
["Don't", "you", "love", "🤗", "Transformers?", "We", "sure", "do."]
```
이는 첫 번째 결과로는 합리적이지만, `"Transformers?"`와 `"do."`토큰을 보면 각각 `"Transformer"`와 `"do"`에 구두점이 붙어있는 것을 확인할 수 있습니다.
구두점을 고려해야 모델이 단어의 다른 표현과 그 뒤에 올 수 있는 모든 가능한 구두점을 학습할 필요가 없습니다. 그렇지 않으면 모델이 학습해야 하는 표현의 수가 폭발적으로 증가하게 됩니다.

구두점을 고려한 토큰화 결과는 다음과 같습니다:

```
["Don", "'", "t", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]
```

이전보다 나아졌습니다. 하지만, `"Don't"`의 토큰화 결과도 수정이 필요합니다.
`"Don't"`는 `"do not"`의 줄임말이기 때문에 `["Do", "n't"]`로 토큰화되는 것이 좋습니다.
여기서부터 복잡해지기 시작합니다. 그리고 이 점이 각 모델마다 고유한 토큰화 유형이 존재하는 이유 중 하나입니다.
텍스트를 토큰화하는 데 적용하는 규칙에 따라 동일한 텍스트에 대해 토큰화된 결과가 달라집니다.
사전 훈련된 모델은 훈련 데이터를 토큰화하는 데 사용된 것과 동일한 규칙으로 토큰화된 입력을 제공해야만 제대로 작동합니다.

[spaCy](https://spacy.io/)와 [Moses](http://www.statmt.org/moses/?n=Development.GetStarted)는 유명한 규칙 기반 토크나이저입니다. 예제에 *spaCy*와 *Moses* 를 적용한 결과는 다음과 같습니다:

```
["Do", "n't", "you", "love", "🤗", "Transformers", "?", "We", "sure", "do", "."]
```

보시다시피 공백 및 구두점 토큰화와 규칙 기반 토큰화가 사용됩니다.
공백 및 구두점, 규칙 기반 토큰화은 모두 단어 문장을 단어로 쪼개는 단어 토큰화에 해당합니다.
이 토큰화 방법은 텍스트를 더 작은 묶음(chunk)로 분할하는 가장 직관적인 방법이지만, 대규모 텍스트 말뭉치에 대해서는 문제가 발생할 수 있습니다.
이 경우 공백 및 구두점 토큰화는 일반적으로 매우 큰 어휘(사용된 모든 고유 단어와 토큰 집합)을 생성합니다.
*예를 들어*, [Transformer XL](model_doc/transformerxl)은 공백 및 구두점 토큰화를 사용해 어휘(vocabulary) 크기가 267,735입니다!

어휘 크기가 크면 모델에 입력 및 출력 레이어로 엄청난 임베딩 행렬이 필요하므로 메모리와 시간 복잡성이 모두 증가합니다.
일반적으로 트랜스포머 모델은 어휘 크기가 50,000개를 넘는 경우가 드물며, 특히 단일 언어에 대해서만 사전 훈련된 경우에는 더욱 그렇습니다.
단순한 공백과 구두점 토큰화가 만족스럽지 않다면 단순히 문자를 토큰화하면 어떨까요?

<Youtube id="ssLq_EK2jLE"/>

문자 토큰화는 아주 간단하고 메모리와 시간 복잡도를 크게 줄일 수 있지만, 모델이 의미 있는 입력 표현을 학습하기에는 훨씬 더 어렵습니다.

*예를 들어*, 문자 `"t"`에 대한 의미 있는 문맥 독립적 표현을 배우는 것 보다 단어 `"today"`에 대한 의미 있는 문맥 독립적 표현을 배우는 것이 훨씬 더 어렵습니다.
문자 토큰화는 종종 성능 저하를 동반하기 때문에 두 가지 장점을 모두 얻기 위해 트랜스포머 모델은 **서브워드** 토큰화라고 하는 단어 수준과 문자 수준 토큰화의 하이브리드를 사용합니다.

## 서브워드 토큰화[[subword-tokenization]]

<Youtube id="zHvTiHr506c"/>

서브워드 토큰화 알고리즘은 자주 사용되는 단어는 더 작은 하위 단어로 쪼개고, 드문 단어는 의미 있는 하위 단어로 분해되어야 한다는 원칙에 따라 작동합니다.
예를 들어 `"annoyingly"`는 드문 단어로 간주되어 `"annoying"`과 `"ly"`로 분해될 수 있습니다.
`"annoyingly"`가 `"annoying"`과 `"ly"`의 합성어인 반면, `"annoying"`과 `"ly"` 둘 다 독립적인 서브워드로 자주 등장합니다.
이는 터키어와 같은 응집성 언어에서 특히 유용하며, 서브워드를 묶어 임의로 긴 복합 단어를 만들 수 있습니다.

서브워드 토큰화를 사용하면 모델이 의미 있는 문맥 독립적 표현을 학습하면서 합리적인 어휘 크기를 가질 수 있습니다.
또한, 서브워드 토큰화를 통해 모델은 이전에 본 적이 없는 단어를 알려진 서브워드로 분해하여 처리할 수 있습니다.

예를 들어, [`~transformers.BertTokenizer`]는 `"I have a new GPU!"` 라는 문장을 아래와 같이 토큰화합니다:

```py
>>> from transformers import BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> tokenizer.tokenize("I have a new GPU!")
["i", "have", "a", "new", "gp", "##u", "!"]
```

대소문자가 없는 모델을 사용해 문장의 시작이 소문자로 표기되었습니다.
단어 `["i", "have", "a", "new"]`는 토크나이저의 어휘에 속하지만, `"gpu"`는 속하지 않는 것을 확인할 수 있습니다.
결과적으로 토크나이저는 `"gpu"`를 알려진 두 개의 서브워드로 쪼갭니다: `["gp" and "##u"]`.
`"##"`은 토큰의 나머지 부분이 공백 없이 이전 토큰에 연결되어야(attach) 함을 의미합니다(토큰화 디코딩 또는 역전을 위해).

또 다른 예로, [`~transformers.XLNetTokenizer`]는 이전에 예시 문장을 다음과 같이 토큰화합니다:
```py
>>> from transformers import XLNetTokenizer

>>> tokenizer = XLNetTokenizer.from_pretrained("xlnet/xlnet-base-cased")
>>> tokenizer.tokenize("Don't you love 🤗 Transformers? We sure do.")
["▁Don", "'", "t", "▁you", "▁love", "▁", "🤗", "▁", "Transform", "ers", "?", "▁We", "▁sure", "▁do", "."]
```

`"▁"`가 가지는 의미는 [SentencePiece](#sentencepiece)에서 다시 살펴보도록 하겠습니다.
보다시피 `"Transformers"` 라는 드문 단어는 서브워드 `"Transform"`와 `"ers"`로 쪼개집니다.

이제 다양한 하위 단어 토큰화 알고리즘이 어떻게 작동하는지 살펴보겠습니다.
이러한 토큰화 알고리즘은 일반적으로 해당 모델이 학습되는 말뭉치에 대해 수행되는 어떤 형태의 학습에 의존한다는 점에 유의하세요.

<a id='byte-pair-encoding'></a>

### 바이트 페어 인코딩 (Byte-Pair Encoding, BPE)[[bytepair-encoding-bpe]]

바이트 페어 인코딩(BPE)은 [Neural Machine Translation of Rare Words with Subword Units (Sennrich et
al., 2015)](https://arxiv.org/abs/1508.07909) 에서 소개되었습니다.
BPE는 훈련 데이터를 단어로 분할하는 사전 토크나이저(pre-tokenizer)에 의존합니다.
사전 토큰화(Pretokenization)에는 [GPT-2](model_doc/gpt2), [Roberta](model_doc/roberta)와 같은 간단한 공백 토큰화가 있습니다.
복잡한 사전 토큰화에는 규칙 기반 토큰화가 해당하는데, 훈련 말뭉치에서 각 단어의 빈도를 계산하기 위해 사용합니다.
[XLM](model_doc/xlm), 대부분의 언어에서 Moses를 사용하는 [FlauBERT](model_doc/flaubert), Spacy와 ftfy를 사용하는 [GPT](model_doc/gpt)가 해당합니다.


사전 토큰화 이후에, 고유 단어 집합가 생성되고 훈련 데이터에서 각 단어가 등장하는 빈도가 결정됩니다.
다음으로, BPE는 고유 단어 집합에 나타나는 모든 기호로 구성된 기본 어휘를 생성하고 기본 어휘의 두 기호에서 새로운 기호를 형성하는 병합 규칙을 학습합니다.
어휘가 원하는 어휘 크기에 도달할 때까지 위의 과정을 반복합니다.
어휘 크기는 토크나이저를 훈련시키기 전에 정의해야 하는 하이퍼파라미터라는 점을 유의하세요.

예를 들어, 사전 토큰화 후 빈도를 포함한 다음과 같은 어휘 집합이 결정되었다고 가정해 보겠습니다:

```
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

결과적으로 기본 어휘는 `["b", "g", "h", "n", "p", "s", "u"]` 이고, 각 단어를 기본 어휘에 속하는 기호로 쪼개면 아래와 같습니다:

```
("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)
```

그런 다음 BPE는 가능한 각 기호 쌍의 빈도를 계산하여 가장 자주 발생하는 기호 쌍을 선택합니다.
위의 예시에서 `"h"` 뒤에 오는 `"u"`는 _10 + 5 = 15_ 번 등장합니다. (`"hug"`에서 10번, `"hugs"`에서 5번 등장)

하지만, 가장 등장 빈도가 높은 기호 쌍은 `"u"` 뒤에 오는 `"g"`입니다. _10 + 5 + 5 = 20_ 으로 총 20번 등장합니다.
따라서 토크나이저가 병합하는 가장 첫 번째 쌍은 `"u"` 뒤에 오는 `"g"`입니다. `"ug"`가 어휘에 추가되어 어휘는 다음과 같습니다:

```
("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)
```

BPE는 다음으로 가장 많이 등장하는 기호 쌍을 식별합니다.
`"u"` 뒤에 오는 `"n"`은 16번 등장해 `"un"` 으로 병합되어 어휘에 추가됩니다.
그 다음으로 빈도수가 놓은 기호 쌍은 `"h"` 뒤에 오는 `"ug"`로 15번 등장합니다.
다시 한 번 `"hug"`로 병합되어 어휘에 추가됩니다.

현재 단계에서 어휘는 `["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"]` 이고, 고유 단어 집합은 다음과 같습니다:

```
("hug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("hug" "s", 5)
```

이 시점에서 바이트 페어 인코딩 훈련이 중단된다고 가정하면, 훈련된 병합 규칙은 새로운 단어에 적용됩니다(기본 어휘에 포함된 기호가 새로운 단어에 포함되지 않는 한).
예를 들어, 단어 `"bug"`는 `["b", "ug"]`로 토큰화되지만, `"m"`이 기본 어휘에 없기 때문에 `"mug"`는 `["<unk>", "ug"]`로 토큰화될 것입니다.
훈련 데이터에는 단일 문자가 최소한 한 번 등장하기 때문에 일반적으로 `"m"`과 같은 단일 문자는 `"<unk>"` 기호로 대체되지 않지만, 이모티콘과 같은 특별한 문자인 경우에는 대체될 수 있습니다.

이전에 언급했듯이 어휘 크기(즉 기본 어휘 크기 + 병합 횟수)는 선택해야하는 하이퍼파라미터입니다.
예를 들어 [GPT](model_doc/gpt)의 기본 어휘 크기는 478, 40,000번의 병합 이후에 훈련을 종료하기 때문에 어휘 크기가 40,478입니다.

#### 바이트 수준 BPE (Byte-level BPE)[[bytelevel-bpe]]

가능한 모든 기본 문자를 포함하는 기본 어휘의 크기는 굉장히 커질 수 있습니다. (예: 모든 유니코드 문자를 기본 문자로 간주하는 경우)
더 나은 기본 어휘를 갖도록 [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)는 기본 어휘로 바이트(bytes)를 사용합니다.
이 방식은 모든 기본 문자가 어휘에 포함되도록 하면서 기본 어휘의 크기를 256으로 제한합니다.
구두점을 다루는 추가적인 규칙을 사용해 GPT2 토크나이저는 모든 텍스트를 <unk> 기호 없이 토큰화할 수 있습니다.
[GPT-2](model_doc/gpt)의 어휘 크기는 50,257로 256 바이트 크기의 기본 토큰, 특별한 end-of-text 토큰과 50,000번의 병합으로 학습한 기호로 구성됩니다.

<a id='wordpiece'></a>

### 워드피스 (WordPiece)[[wordpiece]]

워드피스는 [BERT](model_doc/bert), [DistilBERT](model_doc/distilbert), [Electra](model_doc/electra)에 사용된 서브워드 토큰화 알고리즘입니다.
이 알고리즘은 [Japanese and Korean Voice Search (Schuster et al., 2012)](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)에서 소개되었고, BPE와 굉장히 유사합니다.
워드피스는 훈련 데이터에 등장하는 모든 문자로 기본 어휘를 초기화한 후, 주어진 병합 규칙에 따라 점진적으로 학습합니다.
BPE와는 대조적으로 워드피스는 가장 빈도수가 높은 기호 쌍을 선택하지 않고, 어휘에 추가되었을 때 훈련 데이터의 우도가 최대화되는 쌍을 선택합니다.

정확히 무슨 의미일까요?
이전 예시를 참조하면, 훈련 데이터의 우도 값을 최대화하는 것은 모든 기호 쌍 중에서 첫 번째 기호와 두 번째 기호의 확률로 나눈 확률이 가장 큰 기호 쌍을 찾는 것과 동일합니다.
예를 들어 `"ug"`의 확률이 `"u"`와 `"g"` 각각으로 쪼개졌을 때 보다 높아야 `"u"` 뒤에 오는 `"g"`는 병합될 것입니다.
직관적으로 워드피스는 두 기호를 병합하여 _잃는_ 것을 평가하여 그만한 _가치_가 있는지 확인한다는 점에서 BPE와 약간 다릅니다.

<a id='unigram'></a>

### 유니그램 (Unigram)[[unigram]]

유니그램은 [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates (Kudo, 2018)](https://arxiv.org/pdf/1804.10959.pdf)에서 제안된 서브워드 토큰화 알고리즘입니다.
BPE나 워드피스와 달리 유니그램은 기본 어휘를 많은 수의 기호로 초기화한 후 각 기호를 점진적으로 줄여 더 작은 어휘를 얻습니다.
예를 들어 기본 어휘는 모든 사전 토큰화된 단어와 가장 일반적인 하위 문자열에 해당할 수 있습니다.
유니그램은 transformers 모델에서 직접적으로 사용되지는 않지만, [SentencePiece](#sentencepiece)와 함께 사용됩니다.

각 훈련 단계에서 유니그램 알고리즘은 현재 어휘와 유니그램 언어 모델이 주어졌을 때 훈련 데이터에 대한 손실(흔히 로그 우도로 정의됨)을 정의합니다.
그런 다음 어휘의 각 기호에 대해 알고리즘은 해당 기호를 어휘에서 제거할 경우 전체 손실이 얼마나 증가할지 계산합니다.
이후에 유니그램은 손실 증가율이 가장 낮은 기호의 p(보통 10% 또는 20%) 퍼센트를 제거합니다. (제거되는 기호는 훈련 데이터에 대한 전체 손실에 가장 작은 영향을 미칩니다.)
어휘가 원하는 크기에 도달할 때까지 이 과정을 반복합니다.
유니그램 알고리즘은 항상 기본 문자를 포함해 어떤 단어라도 토큰화할 수 있습니다.
유니그램이 병합 규칙에 기반하지 않기 떄문에 (BPE나 워드피스와는 대조적으로), 해당 알고리즘은 훈련 이후에 새로운 텍스트를 토큰화하는데 여러 가지 방법이 있습니다.

예를 들어, 훈련된 유니그램 토큰화가 다음과 같은 어휘를 가진다면:

```
["b", "g", "h", "n", "p", "s", "u", "ug", "un", "hug"],
```

`"hugs"`는 두 가지로 토큰화할 수 있습니다. `["hug", "s"]`와 `["h", "ug", "s"]` 또는 `["h", "u", "g", "s"]`.

그렇다면 어떤 토큰화 방법을 선택해야 할까요?
유니그램은 어휘를 저장하는 것 외에도 훈련 말뭉치에 각 토큰의 확률을 저장하여 훈련 후 가능한 각 토큰화의 확률을 계산할 수 있도록 합니다.
이 알고리즘은 단순히 실제로 가장 가능성이 높은 토큰화를 선택하지만, 확률에 따라 가능한 토큰화를 샘플링할 수 있는 가능성도 제공합니다.
이러한 확률은 토크나이저가 학습한 손실에 의해 정의됩니다.

단어로 구성된 훈련 데이터를 \\(x_{1}, \dots, x_{N}\\)라 하고, 단어 \\(x_{i}\\)에 대한 가능한 모든 토큰화 결과를 \\(S(x_{i})\\)라 한다면, 전체 손실은 다음과 같이 정의됩니다:

$$\mathcal{L} = -\sum_{i=1}^{N} \log \left ( \sum_{x \in S(x_{i})} p(x) \right )$$



<a id='sentencepiece'></a>

### 센텐스피스 (SentencePiece)[[sentencepiece]]

지금까지 다룬 토큰화 알고리즘은 동일한 문제를 가집니다: 입력 텍스트는 공백을 사용하여 단어를 구분한다고 가정합니다.
하지만, 모든 언어에서 단어를 구분하기 위해 공백을 사용하지 않습니다.
한가지 가능한 해결방안은 특정 언어에 특화된 사전 토크나이저를 사용하는 것입니다. 예를 들어 [XLM](model_doc/xlm)은 특정 중국어, 일본어, 태국어 사전 토크나이저를 사용합니다.
이 문제를 일반적인 방법으로 해결하기 위해, [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing (Kudo et al., 2018)](https://arxiv.org/pdf/1808.06226.pdf)는 입력을 스트림으로 처리해 공백를 하나의 문자로 사용합니다.
이후에 BPE 또는 유니그램 알고리즘을 사용해 적절한 어휘를 구성합니다.

[`XLNetTokenizer`]는 센텐스피스를 사용하기 때문에, 위에서 다룬 예시에서 어휘에  `"▁"`가 포함되어있습니다.
모든 토큰을 합친 후 `"▁"`을 공백으로 대체하면 되기 때문에 센텐스피스로 토큰화된 결과는 디코딩하기 수월합니다.

transformers에서 제공하는 센텐스피스 토크나이저를 사용하는 모든 모델은 유니그램과 함께 사용됩니다. 
[ALBERT](model_doc/albert), [XLNet](model_doc/xlnet), [Marian](model_doc/marian), [T5](model_doc/t5) 모델이 센텐스피스 토크나이저를 사용합니다.