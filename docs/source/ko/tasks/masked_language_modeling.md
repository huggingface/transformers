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

# 마스킹된 언어 모델링(Masked language modeling)[[masked-language-modeling]]

[[open-in-colab]]

<Youtube id="mqElG5QJWUg"/>

마스킹된 언어 모델링은 시퀀스에서 마스킹된 토큰을 예측하며, 모델은 양방향으로 토큰에 액세스할 수 있습니다.
즉, 모델은 토큰의 왼쪽과 오른쪽 양쪽에서 접근할 수 있습니다.
마스킹된 언어 모델링은 전체 시퀀스에 대한 문맥적 이해가 필요한 작업에 적합하며, BERT가 그 예에 해당합니다.

이번 가이드에서 다룰 내용은 다음과 같습니다:

1. [ELI5](https://huggingface.co/datasets/eli5) 데이터 세트에서 [r/askscience](https://www.reddit.com/r/askscience/) 부분을 사용해 [DistilRoBERTa](https://huggingface.co/distilbert/distilroberta-base) 모델을 미세 조정합니다.
2. 추론 시에 직접 미세 조정한 모델을 사용합니다.

<Tip>

이 작업과 호환되는 모든 아키텍처와 체크포인트를 보려면 [작업 페이지](https://huggingface.co/tasks/fill-mask)를 확인하는 것이 좋습니다.

</Tip>

시작하기 전에 필요한 라이브러리가 모두 설치되어 있는지 확인하세요:

```bash
pip install transformers datasets evaluate
```

Hugging Face 계정에 로그인하여 모델을 업로드하고 커뮤니티와의 공유를 권장합니다. 메시지가 표시되면(When prompted) 토큰을 입력하여 로그인합니다:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## ELI5 데이터 세트 가져오기[[load-eli5-dataset]]

먼저 🤗 Datasets 라이브러리에서 ELI5 데이터 세트의 r/askscience 중 일부만 가져옵니다.
이렇게 하면 전체 데이터 세트 학습에 더 많은 시간을 할애하기 전에 모든 것이 작동하는지 실험하고 확인할 수 있습니다.

```py
>>> from datasets import load_dataset

>>> eli5 = load_dataset("eli5", split="train_asks[:5000]")
```

데이터 세트의 `train_asks`를 [`~datasets.Dataset.train_test_split`] 메소드를 사용해 훈련 데이터와 테스트 데이터로 분할합니다:

```py
>>> eli5 = eli5.train_test_split(test_size=0.2)
```

그리고 아래 예시를 살펴보세요:

```py
>>> eli5["train"][0]
{'answers': {'a_id': ['c3d1aib', 'c3d4lya'],
  'score': [6, 3],
  'text': ["The velocity needed to remain in orbit is equal to the square root of Newton's constant times the mass of earth divided by the distance from the center of the earth. I don't know the altitude of that specific mission, but they're usually around 300 km. That means he's going 7-8 km/s.\n\nIn space there are no other forces acting on either the shuttle or the guy, so they stay in the same position relative to each other. If he were to become unable to return to the ship, he would presumably run out of oxygen, or slowly fall into the atmosphere and burn up.",
   "Hope you don't mind me asking another question, but why aren't there any stars visible in this photo?"]},
 'answers_urls': {'url': []},
 'document': '',
 'q_id': 'nyxfp',
 'selftext': '_URL_0_\n\nThis was on the front page earlier and I have a few questions about it. Is it possible to calculate how fast the astronaut would be orbiting the earth? Also how does he stay close to the shuttle so that he can return safely, i.e is he orbiting at the same speed and can therefore stay next to it? And finally if his propulsion system failed, would he eventually re-enter the atmosphere and presumably die?',
 'selftext_urls': {'url': ['http://apod.nasa.gov/apod/image/1201/freeflyer_nasa_3000.jpg']},
 'subreddit': 'askscience',
 'title': 'Few questions about this space walk photograph.',
 'title_urls': {'url': []}}
```

많아 보일 수 있지만 실제로는 `text` 필드에만 집중하면 됩나다.
언어 모델링 작업의 멋진 점은 (비지도 학습으로) *다음 단어가 레이블*이기 때문에 레이블이 따로 필요하지 않습니다.

## 전처리[[preprocess]]

<Youtube id="8PmhEIXhBvI"/>

마스킹된 언어 모델링을 위해, 다음 단계로 DistilRoBERTa 토크나이저를 가져와서 `text` 하위 필드를 처리합니다:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
```

위의 예제에서와 마찬가지로, `text` 필드는 `answers` 안에 중첩되어 있습니다.
따라서 중첩된 구조에서 [`flatten`](https://huggingface.co/docs/datasets/process#flatten) 메소드를 사용하여 `text` 하위 필드를 추출합니다:

```py
>>> eli5 = eli5.flatten()
>>> eli5["train"][0]
{'answers.a_id': ['c3d1aib', 'c3d4lya'],
 'answers.score': [6, 3],
 'answers.text': ["The velocity needed to remain in orbit is equal to the square root of Newton's constant times the mass of earth divided by the distance from the center of the earth. I don't know the altitude of that specific mission, but they're usually around 300 km. That means he's going 7-8 km/s.\n\nIn space there are no other forces acting on either the shuttle or the guy, so they stay in the same position relative to each other. If he were to become unable to return to the ship, he would presumably run out of oxygen, or slowly fall into the atmosphere and burn up.",
  "Hope you don't mind me asking another question, but why aren't there any stars visible in this photo?"],
 'answers_urls.url': [],
 'document': '',
 'q_id': 'nyxfp',
 'selftext': '_URL_0_\n\nThis was on the front page earlier and I have a few questions about it. Is it possible to calculate how fast the astronaut would be orbiting the earth? Also how does he stay close to the shuttle so that he can return safely, i.e is he orbiting at the same speed and can therefore stay next to it? And finally if his propulsion system failed, would he eventually re-enter the atmosphere and presumably die?',
 'selftext_urls.url': ['http://apod.nasa.gov/apod/image/1201/freeflyer_nasa_3000.jpg'],
 'subreddit': 'askscience',
 'title': 'Few questions about this space walk photograph.',
 'title_urls.url': []}
```

이제 각 하위 필드는 `answers` 접두사(prefix)로 표시된 대로 별도의 열이 되고, `text` 필드는 이제 리스트가 되었습니다.
각 문장을 개별적으로 토큰화하는 대신 리스트를 문자열로 변환하여 한번에 토큰화할 수 있습니다.

다음은 각 예제에 대해 문자열로 이루어진 리스트를 `join`하고 결과를 토큰화하는 첫 번째 전처리 함수입니다:

```py
>>> def preprocess_function(examples):
...     return tokenizer([" ".join(x) for x in examples["answers.text"]])
```

이 전처리 함수를 전체 데이터 세트에 적용하기 위해 🤗 Datasets [`~datasets.Dataset.map`] 메소드를 사용합니다.
데이터 세트의 여러 요소를 한 번에 처리하도록 `batched=True`를 설정하고 `num_proc`로 처리 횟수를 늘리면 `map` 함수의 속도를 높일 수 있습니다.
필요하지 않은 열은 제거합니다:

```py
>>> tokenized_eli5 = eli5.map(
...     preprocess_function,
...     batched=True,
...     num_proc=4,
...     remove_columns=eli5["train"].column_names,
... )
```

이 데이터 세트에는 토큰 시퀀스가 포함되어 있지만 이 중 일부는 모델의 최대 입력 길이보다 깁니다.

이제 두 번째 전처리 함수를 사용해
- 모든 시퀀스를 연결하고
- 연결된 시퀀스를 정의한 `block_size` 보다 더 짧은 덩어리로 분할하는데, 이 덩어리는 모델의 최대 입력 길이보다 짧고 GPU RAM이 수용할 수 있는 길이여야 합니다.


```py
>>> block_size = 128


>>> def group_texts(examples):
...     # Concatenate all texts.
...     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
...     total_length = len(concatenated_examples[list(examples.keys())[0]])
...     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
...     # customize this part to your needs.
...     if total_length >= block_size:
...         total_length = (total_length // block_size) * block_size
...     # Split by chunks of block_size.
...     result = {
...         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
...         for k, t in concatenated_examples.items()
...     }
...     result["labels"] = result["input_ids"].copy()
...     return result
```

전체 데이터 세트에 `group_texts` 함수를 적용합니다:

```py
>>> lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
```

이제 [`DataCollatorForLanguageModeling`]을 사용하여 데이터 예제의 배치를 생성합니다.
데이터 세트 전체를 최대 길이로 패딩하는 것보다 collation 단계에서 매 배치안에서의 최대 길이로 문장을 *동적으로 패딩*하는 것이 더 효율적입니다.


시퀀스 끝 토큰을 패딩 토큰으로 사용하고 데이터를 반복할 때마다 토큰을 무작위로 마스킹하도록 `mlm_-probability`를 지정합니다:

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> tokenizer.pad_token = tokenizer.eos_token
>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
```

## 훈련[[train]]

<Tip>

[`Trainer`]로 모델을 미세 조정하는 데 익숙하지 않다면 기본 튜토리얼 [여기](../training#train-with-pytorch-trainer)를 살펴보세요!
</Tip>

이제 모델 훈련을 시작할 준비가 되었습니다! [`AutoModelForMaskedLM`]를 사용해 DistilRoBERTa 모델을 가져옵니다:

```py
>>> from transformers import AutoModelForMaskedLM

>>> model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
```

이제 세 단계가 남았습니다:

1. [`TrainingArguments`]의 훈련 하이퍼파라미터를 정의합니다. 모델 저장 위치를 지정하는 `output_dir`은 유일한 필수 파라미터입니다. `push_to_hub=True`를 설정하여 이 모델을 Hub에 업로드합니다 (모델을 업로드하려면 Hugging Face에 로그인해야 합니다).
2. 모델, 데이터 세트 및 데이터 콜레이터(collator)와 함께 훈련 인수를 [`Trainer`]에 전달합니다.
3. [`~Trainer.train`]을 호출하여 모델을 미세 조정합니다.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_eli5_mlm_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=lm_dataset["train"],
...     eval_dataset=lm_dataset["test"],
...     data_collator=data_collator,
... )

>>> trainer.train()
```

훈련이 완료되면 [`~transformers.Trainer.evaluate`] 메소드를 사용하여 펄플렉서티(perplexity)를 계산하고 모델을 평가합니다:

```py
>>> import math

>>> eval_results = trainer.evaluate()
>>> print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
Perplexity: 8.76
```

그리고 [`~transformers.Trainer.push_to_hub`] 메소드를 사용해 다른 사람들이 사용할 수 있도록, Hub로 모델을 업로드합니다.

```py
>>> trainer.push_to_hub()
```

<Tip>

마스킹된 언어 모델링을 위해 모델을 미세 조정하는 방법에 대한 보다 심층적인 예제는
[PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)
또는 [TensorFlow notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb)을 참조하세요.
</Tip>

## 추론[[inference]]

지금까지 모델 미세 조정을 잘 했으니, 추론에 사용할 수 있습니다!

모델이 빈칸을 채울 텍스트를 스페셜 토큰(special token)인 `<mask>` 토큰으로 표시합니다:


```py
>>> text = "The Milky Way is a <mask> galaxy."
```
추론을 위해 미세 조정한 모델을 테스트하는 가장 간단한 방법은 [`pipeline`]에서 사용하는 것입니다.
`fill-mask`태스크로 `pipeline`을 인스턴스화하고 텍스트를 전달합니다.
`top_k` 매개변수를 사용하여 반환하는 예측의 수를 지정할 수 있습니다:

```py
>>> from transformers import pipeline

>>> mask_filler = pipeline("fill-mask", "stevhliu/my_awesome_eli5_mlm_model")
>>> mask_filler(text, top_k=3)
[{'score': 0.5150994658470154,
  'token': 21300,
  'token_str': ' spiral',
  'sequence': 'The Milky Way is a spiral galaxy.'},
 {'score': 0.07087188959121704,
  'token': 2232,
  'token_str': ' massive',
  'sequence': 'The Milky Way is a massive galaxy.'},
 {'score': 0.06434620916843414,
  'token': 650,
  'token_str': ' small',
  'sequence': 'The Milky Way is a small galaxy.'}]
```

텍스트를 토큰화하고 `input_ids`를 PyTorch 텐서 형태로 반환합니다.
또한, `<mask>` 토큰의 위치를 지정해야 합니다:
```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_eli5_mlm_model")
>>> inputs = tokenizer(text, return_tensors="pt")
>>> mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
```

모델에 `inputs`를 입력하고, 마스킹된 토큰의 `logits`를 반환합니다:

```py
>>> from transformers import AutoModelForMaskedLM

>>> model = AutoModelForMaskedLM.from_pretrained("stevhliu/my_awesome_eli5_mlm_model")
>>> logits = model(**inputs).logits
>>> mask_token_logits = logits[0, mask_token_index, :]
```

그런 다음 가장 높은 확률은 가진 마스크 토큰 3개를 반환하고, 출력합니다:
```py
>>> top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()

>>> for token in top_3_tokens:
...     print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))
The Milky Way is a spiral galaxy.
The Milky Way is a massive galaxy.
The Milky Way is a small galaxy.
```
