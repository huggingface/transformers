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

# 번역[[translation]]

[[open-in-colab]]

<Youtube id="1JvfrvZgi6c"/>

번역은 한 언어로 된 시퀀스를 다른 언어로 변환합니다. 번역이나 요약은 입력을 받아 일련의 출력을 반환하는 강력한 프레임워크인 시퀀스-투-시퀀스 문제로 구성할 수 있는 대표적인 태스크입니다. 번역 시스템은 일반적으로 다른 언어로 된 텍스트 간의 번역에 사용되지만, 음성 간의 통역이나 텍스트-음성 또는 음성-텍스트와 같은 조합에도 사용될 수 있습니다.

이 가이드에서 학습할 내용은:

1. 영어 텍스트를 프랑스어로 번역하기 위해 [T5](https://huggingface.co/google-t5/t5-small) 모델을 OPUS Books 데이터세트의 영어-프랑스어 하위 집합으로 파인튜닝하는 방법과
2. 파인튜닝된 모델을 추론에 사용하는 방법입니다.

<Tip>

이 작업과 호환되는 모든 아키텍처와 체크포인트를 보려면 [작업 페이지](https://huggingface.co/tasks/translation)를 확인하는 것이 좋습니다.

</Tip>

시작하기 전에 필요한 라이브러리가 모두 설치되어 있는지 확인하세요:

```bash
pip install transformers datasets evaluate sacrebleu
```

모델을 업로드하고 커뮤니티와 공유할 수 있도록 Hugging Face 계정에 로그인하는 것이 좋습니다. 새로운 창이 표시되면 토큰을 입력하여 로그인하세요.

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## OPUS Books 데이터세트 가져오기[[load-opus-books-dataset]]

먼저 🤗 Datasets 라이브러리에서 [OPUS Books](https://huggingface.co/datasets/opus_books) 데이터세트의 영어-프랑스어 하위 집합을 가져오세요.

```py
>>> from datasets import load_dataset

>>> books = load_dataset("opus_books", "en-fr")
```

데이터세트를 [`~datasets.Dataset.train_test_split`] 메서드를 사용하여 훈련 및 테스트 데이터로 분할하세요.

```py
>>> books = books["train"].train_test_split(test_size=0.2)
```

훈련 데이터에서 예시를 살펴볼까요?

```py
>>> books["train"][0]
{'id': '90560',
 'translation': {'en': 'But this lofty plateau measured only a few fathoms, and soon we reentered Our Element.',
  'fr': 'Mais ce plateau élevé ne mesurait que quelques toises, et bientôt nous fûmes rentrés dans notre élément.'}}
```

반환된 딕셔너리의 `translation` 키가 텍스트의 영어, 프랑스어 버전을 포함하고 있는 것을 볼 수 있습니다.

## 전처리[[preprocess]]

<Youtube id="XAR8jnZZuUs"/>

다음 단계로 영어-프랑스어 쌍을 처리하기 위해 T5 토크나이저를 가져오세요.

```py
>>> from transformers import AutoTokenizer

>>> checkpoint = "google-t5/t5-small"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

만들 전처리 함수는 아래 요구사항을 충족해야 합니다:

1. T5가 번역 태스크임을 인지할 수 있도록 입력 앞에 프롬프트를 추가하세요. 여러 NLP 태스크를 할 수 있는 모델 중 일부는 이렇게 태스크 프롬프트를 미리 줘야합니다.
2. 원어(영어)과 번역어(프랑스어)를 별도로 토큰화하세요. 영어 어휘로 사전 학습된 토크나이저로 프랑스어 텍스트를 토큰화할 수는 없기 때문입니다.
3. `max_length` 매개변수로 설정한 최대 길이보다 길지 않도록 시퀀스를 truncate하세요.

```py
>>> source_lang = "en"
>>> target_lang = "fr"
>>> prefix = "translate English to French: "


>>> def preprocess_function(examples):
...     inputs = [prefix + example[source_lang] for example in examples["translation"]]
...     targets = [example[target_lang] for example in examples["translation"]]
...     model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
...     return model_inputs
```

전체 데이터세트에 전처리 함수를 적용하려면 🤗 Datasets의 [`~datasets.Dataset.map`] 메서드를 사용하세요. `map` 함수의 속도를 높이려면 `batched=True`를 설정하여 데이터세트의 여러 요소를 한 번에 처리하는 방법이 있습니다.

```py
>>> tokenized_books = books.map(preprocess_function, batched=True)
```

이제 [`DataCollatorForSeq2Seq`]를 사용하여 예제 배치를 생성합니다. 데이터세트의 최대 길이로 전부를 padding하는 대신, 데이터 정렬 중 각 배치의 최대 길이로 문장을 *동적으로 padding*하는 것이 더 효율적입니다.

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
```

## 평가[[evalulate]]

훈련 중에 메트릭을 포함하면 모델의 성능을 평가하는 데 도움이 됩니다. 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 라이브러리로 평가 방법(evaluation method)을 빠르게 가져올 수 있습니다. 현재 태스크에 적합한 SacreBLEU 메트릭을 가져오세요. (메트릭을 가져오고 계산하는 방법에 대해 자세히 알아보려면 🤗 Evaluate [둘러보기](https://huggingface.co/docs/evaluate/a_quick_tour)를 참조하세요):

```py
>>> import evaluate

>>> metric = evaluate.load("sacrebleu")
```

그런 다음 [`~evaluate.EvaluationModule.compute`]에 예측값과 레이블을 전달하여 SacreBLEU 점수를 계산하는 함수를 생성하세요:

```py
>>> import numpy as np


>>> def postprocess_text(preds, labels):
...     preds = [pred.strip() for pred in preds]
...     labels = [[label.strip()] for label in labels]

...     return preds, labels


>>> def compute_metrics(eval_preds):
...     preds, labels = eval_preds
...     if isinstance(preds, tuple):
...         preds = preds[0]
...     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

...     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
...     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

...     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

...     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
...     result = {"bleu": result["score"]}

...     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
...     result["gen_len"] = np.mean(prediction_lens)
...     result = {k: round(v, 4) for k, v in result.items()}
...     return result
```

이제 `compute_metrics` 함수는 준비되었고, 훈련 과정을 설정할 때 다시 살펴볼 예정입니다.

## 훈련[[train]]

<Tip>

[`Trainer`]로 모델을 파인튜닝하는 방법에 익숙하지 않다면 [여기](../training#train-with-pytorch-trainer)에서 기본 튜토리얼을 살펴보시기 바랍니다!

</Tip>

모델을 훈련시킬 준비가 되었군요! [`AutoModelForSeq2SeqLM`]으로 T5를 로드하세요:

```py
>>> from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

이제 세 단계만 거치면 끝입니다:

1. [`Seq2SeqTrainingArguments`]에서 훈련 하이퍼파라미터를 정의하세요. 유일한 필수 매개변수는 모델을 저장할 위치인 `output_dir`입니다. 모델을 Hub에 푸시하기 위해 `push_to_hub=True`로 설정하세요. (모델을 업로드하려면 Hugging Face에 로그인해야 합니다.) [`Trainer`]는 에폭이 끝날때마다 SacreBLEU 메트릭을 평가하고 훈련 체크포인트를 저장합니다.
2. [`Seq2SeqTrainer`]에 훈련 인수를 전달하세요. 모델, 데이터 세트, 토크나이저, data collator 및 `compute_metrics` 함수도 덩달아 전달해야 합니다.
3. [`~Trainer.train`]을 호출하여 모델을 파인튜닝하세요.

```py
>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="my_awesome_opus_books_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     weight_decay=0.01,
...     save_total_limit=3,
...     num_train_epochs=2,
...     predict_with_generate=True,
...     fp16=True,
...     push_to_hub=True,
... )

>>> trainer = Seq2SeqTrainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_books["train"],
...     eval_dataset=tokenized_books["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

학습이 완료되면 [`~transformers.Trainer.push_to_hub`] 메서드로 모델을 Hub에 공유하세요. 이러면 누구나 모델을 사용할 수 있게 됩니다:

```py
>>> trainer.push_to_hub()
```

<Tip>

번역을 위해 모델을 파인튜닝하는 방법에 대한 보다 자세한 예제는 해당 [PyTorch 노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb) 또는 [TensorFlow 노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation-tf.ipynb)을 참조하세요.

</Tip>

## 추론[[inference]]

좋아요, 이제 모델을 파인튜닝했으니 추론에 사용할 수 있습니다!

다른 언어로 번역하고 싶은 텍스트를 써보세요. T5의 경우 원하는 태스크를 입력의 접두사로 추가해야 합니다. 예를 들어 영어에서 프랑스어로 번역하는 경우, 아래와 같은 접두사가 추가됩니다:

```py
>>> text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
```

텍스트를 토큰화하고 `input_ids`를 PyTorch 텐서로 반환하세요:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_opus_books_model")
>>> inputs = tokenizer(text, return_tensors="pt").input_ids
```

[`~generation.GenerationMixin.generate`] 메서드로 번역을 생성하세요. 다양한 텍스트 생성 전략 및 생성을 제어하기 위한 매개변수에 대한 자세한 내용은 [Text Generation](../main_classes/text_generation) API를 살펴보시기 바랍니다.

```py
>>> from transformers import AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained("my_awesome_opus_books_model")
>>> outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
```

생성된 토큰 ID들을 다시 텍스트로 디코딩하세요:

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'Les lignées partagent des ressources avec des bactéries enfixant l'azote.'
```
