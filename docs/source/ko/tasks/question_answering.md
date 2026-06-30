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

# 질의 응답(Question Answering)[[question-answering]]

[[open-in-colab]]

<Youtube id="ajPx5LwJD-I"/>

질의 응답 태스크는 주어진 질문에 대한 답변을 제공합니다. Alexa, Siri 또는 Google과 같은 가상 비서에게 날씨가 어떤지 물어본 적이 있다면 질의 응답 모델을 사용해본 적이 있을 것입니다. 질의 응답 태스크에는 일반적으로 두 가지 유형이 있습니다.

- 추출적(Extractive) 질의 응답: 주어진 문맥에서 답변을 추출합니다.
- 생성적(Abstractive) 질의 응답: 문맥에서 질문에 올바르게 답하는 답변을 생성합니다.

이 가이드는 다음과 같은 방법들을 보여줍니다.

1. 추출적 질의 응답을 하기 위해 [SQuAD](https://huggingface.co/datasets/squad) 데이터 세트에서 [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) 미세 조정하기
2. 추론에 미세 조정된 모델 사용하기

<Tip>

이 작업과 호환되는 모든 아키텍처와 체크포인트를 보려면 [작업 페이지](https://huggingface.co/tasks/question-answering)를 확인하는 것이 좋습니다.

</Tip>

시작하기 전에, 필요한 라이브러리가 모두 설치되어 있는지 확인하세요:

```bash
pip install transformers datasets evaluate
```

여러분의 모델을 업로드하고 커뮤니티에 공유할 수 있도록 Hugging Face 계정에 로그인하는 것이 좋습니다. 메시지가 표시되면 토큰을 입력해서 로그인합니다:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## SQuAD 데이터 세트 가져오기[[load-squad-dataset]]

먼저 🤗 Datasets 라이브러리에서 SQuAD 데이터 세트의 일부를 가져옵니다. 이렇게 하면 전체 데이터 세트로 훈련하며 더 많은 시간을 할애하기 전에 모든 것이 잘 작동하는지 실험하고 확인할 수 있습니다.

```py
>>> from datasets import load_dataset

>>> squad = load_dataset("squad", split="train[:5000]")
```

데이터 세트의 분할된 `train`을 [`~datasets.Dataset.train_test_split`] 메소드를 사용해 훈련 데이터 세트와 테스트 데이터 세트로 나누어줍니다:

```py
>>> squad = squad.train_test_split(test_size=0.2)
```

그리고나서 예시로 데이터를 하나 살펴봅니다:

```py
>>> squad["train"][0]
{'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
 'id': '5733be284776f41900661182',
 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
 'title': 'University_of_Notre_Dame'
}
```

이 중에서 몇 가지 중요한 항목이 있습니다:

- `answers`: 답안 토큰의 시작 위치와 답안 텍스트
- `context`: 모델이 답을 추출하는데 필요한 배경 지식
- `question`: 모델이 답해야 하는 질문

## 전처리[[preprocess]]

<Youtube id="qgaM0weJHpA"/>

다음 단계에서는 `question` 및 `context` 항목을 처리하기 위해 DistilBERT 토크나이저를 가져옵니다:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

질의 응답 태스크와 관련해서 특히 유의해야할 몇 가지 전처리 단계가 있습니다:

1. 데이터 세트의 일부 예제에는 모델의 최대 입력 길이를 초과하는 매우 긴 `context`가 있을 수 있습니다. 긴 시퀀스를 다루기 위해서는, `truncation="only_second"`로 설정해 `context`만 잘라내면 됩니다.
2. 그 다음, `return_offset_mapping=True`로 설정해 답변의 시작과 종료 위치를 원래의 `context`에 매핑합니다.
3. 매핑을 완료하면, 이제 답변에서 시작 토큰과 종료 토큰을 찾을 수 있습니다. 오프셋의 어느 부분이 `question`과 `context`에 해당하는지 찾을 수 있도록 [`~tokenizers.Encoding.sequence_ids`] 메소드를 사용하세요.

다음은 `answer`의 시작 토큰과 종료 토큰을 잘라내서 `context`에 매핑하는 함수를 만드는 방법입니다:

```py
>>> def preprocess_function(examples):
...     questions = [q.strip() for q in examples["question"]]
...     inputs = tokenizer(
...         questions,
...         examples["context"],
...         max_length=384,
...         truncation="only_second",
...         return_offsets_mapping=True,
...         padding="max_length",
...     )

...     offset_mapping = inputs.pop("offset_mapping")
...     answers = examples["answers"]
...     start_positions = []
...     end_positions = []

...     for i, offset in enumerate(offset_mapping):
...         answer = answers[i]
...         start_char = answer["answer_start"][0]
...         end_char = answer["answer_start"][0] + len(answer["text"][0])
...         sequence_ids = inputs.sequence_ids(i)

...         # Find the start and end of the context
...         idx = 0
...         while sequence_ids[idx] != 1:
...             idx += 1
...         context_start = idx
...         while sequence_ids[idx] == 1:
...             idx += 1
...         context_end = idx - 1

...         # If the answer is not fully inside the context, label it (0, 0)
...         if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
...             start_positions.append(0)
...             end_positions.append(0)
...         else:
...             # Otherwise it's the start and end token positions
...             idx = context_start
...             while idx <= context_end and offset[idx][0] <= start_char:
...                 idx += 1
...             start_positions.append(idx - 1)

...             idx = context_end
...             while idx >= context_start and offset[idx][1] >= end_char:
...                 idx -= 1
...             end_positions.append(idx + 1)

...     inputs["start_positions"] = start_positions
...     inputs["end_positions"] = end_positions
...     return inputs
```

모든 데이터 세트에 전처리를 적용하려면, 🤗 Datasets [`~datasets.Dataset.map`] 함수를 사용하세요. `batched=True`로 설정해 데이터 세트의 여러 요소들을 한 번에 처리하면 `map` 함수의 속도를 빠르게 할 수 있습니다. 필요하지 않은 열은 모두 제거합니다:

```py
>>> tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
```

이제 [`DefaultDataCollator`]를 이용해 예시 배치를 생성합니다. 🤗 Transformers의 다른 데이터 콜레이터(data collator)와 달리, [`DefaultDataCollator`]는 패딩과 같은 추가 전처리를 적용하지 않습니다:

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

## 훈련[[train]]

<Tip>

[`Trainer`]를 이용해 모델을 미세 조정하는 것에 익숙하지 않다면, [여기](../training#train-with-pytorch-trainer)에서 기초 튜토리얼을 살펴보세요!

</Tip>

이제 모델 훈련을 시작할 준비가 되었습니다! [`AutoModelForQuestionAnswering`]으로 DistilBERT를 가져옵니다:

```py
>>> from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

>>> model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```

이제 세 단계만 남았습니다:

1. [`TrainingArguments`]에서 훈련 하이퍼파라미터를 정합니다. 꼭 필요한 매개변수는 모델을 저장할 위치를 지정하는 `output_dir` 입니다. `push_to_hub=True`로 설정해서 이 모델을 Hub로 푸시합니다 (모델을 업로드하려면 Hugging Face에 로그인해야 합니다).
2. 모델, 데이터 세트, 토크나이저, 데이터 콜레이터와 함께 [`Trainer`]에 훈련 인수들을 전달합니다.
3. [`~Trainer.train`]을 호출해서 모델을 미세 조정합니다.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_qa_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_squad["train"],
...     eval_dataset=tokenized_squad["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
... )

>>> trainer.train()
```

훈련이 완료되면, [`~transformers.Trainer.push_to_hub`] 매소드를 사용해 모델을 Hub에 공유해서 모든 사람들이 사용할 수 있게 공유해주세요:

```py
>>> trainer.push_to_hub()
```

<Tip>

질의 응답을 위해 모델을 미세 조정하는 방법에 대한 더 자세한 예시는 [PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb) 또는 [TensorFlow notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb)을 참조하세요.

</Tip>

## 평가[[evaluate]]

질의 응답을 평가하려면 상당한 양의 후처리가 필요합니다. 시간이 너무 많이 걸리지 않도록 이 가이드에서는 평가 단계를 생략합니다. [`Trainer`]는 훈련 과정에서 평가 손실(evaluation loss)을 계속 계산하기 때문에 모델의 성능을 대략적으로 알 수 있습니다.

시간에 여유가 있고 질의 응답 모델을 평가하는 방법에 관심이 있다면 🤗 Hugging Face Course의 [Question answering](https://huggingface.co/course/chapter7/7?fw=pt#postprocessing) 챕터를 살펴보세요!

## 추론[[inference]]

이제 모델을 미세 조정했으니 추론에 사용할 수 있습니다!

질문과 모델이 예측하기 원하는 문맥(context)를 생각해보세요:

```py
>>> question = "How many programming languages does BLOOM support?"
>>> context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
```

추론을 위해 미세 조정된 모델을 테스트하는 가장 쉬운 방법은 tokenizer와 model을 직접 사용하는 것 입니다. 텍스트를 토큰화해서 PyTorch 텐서를 반환합니다:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
>>> inputs = tokenizer(question, context, return_tensors="pt")
```

모델에 입력을 전달하고 `logits`을 반환합니다:

```py
>>> from transformers import AutoModelForQuestionAnswering

>>> model = AutoModelForQuestionAnswering.from_pretrained("my_awesome_qa_model")
>>> with torch.no_grad():
...     outputs = model(**inputs)
```

모델의 출력에서 시작 및 종료 위치가 어딘지 가장 높은 확률을 얻습니다:

```py
>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()
```

예측된 토큰을 해독해서 답을 얻습니다:

```py
>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
'176 billion parameters and can generate text in 46 languages natural languages and 13'
```
