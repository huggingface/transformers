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

# 요약[[summarization]]

[[open-in-colab]]

<Youtube id="yHnr5Dk2zCI"/>

요약은 문서나 기사에서 중요한 정보를 모두 포함하되 짧게 만드는 일입니다.
번역과 마찬가지로, 시퀀스-투-시퀀스 문제로 구성할 수 있는 대표적인 작업 중 하나입니다.
요약에는 아래와 같이 유형이 있습니다:

- 추출(Extractive) 요약: 문서에서 가장 관련성 높은 정보를 추출합니다.
- 생성(Abstractive) 요약: 가장 관련성 높은 정보를 포착해내는 새로운 텍스트를 생성합니다.

이 가이드에서 소개할 내용은 아래와 같습니다:

1. 생성 요약을 위한 [BillSum](https://huggingface.co/datasets/billsum) 데이터셋 중 캘리포니아 주 법안 하위 집합으로 [T5](https://huggingface.co/google-t5/t5-small)를 파인튜닝합니다.
2. 파인튜닝된 모델을 사용하여 추론합니다.

<Tip>

이 작업과 호환되는 모든 아키텍처와 체크포인트를 보려면 [작업 페이지](https://huggingface.co/tasks/summarization)를 확인하는 것이 좋습니다.

</Tip>

시작하기 전에 필요한 라이브러리가 모두 설치되어 있는지 확인하세요:

```bash
pip install transformers datasets evaluate rouge_score
```

Hugging Face 계정에 로그인하면 모델을 업로드하고 커뮤니티에 공유할 수 있습니다.
토큰을 입력하여 로그인하세요.

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## BillSum 데이터셋 가져오기[[load-billsum-dataset]]

🤗 Datasets 라이브러리에서 BillSum 데이터셋의 작은 버전인 캘리포니아 주 법안 하위 집합을 가져오세요:

```py
>>> from datasets import load_dataset

>>> billsum = load_dataset("billsum", split="ca_test")
```

[`~datasets.Dataset.train_test_split`] 메소드로 데이터셋을 학습용와 테스트용으로 나누세요:

```py
>>> billsum = billsum.train_test_split(test_size=0.2)
```

그런 다음 예시를 하나 살펴보세요:

```py
>>> billsum["train"][0]
{'summary': 'Existing law authorizes state agencies to enter into contracts for the acquisition of goods or services upon approval by the Department of General Services. Existing law sets forth various requirements and prohibitions for those contracts, including, but not limited to, a prohibition on entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between spouses and domestic partners or same-sex and different-sex couples in the provision of benefits. Existing law provides that a contract entered into in violation of those requirements and prohibitions is void and authorizes the state or any person acting on behalf of the state to bring a civil action seeking a determination that a contract is in violation and therefore void. Under existing law, a willful violation of those requirements and prohibitions is a misdemeanor.\nThis bill would also prohibit a state agency from entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between employees on the basis of gender identity in the provision of benefits, as specified. By expanding the scope of a crime, this bill would impose a state-mandated local program.\nThe California Constitution requires the state to reimburse local agencies and school districts for certain costs mandated by the state. Statutory provisions establish procedures for making that reimbursement.\nThis bill would provide that no reimbursement is required by this act for a specified reason.',
 'text': 'The people of the State of California do enact as follows:\n\n\nSECTION 1.\nSection 10295.35 is added to the Public Contract Code, to read:\n10295.35.\n(a) (1) Notwithstanding any other law, a state agency shall not enter into any contract for the acquisition of goods or services in the amount of one hundred thousand dollars ($100,000) or more with a contractor that, in the provision of benefits, discriminates between employees on the basis of an employee’s or dependent’s actual or perceived gender identity, including, but not limited to, the employee’s or dependent’s identification as transgender.\n(2) For purposes of this section, “contract” includes contracts with a cumulative amount of one hundred thousand dollars ($100,000) or more per contractor in each fiscal year.\n(3) For purposes of this section, an employee health plan is discriminatory if the plan is not consistent with Section 1365.5 of the Health and Safety Code and Section 10140 of the Insurance Code.\n(4) The requirements of this section shall apply only to those portions of a contractor’s operations that occur under any of the following conditions:\n(A) Within the state.\n(B) On real property outside the state if the property is owned by the state or if the state has a right to occupy the property, and if the contractor’s presence at that location is connected to a contract with the state.\n(C) Elsewhere in the United States where work related to a state contract is being performed.\n(b) Contractors shall treat as confidential, to the maximum extent allowed by law or by the requirement of the contractor’s insurance provider, any request by an employee or applicant for employment benefits or any documentation of eligibility for benefits submitted by an employee or applicant for employment.\n(c) After taking all reasonable measures to find a contractor that complies with this section, as determined by the state agency, the requirements of this section may be waived under any of the following circumstances:\n(1) There is only one prospective contractor willing to enter into a specific contract with the state agency.\n(2) The contract is necessary to respond to an emergency, as determined by the state agency, that endangers the public health, welfare, or safety, or the contract is necessary for the provision of essential services, and no entity that complies with the requirements of this section capable of responding to the emergency is immediately available.\n(3) The requirements of this section violate, or are inconsistent with, the terms or conditions of a grant, subvention, or agreement, if the agency has made a good faith attempt to change the terms or conditions of any grant, subvention, or agreement to authorize application of this section.\n(4) The contractor is providing wholesale or bulk water, power, or natural gas, the conveyance or transmission of the same, or ancillary services, as required for ensuring reliable services in accordance with good utility practice, if the purchase of the same cannot practically be accomplished through the standard competitive bidding procedures and the contractor is not providing direct retail services to end users.\n(d) (1) A contractor shall not be deemed to discriminate in the provision of benefits if the contractor, in providing the benefits, pays the actual costs incurred in obtaining the benefit.\n(2) If a contractor is unable to provide a certain benefit, despite taking reasonable measures to do so, the contractor shall not be deemed to discriminate in the provision of benefits.\n(e) (1) Every contract subject to this chapter shall contain a statement by which the contractor certifies that the contractor is in compliance with this section.\n(2) The department or other contracting agency shall enforce this section pursuant to its existing enforcement powers.\n(3) (A) If a contractor falsely certifies that it is in compliance with this section, the contract with that contractor shall be subject to Article 9 (commencing with Section 10420), unless, within a time period specified by the department or other contracting agency, the contractor provides to the department or agency proof that it has complied, or is in the process of complying, with this section.\n(B) The application of the remedies or penalties contained in Article 9 (commencing with Section 10420) to a contract subject to this chapter shall not preclude the application of any existing remedies otherwise available to the department or other contracting agency under its existing enforcement powers.\n(f) Nothing in this section is intended to regulate the contracting practices of any local jurisdiction.\n(g) This section shall be construed so as not to conflict with applicable federal laws, rules, or regulations. In the event that a court or agency of competent jurisdiction holds that federal law, rule, or regulation invalidates any clause, sentence, paragraph, or section of this code or the application thereof to any person or circumstances, it is the intent of the state that the court or agency sever that clause, sentence, paragraph, or section so that the remainder of this section shall remain in effect.\nSEC. 2.\nSection 10295.35 of the Public Contract Code shall not be construed to create any new enforcement authority or responsibility in the Department of General Services or any other contracting agency.\nSEC. 3.\nNo reimbursement is required by this act pursuant to Section 6 of Article XIII\u2009B of the California Constitution because the only costs that may be incurred by a local agency or school district will be incurred because this act creates a new crime or infraction, eliminates a crime or infraction, or changes the penalty for a crime or infraction, within the meaning of Section 17556 of the Government Code, or changes the definition of a crime within the meaning of Section 6 of Article XIII\u2009B of the California Constitution.',
 'title': 'An act to add Section 10295.35 to the Public Contract Code, relating to public contracts.'}
```

여기서 다음 두 개의 필드를 사용하게 됩니다:

- `text`: 모델의 입력이 될 법안 텍스트입니다.
- `summary`: `text`의 간략한 버전으로 모델의 타겟이 됩니다.

## 전처리[[preprocess]]

다음으로 `text`와 `summary`를 처리하기 위한 T5 토크나이저를 가져옵니다:

```py
>>> from transformers import AutoTokenizer

>>> checkpoint = "google-t5/t5-small"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

생성하려는 전처리 함수는 아래 조건을 만족해야 합니다:

1. 입력 앞에 프롬프트를 붙여 T5가 요약 작업임을 인식할 수 있도록 합니다. 여러 NLP 작업을 수행할 수 있는 일부 모델은 특정 작업에 대한 프롬프트가 필요합니다.
2. 레이블을 토큰화할 때 `text_target` 인수를 사용합니다.
3. `max_length` 매개변수로 설정된 최대 길이를 넘지 않도록 긴 시퀀스를 잘라냅니다.

```py
>>> prefix = "summarize: "


>>> def preprocess_function(examples):
...     inputs = [prefix + doc for doc in examples["text"]]
...     model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

...     labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

...     model_inputs["labels"] = labels["input_ids"]
...     return model_inputs
```

전체 데이터셋에 전처리 함수를 적용하려면 🤗 Datasets의 [`~datasets.Dataset.map`] 메소드를 사용하세요.
`batched=True`로 설정하여 데이터셋의 여러 요소를 한 번에 처리하면 `map` 함수의 속도를 높일 수 있습니다.

```py
>>> tokenized_billsum = billsum.map(preprocess_function, batched=True)
```

이제 [`DataCollatorForSeq2Seq`]를 사용하여 예제 배치를 만드세요.
전체 데이터셋을 최대 길이로 패딩하는 것보다 배치마다 가장 긴 문장 길이에 맞춰 *동적 패딩*하는 것이 더 효율적입니다.

<frameworkcontent>
<pt>
```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
```
</pt>
<tf>
```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")
```
</tf>
</frameworkcontent>

## 평가[[evaluate]]

학습 중에 평가 지표를 포함하면 모델의 성능을 평가하는 데 도움이 되는 경우가 많습니다.
🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) 라이브러리를 사용하면 평가 방법을 빠르게 불러올 수 있습니다.
이 작업에서는 [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge) 평가 지표를 가져옵니다.
(평가 지표를 불러오고 계산하는 방법은 🤗 Evaluate [둘러보기](https://huggingface.co/docs/evaluate/a_quick_tour)를 참조하세요.)

```py
>>> import evaluate

>>> rouge = evaluate.load("rouge")
```

그런 다음 예측값과 레이블을 [`~evaluate.EvaluationModule.compute`]에 전달하여 ROUGE 지표를 계산하는 함수를 만듭니다:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
...     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
...     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

...     result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

...     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
...     result["gen_len"] = np.mean(prediction_lens)

...     return {k: round(v, 4) for k, v in result.items()}
```

이제 `compute_metrics` 함수를 사용할 준비가 되었으며, 학습을 설정할 때 이 함수로 되돌아올 것입니다.

## 학습[[train]]

<frameworkcontent>
<pt>
<Tip>

모델을 [`Trainer`]로 파인튜닝 하는 것이 익숙하지 않다면, [여기](../training#train-with-pytorch-trainer)에서 기본 튜토리얼을 확인해보세요!

</Tip>

이제 모델 학습을 시작할 준비가 되었습니다! [`AutoModelForSeq2SeqLM`]로 T5를 가져오세요:

```py
>>> from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

이제 세 단계만 남았습니다:

1. [`Seq2SeqTrainingArguments`]에서 학습 하이퍼파라미터를 정의하세요.
유일한 필수 매개변수는 모델을 저장할 위치를 지정하는 `output_dir`입니다.
`push_to_hub=True`를 설정하여 이 모델을 Hub에 푸시할 수 있습니다(모델을 업로드하려면 Hugging Face에 로그인해야 합니다.)
[`Trainer`]는 각 에폭이 끝날 때마다 ROUGE 지표를 평가하고 학습 체크포인트를 저장합니다.
2. 모델, 데이터셋, 토크나이저, 데이터 콜레이터 및 `compute_metrics` 함수와 함께 학습 인수를 [`Seq2SeqTrainer`]에 전달하세요.
3. [`~Trainer.train`]을 호출하여 모델을 파인튜닝하세요.

```py
>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="my_awesome_billsum_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     weight_decay=0.01,
...     save_total_limit=3,
...     num_train_epochs=4,
...     predict_with_generate=True,
...     fp16=True,
...     push_to_hub=True,
... )

>>> trainer = Seq2SeqTrainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_billsum["train"],
...     eval_dataset=tokenized_billsum["test"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

학습이 완료되면, 누구나 모델을 사용할 수 있도록 [`~transformers.Trainer.push_to_hub`] 메소드로 Hub에 공유합니다:

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

Keras로 모델 파인튜닝을 하는 것이 익숙하지 않다면, [여기](../training#train-a-tensorflow-model-with-keras)에서 기본적인 튜토리얼을 확인하세요!

</Tip>
TensorFlow에서 모델을 파인튜닝하려면, 먼저 옵티마이저, 학습률 스케줄 그리고 몇 가지 학습 하이퍼파라미터를 설정하세요:

```py
>>> from transformers import create_optimizer, AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

그런 다음 [`TFAutoModelForSeq2SeqLM`]을 사용하여 T5를 가져오세요:

```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

[`~transformers.TFPreTrainedModel.prepare_tf_dataset`]을 사용하여 데이터셋을 `tf.data.Dataset` 형식으로 변환하세요:

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_billsum["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_test_set = model.prepare_tf_dataset(
...     tokenized_billsum["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

[`compile`](https://keras.io/api/models/model_training_apis/#compile-method)을 사용하여 모델을 학습할 수 있도록 구성하세요:

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)
```

학습을 시작하기 전에 설정해야 할 마지막 두 가지는 예측에서 ROUGE 점수를 계산하고 모델을 Hub에 푸시하는 방법을 제공하는 것입니다.
두 작업 모두 [Keras callbacks](../main_classes/keras_callbacks)으로 수행할 수 있습니다.

[`~transformers.KerasMetricCallback`]에 `compute_metrics` 함수를 전달하세요:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

[`~transformers.PushToHubCallback`]에서 모델과 토크나이저를 푸시할 위치를 지정하세요:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_billsum_model",
...     tokenizer=tokenizer,
... )
```

그런 다음 콜백을 번들로 묶어줍니다:

```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

드디어 모델 학습을 시작할 준비가 되었습니다!
학습 및 검증 데이터셋, 에폭 수 및 콜백과 함께 [`fit`](https://keras.io/api/models/model_training_apis/#fit-method)을 호출하여 모델을 파인튜닝하세요.

```py
>>> model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=callbacks)
```

학습이 완료되면 모델이 자동으로 Hub에 업로드되어 누구나 사용할 수 있게 됩니다!
</tf>
</frameworkcontent>

<Tip>

요약을 위해 모델을 파인튜닝하는 방법에 대한 더 자세한 예제를 보려면 [PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb)
또는 [TensorFlow notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb)을 참고하세요.

</Tip>

## 추론[[inference]]

좋아요, 이제 모델을 파인튜닝했으니 추론에 사용할 수 있습니다!

요약할 텍스트를 작성해보세요. T5의 경우 작업에 따라 입력 앞에 접두사를 붙여야 합니다. 요약의 경우, 아래와 같은 접두사를 입력 앞에 붙여야 합니다:

```py
>>> text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
```

추론을 위해 파인튜닝한 모델을 시험해 보는 가장 간단한 방법은 [`pipeline`]에서 사용하는 것입니다.
모델을 사용하여 요약을 수행할 [`pipeline`]을 인스턴스화하고 텍스트를 전달하세요:

```py
>>> from transformers import pipeline

>>> summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
>>> summarizer(text)
[{"summary_text": "The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country."}]
```

원한다면 수동으로 다음과 같은 작업을 수행하여 [`pipeline`]의 결과와 동일한 결과를 얻을 수 있습니다:


<frameworkcontent>
<pt>
텍스트를 토크나이즈하고 `input_ids`를 PyTorch 텐서로 반환합니다:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
>>> inputs = tokenizer(text, return_tensors="pt").input_ids
```

요약문을 생성하려면 [`~generation.GenerationMixin.generate`] 메소드를 사용하세요.
텍스트 생성에 대한 다양한 전략과 생성을 제어하기 위한 매개변수에 대한 자세한 내용은 [텍스트 생성](../main_classes/text_generation) API를 참조하세요.

```py
>>> from transformers import AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
>>> outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
```

생성된 토큰 ID를 텍스트로 디코딩합니다:

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'the inflation reduction act lowers prescription drug costs, health care costs, and energy costs. it's the most aggressive action on tackling the climate crisis in american history. it will ask the ultra-wealthy and corporations to pay their fair share.'
```
</pt>
<tf>
텍스트를 토크나이즈하고 `input_ids`를 TensorFlow 텐서로 반환합니다:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
>>> inputs = tokenizer(text, return_tensors="tf").input_ids
```

요약문을 생성하려면 [`~transformers.generation_tf_utils.TFGenerationMixin.generate`] 메소드를 사용하세요.
텍스트 생성에 대한 다양한 전략과 생성을 제어하기 위한 매개변수에 대한 자세한 내용은 [텍스트 생성](../main_classes/text_generation) API를 참조하세요.

```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
>>> outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
```

생성된 토큰 ID를 텍스트로 디코딩합니다:

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'the inflation reduction act lowers prescription drug costs, health care costs, and energy costs. it's the most aggressive action on tackling the climate crisis in american history. it will ask the ultra-wealthy and corporations to pay their fair share.'
```
</tf>
</frameworkcontent>
