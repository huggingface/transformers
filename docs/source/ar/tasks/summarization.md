# تلخيص

[[open-in-colab]]

<Youtube id="yHnr5Dk2zCI"/>

يخلق التلخيص نسخة أقصر من المستند أو المقال الذي يلتقط جميع المعلومات المهمة. إلى جانب الترجمة، يعد التلخيص مثال آخر على المهمة التي يمكن صياغتها كمهمة تسلسل إلى تسلسل. يمكن أن يكون التلخيص:

- استخراجي: استخراج أهم المعلومات من المستند.
- مجردة: قم بتوليد نص جديد يلخص أهم المعلومات.

سيوضح هذا الدليل كيفية:

1. ضبط دقيق [T5](https://huggingface.co/google-t5/t5-small) على مجموعة فرعية من فاتورة ولاية كاليفورنيا من [BillSum](https://huggingface.co/datasets/billsum) مجموعة البيانات للتلخيص المجرد.
2. استخدام نموذج ضبط دقيق الخاص بك للاستدلال.

<Tip>

لمعرفة جميع التصميمات ونقاط التحقق المتوافقة مع هذه المهمة، نوصي بالتحقق من [صفحة المهمة](https://huggingface.co/tasks/summarization)

</Tip>

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate rouge_score
```

نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل نموذجك ومشاركته مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات BillSum

ابدأ بتحميل مجموعة البيانات الفرعية الأصغر من فاتورة ولاية كاليفورنيا من مجموعة بيانات BillSum من مكتبة Datasets 🤗:

```py
>>> from datasets import load_dataset

>>> billsum = load_dataset("billsum", split="ca_test")
```

قسّم مجموعة البيانات إلى مجموعات تدريب واختبار باستخدام طريقة [`~datasets.Dataset.train_test_split`] :

```py
>>> billsum = billsum.train_test_split(test_size=0.2)
```

ثم الق نظرة على مثال:
```py
>>> billsum["train"][0]
{'summary': 'Existing law authorizes state agencies to enter into contracts for the acquisition of goods or services upon approval by the Department of General Services. Existing law sets forth various requirements and prohibitions for those contracts, including, but not limited to, a prohibition on entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between spouses and domestic partners or same-sex and different-sex couples in the provision of benefits. Existing law provides that a contract entered into in violation of those requirements and prohibitions is void and authorizes the state or any person acting on behalf of the state to bring a civil action seeking a determination that a contract is in violation and therefore void. Under existing law, a willful violation of those requirements and prohibitions is a misdemeanor.\nThis bill would also prohibit a state agency from entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between employees on the basis of gender identity in the provision of benefits, as specified. By expanding the scope of a crime, this bill would impose a state-mandated local program.\nThe California Constitution requires the state to reimburse local agencies and school districts for certain costs mandated by the state. Statutory provisions establish procedures for making that reimbursement.\nThis bill would provide that no reimbursement is required by this act for a specified reason.',
 'text': 'The people of the State of California do enact as follows:\n\n\nSECTION 1.\nSection 10295.35 is added to the Public Contract Code, to read:\n10295.35.\n(a) (1) Notwithstanding any other law, a state agency shall not enter into any contract for the acquisition of goods or services in the amount of one hundred thousand dollars ($100,000) or more with a contractor that, in the provision of benefits, discriminates between employees on the basis of an employee’s or dependent’s actual or perceived gender identity, including, but not limited to, the employee’s or dependent’s identification as transgender.\n(2) For purposes of this section, “contract” includes contracts with a cumulative amount of one hundred thousand dollars ($100,000) or more per contractor in each fiscal year.\n(3) For purposes of this section, an employee health plan is discriminatory if the plan is not consistent with Section 1365.5 of the Health and Safety Code and Section 10140 of the Insurance Code.\n(4) The requirements of this section shall apply only to those portions of a contractor’s operations that occur under any of the following conditions:\n(A) Within the state.\n(B) On real property outside the state if the property is owned by the state or if the state has a right to occupy the property, and if the contractor’s presence at that location is connected to a contract with the state.\n(C) Elsewhere in the United States where work related to a state contract is being performed.\n(b) Contractors shall treat as confidential, to the maximum extent allowed by law or by the requirement of the contractor’s insurance provider, any request by an employee or applicant for employment benefits or any documentation of eligibility for benefits submitted by an employee or applicant for employment.\n(c) After taking all reasonable measures to find a contractor that complies with this section, as determined by the state agency, the requirements of this section may be waived under any of the following circumstances:\n(1) There is only one prospective contractor willing to enter into a specific contract with the state agency.\n(2) The contract is necessary to respond to an emergency, as determined by the state agency, that endangers the public health, welfare, or safety, or the contract is necessary for the provision of essential services, and no entity that complies with the requirements of this section capable of responding to the emergency is immediately available.\n(3) The requirements of this section violate, or are inconsistent with, the terms or conditions of a grant, subvention, or agreement, if the agency has made a good faith attempt to change the terms or conditions of any grant, subvention, or agreement to authorize application of this section.\n(4) The contractor is providing wholesale or bulk water, power, or natural gas, the conveyance or transmission of the same, or ancillary services, as required for ensuring reliable services in accordance with good utility practice, if the purchase of the same cannot practically be accomplished through the standard competitive bidding procedures and the contractor is not providing direct retail services to end users.\n(d) (1) A contractor shall not be deemed to discriminate in the provision of benefits if the contractor, in providing the benefits, pays the actual costs incurred in obtaining the benefit.\n(2) If a contractor is unable to provide a certain benefit, despite taking reasonable measures to do so, the contractor shall not be deemed to discriminate in the provision of benefits.\n(e) (1) Every contract subject to this chapter shall contain a statement by which the contractor certifies that the contractor is in compliance with this section.\n(2) The department or other contracting agency shall enforce this section pursuant to its existing enforcement powers.\n(3) (A) If a contractor falsely certifies that it is in compliance with this section, the contract with that contractor shall be subject to Article 9 (commencing with Section 10420), unless, within a time period specified by the department or other contracting agency, the contractor provides to the department or agency proof that it has complied, or is in the process of complying, with this section.\n(B) The application of the remedies or penalties contained in Article 9 (commencing with Section 10420) to a contract subject to this chapter shall not preclude the application of any existing remedies otherwise available to the department or other contracting agency under its existing enforcement powers.\n(f) Nothing in this section is intended to regulate the contracting practices of any local jurisdiction.\n(g) This section shall be construed so as not to conflict with applicable federal laws, rules, or regulations. In the event that a court or agency of competent jurisdiction holds that federal law, rule, or regulation invalidates any clause, sentence, paragraph, or section of this code or the application thereof to any person or circumstances, it is the intent of the state that the court or agency sever that clause, sentence, paragraph, or section so that the remainder of this section shall remain in effect.\nSEC. 2.\nSection 10295.35 of the Public Contract Code shall not be construed to create any new enforcement authority or responsibility in the Department of General Services or any other contracting agency.\nSEC. 3.\nNo reimbursement is required by this act pursuant to Section 6 of Article XIII\u2009B of the California Constitution because the only costs that may be incurred by a local agency or school district will be incurred because this act creates a new crime or infraction, eliminates a crime or infraction, or changes the penalty for a crime or infraction, within the meaning of Section 17556 of the Government Code, or changes the definition of a crime within the meaning of Section 6 of Article XIII\u2009B of the California Constitution.',
 'title': 'An act to add Section 10295.35 to the Public Contract Code, relating to public contracts.'}
```
هناك حقلان تريد استخدامهما:

- `النص`: نص الفاتورة الذي سيكون المدخل للنموذج.
- `ملخص`: نسخة مختصرة من `النص` الذي سيكون هدف النموذج.

## معالجة مسبقة

الخطوة التالية هي تحميل برنامج Tokenizer T5 لمعالجة `النص` و `الملخص`:

```py
>>> from transformers import AutoTokenizer

>>> checkpoint = "google-t5/t5-small"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

تحتاج دالة المعالجة المسبقة التي تريد إنشاؤها إلى:

1. إضافة بادئة إلى الإدخال باستخدام موجه حتى يعرف T5 أن هذه مهمة تلخيص. تتطلب بعض النماذج القادرة على مهام NLP متعددة المطالبة بمهام محددة.
2. استخدام حجة `text_target` الكلمة عند توكين التصنيفات.
3. اقطع التسلسلات بحيث لا يكون طولها أطول من الطول الأقصى المحدد بواسطة معلمة `max_length` .

```py
>>> prefix = "summarize: "


>>> def preprocess_function(examples):
...     inputs = [prefix + doc for doc in examples["text"]]
...     model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

...     labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

...     model_inputs["labels"] = labels["input_ids"]
...     return model_inputs
```

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم طريقة [`~datasets.Dataset.map`] في Datasets 🤗 . يمكنك تسريع وظيفة `map` عن طريق تعيين `batched=True` لمعالجة عناصر متعددة من مجموعة البيانات في وقت واحد:

```py
>>> tokenized_billsum = billsum.map(preprocess_function, batched=True)
```

الآن قم بإنشاء دفعة من الأمثلة باستخدام [`DataCollatorForSeq2Seq`]. من الأكثر كفاءة *الحشو الديناميكي* الجمل إلى أطول طول في دفعة أثناء التجليد، بدلاً من حشو مجموعة البيانات بأكملها إلى الطول الأقصى.

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

## تقييم

غالبًا ما يكون تضمين مقياس أثناء التدريب مفيدًا لتقييم أداء نموذجك. يمكنك تحميل طريقة تقييم بسرعة باستخدام مكتبة 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) . بالنسبة لهذه المهمة، قم بتحميل مقياس [ROUGE](https://huggingface.co/spaces/evaluate-metric/rouge) (راجع دليل المستخدم لـ 🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour) لمعرفة المزيد حول كيفية تحميل وحساب مقياس):

```py
>>> import evaluate

>>> rouge = evaluate.load("rouge")
```

ثم قم بإنشاء دالة تمرر تنبؤاتك وتصنيفاتك إلى [`~evaluate.EvaluationModule.compute`] لحساب مقياس ROUGE:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions، labels = eval_pred
...     decoded_preds = tokenizer.batch_decode(predictions، skip_special_tokens=True)
...     labels = np.where (labels! = -100، labels، tokenizer.pad_token_id)
...     decoded_labels = tokenizer.batch_decode(labels، skip_special_tokens=True)

...     النتيجة = rouge.compute(predictions=decoded_preds، references=decoded_labels، use_stemmer=True)

...     prediction_lens = [np.count_nonzero (pred! = tokenizer.pad_token_id) for pred in predictions]
...     result ["gen_len"] = np.mean (prediction_lens)

...     return {k: round (v، 4) for k، v in result.items()}
```

وظيفة `compute_metrics` الخاصة بك جاهزة الآن، وستعود إليها عند إعداد التدريب الخاص بك.

## تدريب

<frameworkcontent>
<pt>
<Tip>

إذا لم تكن على دراية بضبط نموذج باستخدام [`Trainer` ]، فراجع البرنامج التعليمي الأساسي [هنا](../training#train-with-pytorch-trainer) !

</Tip>

أنت مستعد الآن لبدء تدريب نموذجك! قم بتحميل T5 باستخدام [`AutoModelForSeq2SeqLM`]:

```py
>>> from transformers import AutoModelForSeq2SeqLM، Seq2SeqTrainingArguments، Seq2SeqTrainer
</Tip>

أنت مستعد الآن لبدء تدريب نموذجك! قم بتحميل T5 باستخدام [`AutoModelForSeq2SeqLM`]:

```py
>>> from transformers import AutoModelForSeq2SeqLM، Seq2SeqTrainingArguments، Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

في هذه المرحلة، هناك ثلاث خطوات فقط:

1. حدد فرط معلمات التدريب الخاصة بك في [`Seq2SeqTrainingArguments`]. المعلمة الوحيدة المطلوبة هي `output_dir` التي تحدد مكان حفظ نموذجك. ستقوم بالدفع إلى Hub عن طريق تعيين `push_to_hub=True` (يجب أن تكون مسجلاً الدخول إلى Hugging Face لتحميل نموذجك). في نهاية كل حقبة، سيقوم [`Trainer`] بتقييم مقياس ROUGE وحفظ نقطة التدريب.
2. قم بتمرير الحجج التدريبية إلى [`Seq2SeqTrainer`] جنبًا إلى جنب مع النموذج ومجموعة البيانات والمحلل اللغوي ومجلد البيانات ووظيفة `compute_metrics` .
3. استدعاء [`~Trainer.train`] لضبط نموذجك بدقة.

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
بمجرد الانتهاء من التدريب، شارك نموذجك على المنصة باستخدام طريقة [~transformers.Trainer.push_to_hub`] بحيث يمكن للجميع استخدام نموذجك:

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

إذا لم تكن معتادًا على ضبط نموذج باستخدام Keras، فراجع البرنامج التعليمي الأساسي [هنا](../training#train-a-tensorflow-model-with-keras)!

</Tip>
لضبط نموذج في TensorFlow، ابدأ بإعداد دالة المحسن ومعدل التعلم وجدول التدريب وبعض فرط المعلمات:

```py
>>> from transformers import create_optimizer, AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

بعد ذلك، يمكنك تحميل T5 باستخدام [`TFAutoModelForSeq2SeqLM`]:

```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

قم بتحويل مجموعات البيانات الخاصة بك إلى تنسيق `tf.data.Dataset` باستخدام [`~transformers.TFPreTrainedModel.prepare_tf_dataset`]:

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

قم بتهيئة النموذج للتدريب باستخدام [`compile`](https://keras.io/api/models/model_training_apis/#compile-method). لاحظ أن جميع نماذج Transformers لديها دالة خسارة افتراضية ذات صلة بالمهمة، لذلك لا تحتاج إلى تحديد واحدة ما لم ترغب في ذلك:

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)  # لا توجد حجة الخسارة!
```

الشيئان الأخيران اللذان يجب إعدادهما قبل بدء التدريب هما حساب درجة ROUGE من التوقعات، وتوفير طريقة لدفع نموذجك إلى المنصة. يتم تنفيذ كلاهما باستخدام [Keras callbacks](../main_classes/keras_callbacks).

مرر دالة `compute_metrics` الخاصة بك إلى [`~transformers.KerasMetricCallback`]:

```py
>>> from transformers.keras_callbacks import KerasMetricCallback

>>> metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
```

حدد أين تريد دفع نموذجك ومحولك في [`~transformers.PushToHubCallback`]:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="my_awesome_billsum_model",
...     tokenizer=tokenizer,
... )
```

ثم قم بتجميع مكالمات الإرجاع الخاصة بك معًا:

```py
>>> callbacks = [metric_callback, push_to_hub_callback]
```

أخيرًا، أنت مستعد لبدء تدريب نموذجك! اتصل بـ [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) باستخدام مجموعات البيانات التدريبية والتحقق من صحتها، وعدد العصور، ومكالمات الإرجاع الخاصة بك لضبط النموذج:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=callbacks)
```

بمجرد الانتهاء من التدريب، يتم تحميل نموذجك تلقائيًا إلى المنصة حتى يتمكن الجميع من استخدامه!
</tf>
</frameworkcontent>

<Tip>

لمثال أكثر تفصيلاً حول كيفية ضبط نموذج للتلخيص، راجع الدفتر المقابل
[دفتر ملاحظات PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb)
أو [دفتر ملاحظات TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb).

</Tip>

## الاستنتاج

رائع، الآن بعد أن ضبطت نموذجًا، يمكنك استخدامه للاستنتاج!

فكر في بعض النصوص التي تريد تلخيصها. بالنسبة إلى T5، تحتاج إلى إضافة بادئة إلى إدخالك اعتمادًا على المهمة التي تعمل عليها. للتلخيص، يجب إضافة بادئة إلى إدخالك كما هو موضح أدناه:

```py
>>> text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
```

أبسط طريقة لتجربة نموذجك المضبوط للاستنتاج هي استخدامه في [`pipeline`]. قم بتنفيذ `pipeline` للتلخيص باستخدام نموذجك، ومرر نصك إليه:

```py
>>> from transformers import pipeline

>>> summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
>>> summarizer(text)
[{"summary_text": "The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country."}]
```

يمكنك أيضًا إعادة إنتاج نتائج `pipeline` يدويًا إذا كنت تريد ذلك:


<frameworkcontent>
<pt>
قم برمز النص وإرجاع `input_ids` كرموز PyTorch:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
>>> inputs = tokenizer(text, return_tensors="pt").input_ids
```

استخدم طريقة [`~generation.GenerationMixin.generate`] لإنشاء الملخص. لمزيد من التفاصيل حول استراتيجيات إنشاء النصوص المختلفة ومعلمات التحكم في الإنشاء، راجع واجهة برمجة التطبيقات [Text Generation](../main_classes/text_generation).

```py
>>> from transformers import AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
>>> outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
```

فك تشفير رموز المعرفات المولدة مرة أخرى إلى نص:

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'the inflation reduction act lowers prescription drug costs, health care costs, and energy costs. it's the most aggressive action on tackling the climate crisis in american history. it will ask the ultra-wealthy and corporations to pay their fair share.'
```
</pt>
<tf>
قم برمز النص وإرجاع `input_ids` كرموز TensorFlow:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
>>> inputs = tokenizer(text, return_tensors="tf").input_ids
```

استخدم طريقة [`~transformers.generation_tf_utils.TFGenerationMixin.generate`] لإنشاء الملخص. لمزيد من التفاصيل حول استراتيجيات إنشاء النصوص المختلفة ومعلمات التحكم في الإنشاء، راجع واجهة برمجة التطبيقات [Text Generation](../main_classes/text_generation).

```py
>>> from transformers import TFAutoModelForSeq2SeqLM

>>> model = TFAutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
>>> outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
```

فك تشفير رموز المعرفات المولدة مرة أخرى إلى نص:

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'the inflation reduction act lowers prescription drug costs, health care costs, and energy costs. it's the most aggressive action on tackling the climate crisis in american history. it will ask the ultra-wealthy and corporations to pay their fair share.'
```
</tf>
</frameworkcontent>