# الإجابة على الأسئلة

[[open-in-colab]]

<Youtube id="ajPx5LwJD-I"/>

تعيد مهام الإجابة على الأسئلة إجابةً على سؤال معين. إذا سبق لك أن سألت مساعدًا افتراضيًا مثل Alexa أو Siri أو Google عن حالة الطقس، فقد استخدمت بالفعل نموذج إجابة على الأسئلة. هناك نوعان شائعان من مهام الإجابة على الأسئلة:

- الاستخراجية: استخراج الإجابة من السياق المحدد.
- التلخيصية: إنشاء إجابة من السياق الذي يجيب على السؤال بشكل صحيح.

سيوضح هذا الدليل كيفية:

1. ضبط نموذج [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) الدقيق على مجموعة بيانات [SQuAD](https://huggingface.co/datasets/squad) للإجابة على الأسئلة الاستخراجية.
2. استخدام نموذجك المضبوط دقيقًا للاستنتاج.

<Tip>

لمشاهدة جميع البنيات ونقاط التفتيش المتوافقة مع هذه المهمة، نوصي بالتحقق من [صفحة المهمة](https://huggingface.co/tasks/question-answering)

</Tip>

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate
```

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل نموذجك ومشاركته مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات SQuAD

ابدأ بتحميل جزء فرعي أصغر من مجموعة بيانات SQuAD من مكتبة Datasets 🤗. سيتيح لك هذا فرصة التجربة والتأكد من أن كل شيء يعمل قبل قضاء المزيد من الوقت في التدريب على مجموعة البيانات الكاملة.

```py
>>> from datasets import load_dataset

>>> squad = load_dataset("squad", split="train[:5000]")
```

قسِّم مجموعة البيانات إلى مجموعتين فرعيتين للتدريب والاختبار باستخدام طريقة [`~datasets.Dataset.train_test_split`]:

```py
>>> squad = squad.train_test_split(test_size=0.2)
```

ثم الق نظرة على مثال:

```py
>>> squad["train"][0]
{'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
 'id': '5733be284776f41900661182',
 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
 'title': 'University_of_Notre_Dame'
}
```

هناك العديد من الحقول المهمة هنا:

- `answers`: موقع بداية الإجابة ورقمها.
- `context`: معلومات أساسية يجب على النموذج استخراج الإجابة منها.
- `question`: السؤال الذي يجب أن يجيب عليه النموذج.

## معالجة مسبقة

<Youtube id="qgaM0weJHpA"/>

الخطوة التالية هي تحميل معالج نموذج DistilBERT لمعالجة حقلي `question` و`context`:
## معالجة مسبقة

<Youtube id="qgaM0weJHpA"/>

الخطوة التالية هي تحميل معالج نموذج DistilBERT لمعالجة حقلي `question` و`context`:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

هناك العديد من خطوات المعالجة المسبقة الخاصة بمهام الإجابة على الأسئلة والتي يجب أن تكون على دراية بها:

1. قد تحتوي بعض الأمثلة في مجموعة البيانات على سياق `context` طويل جدًا يتجاوز الطول الأقصى للمدخلات للنموذج. للتعامل مع التسلسلات الأطول، قم بتقطيع السياق `context` فقط عن طريق تعيين `truncation="only_second"`.
2. بعد ذلك، قم بتعيين مواقع البداية والنهاية للإجابة على السياق الأصلي عن طريق تعيين
   `return_offset_mapping=True`.
3. باستخدام الخريطة، يمكنك الآن العثور على رموز البداية والنهاية للإجابة. استخدم طريقة [`~tokenizers.Encoding.sequence_ids`] لمعرفة أي جزء من الإزاحة يقابل `question` وأي جزء يقابل `context`.

فيما يلي كيفية إنشاء دالة لتقطيع السياق `context` ورسم خريطة لمواقع البداية والنهاية للإجابة على السياق:

```py
>>> def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
```

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم وظيفة [`~datasets.Dataset.map`] في مكتبة 🤗 Datasets. يمكنك تسريع وظيفة `map` عن طريق تعيين `batched=True` لمعالجة عناصر متعددة من مجموعة البيانات في وقت واحد. احذف أي أعمدة لا تحتاجها:

```py
>>> tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
```

الآن قم بإنشاء دفعة من الأمثلة باستخدام [`DefaultDataCollator`]. على عكس برامج الجمع الأخرى في مكتبة 🤗 Transformers، فإن [`DefaultDataCollator`] لا تطبق أي معالجة مسبقة إضافية مثل التقسيم.

<frameworkcontent>
<pt>
    
```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

</pt>
<tf>
    
```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")
```

</tf>
</frameworkcontent>


## تدريب

<frameworkcontent>
<pt>
<Tip>

إذا لم تكن على دراية بضبط نموذج دقيق باستخدام [`Trainer`، فراجع البرنامج التعليمي الأساسي [هنا](../training#train-with-pytorch-trainer)!

</Tip>

أنت الآن على استعداد لبدء تدريب نموذجك! قم بتحميل نموذج DistilBERT باستخدام [`AutoModelForQuestionAnswering`]:

```py
>>> from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

>>> model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```

في هذه المرحلة، هناك ثلاث خطوات فقط متبقية:

1. حدد فرط معلمات التدريب الخاصة بك في [`TrainingArguments`]. المعلمة المطلوبة الوحيدة هي `output_dir` التي تحدد مكان حفظ نموذجك. يمكنك دفع هذا النموذج إلى Hub عن طريق تعيين `push_to_hub=True` (يجب أن تكون مسجل الدخول إلى Hugging Face لتحميل نموذجك).
2. مرر فرط معلمات التدريب إلى [`Trainer`] جنبًا إلى جنب مع النموذج ومجموعة البيانات والمعالج وبرنامج الجمع.
3. استدعاء [`~Trainer.train`] لضبط نموذجك دقيقًا.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_qa_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_
size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_squad["train"],
...     eval_dataset=tokenized_squad["test"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب، شارك نموذجك على Hub باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

إذا لم تكن معتادًا على ضبط نموذج باستخدام Keras، فراجع البرنامج التعليمي الأساسي [هنا](../training#train-a-tensorflow-model-with-keras)!

</Tip>
لضبط نموذج في TensorFlow، ابدأ بإعداد دالة محسن، وجدول معدل التعلم، وبعض فرط معلمات التدريب:

```py
>>> from transformers import create_optimizer

>>> batch_size = 16
>>> num_epochs = 2
>>> total_train_steps = (len(tokenized_squad["train"]) // batch_size) * num_epochs
>>> optimizer, schedule = create_optimizer(
...     init_lr=2e-5,
...     num_warmup_steps=0,
...     num_train_steps=total_train_steps,
... )
```

بعد ذلك، يمكنك تحميل DistilBERT باستخدام [`TFAutoModelForQuestionAnswering`]:

```py
>>> from transformers import TFAutoModelForQuestionAnswering

>>> model = TFAutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```

قم بتحويل مجموعات البيانات الخاصة بك إلى تنسيق `tf.data.Dataset` باستخدام [`~transformers.TFPreTrainedModel.prepare_tf_dataset`]:

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     tokenized_squad["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_validation_set = model.prepare_tf_dataset(
...     tokenized_squad["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

قم بتهيئة النموذج للتدريب باستخدام [`compile`](https://keras.io/api/models/model_training_apis/#compile-method):

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer)
```

الشيء الأخير الذي يجب إعداده قبل بدء التدريب هو توفير طريقة لدفع نموذجك إلى Hub. يمكن القيام بذلك عن طريق تحديد المكان الذي ستدفع فيه نموذجك ومصنف الرموز في [`~transformers.PushToHubCallback`]:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> callback = PushToHubCallback(
...     output_dir="my_awesome_qa_model",
...     tokenizer=tokenizer,
... )
```

أخيرًا، أنت مستعد لبدء تدريب نموذجك! استدعاء [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) مع مجموعات البيانات الخاصة بالتدريب والتحقق من الصحة، وعدد العصور، واستدعاء الإرجاع الخاص بك لضبط النموذج:

```py
>>> model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=[callback])
```
بمجرد اكتمال التدريب، يتم تحميل نموذجك تلقائيًا إلى Hub حتى يتمكن الجميع من استخدامه!
</tf>
</frameworkcontent>

<Tip>

لمثال أكثر عمقًا حول كيفية ضبط نموذج للإجابة على الأسئلة، راجع الدفتر الملاحظات المقابل
[دفتر ملاحظات PyTorch](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb)
أو [دفتر ملاحظات TensorFlow](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb).

</Tip>

## تقييم

يتطلب التقييم للإجابة على الأسئلة قدرًا كبيرًا من ما بعد المعالجة. لتجنب استغراق الكثير من وقتك، يتخطى هذا الدليل خطوة التقييم. لا يزال [`Trainer`] يحسب خسارة التقييم أثناء التدريب حتى لا تكون في الظلام تمامًا بشأن أداء نموذجك.

إذا كان لديك المزيد من الوقت وأنت مهتم بكيفية تقييم نموذجك للإجابة على الأسئلة، فراجع الفصل [الإجابة على الأسئلة](https://huggingface.co/course/chapter7/7?fw=pt#post-processing) من دورة 🤗 Hugging Face!

## الاستنتاج

رائع، الآن بعد أن قمت بضبط نموذج، يمكنك استخدامه للاستنتاج!

فكر في سؤال وبعض السياقات التي تود أن يتنبأ بها النموذج:

```py
>>> question = "How many programming languages does BLOOM support?"
>>> context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
```

أبسط طريقة لتجربة نموذجك المضبوط للاستنتاج هي استخدامه في [`pipeline`]: قم بتنفيذ `pipeline` للإجابة على الأسئلة باستخدام نموذجك، ومرر نصك إليه:

```py
>>> from transformers import pipeline

>>> question_answerer = pipeline("question-answering", model="my_awesome_qa_model")
>>> question_answerer(question=question, context=context)
{'score': 0.2058267742395401،
 'start': 10،
 'end': 95،
 'answer': '176 billion parameters and can generate text in 46 languages natural languages and 13'}
```

يمكنك أيضًا محاكاة نتائج `pipeline` يدويًا إذا كنت ترغب في ذلك:

<frameworkcontent>
<pt>
قم بتوكينز النص وإرجاع tensers PyTorch:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
>>> inputs = tokenizer(question, context, return_tensors="pt")
```

مرر المدخلات الخاصة بك إلى النموذج وإرجاع "logits":

```py
>>> import torch
>>> from transformers import AutoModelForQuestionAnswering

>>> model = AutoModelForQuestionAnswering.from_pretrained("my_awesome_qa_model")
>>> with torch.no_grad():
...     outputs = model(**inputs)
```

احصل على أعلى احتمال من إخراج النموذج لمواضع البداية والنهاية:

```py
>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()
```

فك تشفير الرموز المتنبأ بها للحصول على الإجابة:

```py
>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
'176 billion parameters and can generate text in 46 languages natural languages and 13'
```
</pt>
<tf>
قم بتوكينز النص وإرجاع tensers TensorFlow:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
>>> inputs = tokenizer(question, text, return_tensors="tf")
```

مرر المدخلات الخاصة بك إلى النموذج وإرجاع "logits":

```py
>>> from transformers import TFAutoModelForQuestionAnswering

>>> model = TFAutoModelForQuestionAnswering.from_pretrained("my_awesome_qa_model")
>>> outputs = model(**inputs)
```

احصل على أعلى احتمال من إخراج النموذج لمواضع البداية والنهاية:

```py
>>> answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
>>> answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])
```

فك تشفير الرموز المتنبأ بها للحصول على الإجابة:

```py
>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
'176 billion parameters and can generate text in 46 languages natural languages and 13'
```
</tf>
</frameworkcontent>