# Masked language modeling

[[open-in-colab]]

<Youtube id="mqElG5QJWUg"/>

يتم في نمذجة اللغة المقنعة التنبؤ برمز مقنع في تسلسل، ويمكن للنموذج الاهتمام بالرموز ثنائية الاتجاه. وهذا يعني أن النموذج لديه إمكانية الوصول الكامل إلى الرموز الموجودة على اليسار واليمين. تعد نمذجة اللغة المقنعة رائعة للمهام التي تتطلب فهمًا سياقيًا جيدًا لتسلسل كامل. BERT هو مثال على نموذج اللغة المقنع.

سيوضح هذا الدليل لك كيفية:

1. ضبط نموذج [DistilRoBERTa](https://huggingface.co/distilbert/distilroberta-base) الدقيق على مجموعة فرعية [r/askscience](https://www.reddit.com/r/askscience/) من مجموعة بيانات [ELI5](https://huggingface.co/datasets/eli5).
2. استخدام نموذجك الدقيق للاستنتاج.

<Tip>

لمشاهدة جميع التصميمات ونقاط التحقق المتوافقة مع هذه المهمة، نوصي بالتحقق من [صفحة المهمة](https://huggingface.co/tasks/fill-mask)

</Tip>

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate
```

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل نموذجك ومشاركته مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك لتسجيل الدخول:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات ELI5

ابدأ بتحميل أول 5000 مثال من مجموعة بيانات [ELI5-Category](https://huggingface.co/datasets/eli5_category) باستخدام مكتبة Datasets 🤗. سيتيح لك هذا فرصة التجربة والتأكد من أن كل شيء يعمل قبل قضاء المزيد من الوقت في التدريب على مجموعة البيانات الكاملة.

```py
>>> from datasets import load_dataset

>>> eli5 = load_dataset("eli5_category", split="train[:5000]")
```

قسِّم مجموعة البيانات إلى مجموعات فرعية للتدريب والاختبار باستخدام طريقة [`~datasets.Dataset.train_test_split`]:

```py
>>> eli5 = eli5.train_test_split(test_size=0.2)
```

ثم الق نظرة على مثال:
```py
>>> eli5 = eli5.train_test_split(test_size=0.2)
```

ثم الق نظرة على مثال:

```py
>>> eli5["train"][0]
{'q_id': '7h191n',
 'title': 'What does the tax bill that was passed today mean? How will it affect Americans in each tax bracket?',
 'selftext': '',
 'category': 'Economics',
 'subreddit': 'explainlikeimfive',
 'answers': {'a_id': ['dqnds8l', 'dqnd1jl', 'dqng3i1', 'dqnku5x'],
  'text': ["The tax bill is 500 pages long and there were a lot of changes still going on right to the end. It's not just an adjustment to the income tax brackets, it's a whole bunch of changes. As such there is no good answer to your question. The big take aways are: - Big reduction in corporate income tax rate will make large companies very happy. - Pass through rate change will make certain styles of business (law firms, hedge funds) extremely happy - Income tax changes are moderate, and are set to expire (though it's the kind of thing that might just always get re-applied without being made permanent) - People in high tax states (California, New York) lose out, and many of them will end up with their taxes raised.",
   'None yet. It has to be reconciled with a vastly different house bill and then passed again.',
   'Also: does this apply to 2017 taxes? Or does it start with 2018 taxes?',
   'This article explains both the House and senate bills, including the proposed changes to your income taxes based on your income level. URL_0'],
  'score': [21, 19, 5, 3],
  'text_urls': [[],
   [],
   [],
   ['https://www.investopedia.com/news/trumps-tax-reform-what-can-be-done/']]},
 'title_urls': ['url'],
 'selftext_urls': ['url']}
```

على الرغم من أن هذا قد يبدو كثيرًا، إلا أنك مهتم حقًا بحقل "النص". ما هو رائع في مهام نمذجة اللغة هو أنك لا تحتاج إلى علامات تصنيف (تُعرف أيضًا باسم المهمة غير الخاضعة للإشراف) لأن الكلمة التالية هي التصنيف.

## معالجة مسبقة

<Youtube id="8PmhEIXhBvI"/>

بالنسبة لنمذجة اللغة المقنعة، تتمثل الخطوة التالية في تحميل برنامج تشفير DistilRoBERTa لمعالجة حقل "النص" الفرعي:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
```

ستلاحظ من المثال أعلاه، أن حقل "النص" موجود بالفعل داخل "الإجابات". وهذا يعني أنك ستحتاج إلى استخراج حقل "النص" من هيكله المضمن باستخدام طريقة ["flatten"](https://huggingface.co/docs/datasets/process#flatten):
```py
>>> eli5 = eli5.flatten()
>>> eli5["train"][0]
{'q_id': '7h191n',
 'title': 'What does the tax bill that was passed today mean? How will it affect Americans in each tax bracket?',
 'selftext': '',
 'category': 'Economics',
 'subreddit': 'explainlikeimfive',
 'answers.a_id': ['dqnds8l', 'dqnd1jl', 'dqng3i1', 'dqnku5x'],
 'answers.text': ["The tax bill is 500 pages long and there were a lot of changes still going on right to the end. It's not just an adjustment to the income tax brackets, it's a whole bunch of changes. As such there is no good answer to your question. The big take aways are: - Big reduction in corporate income tax rate will make large companies very happy. - Pass through rate change will make certain styles of business (law firms, hedge funds) extremely happy - Income tax changes are moderate, and are set to expire (though it's the kind of thing that might just always get re-applied without being made permanent) - People in high tax states (California, New York) lose out, and many of them will end up with their taxes raised.",
  'None yet. It has to be reconciled with a vastly different house bill and then passed again.',
  'Also: does this apply to 2017 taxes? Or does it start with 2018 taxes?',
  'This article explains both the House and senate bills, including the proposed changes to your income taxes based on your income level. URL_0'],
 'answers.score': [21, 19, 5, 3],
 'answers.text_urls': [[],
  [],
  [],
  ['https://www.investopedia.com/news/trumps-tax-reform-what-can-be-done/']],
 'title_urls': ['url'],
 'selftext_urls': ['url']}
```

كل حقل فرعي هو الآن عمود منفصل كما هو موضح بالبادئة "الإجابات"، وحقل "النص" هو قائمة الآن. بدلاً من تشفير كل جملة بشكل منفصل، قم بتحويل القائمة إلى سلسلة بحيث يمكنك تشفيرها بشكل مشترك.

هذه هي دالة المعالجة المسبقة الأولى لدمج قائمة السلاسل لكل مثال وتشفير النتيجة:

```py
>>> def preprocess_function(examples):
...     return tokenizer([" ".join(x) for x in examples["answers.text"]])
```

لتطبيق دالة المعالجة المسبقة هذه على مجموعة البيانات بأكملها، استخدم طريقة [`~datasets.Dataset.map`] في مكتبة Datasets 🤗. يمكنك تسريع وظيفة "map" عن طريق تعيين "batched=True" لمعالجة عناصر متعددة من مجموعة البيانات في وقت واحد، وزيادة عدد العمليات باستخدام "num_proc". احذف أي أعمدة لا تحتاج إليها:

```py
>>> tokenized_eli5 = eli5.map(
...     preprocess_function,
...     batched=True,
...     num_proc=4,
...     remove_columns=eli5["train"].column_names,
... )
```

تحتوي مجموعة البيانات هذه على تسلسلات الرموز، ولكن بعضها أطول من طول الإدخال الأقصى للنموذج.

الآن يمكنك استخدام دالة المعالجة المسبقة الثانية ل:
- دمج جميع التسلسلات
- تقسيم التسلسلات المدمجة إلى قطع أقصر محددة بواسطة "block_size"، والتي يجب أن تكون أقصر من طول الإدخال الأقصى وقصيرة بدرجة كافية لذاكرة GPU. 

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
...     return result
```

قم بتطبيق دالة "group_texts" على مجموعة البيانات بأكملها:

```py
>>> lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
```
قم بتطبيق دالة "group_texts" على مجموعة البيانات بأكملها:

```py
>>> lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
```

الآن قم بإنشاء دفعة من الأمثلة باستخدام [`DataCollatorForLanguageModeling`]. من الأكثر كفاءة *حشو* الجمل ديناميكيًا إلى أطول طول في دفعة أثناء التجميع، بدلاً من حشو مجموعة البيانات بأكملها إلى الطول الأقصى.

<frameworkcontent>
<pt>

استخدم رمز نهاية التسلسل كرموز حشو وحدد "mlm_probability" لإخفاء الرموز عشوائيًا كلما قمت بالتنقل خلال البيانات:

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> tokenizer.pad_token = tokenizer.eos_token
>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
```
</pt>
<tf>

استخدم رمز نهاية التسلسل كرموز حشو وحدد "mlm_probability" لإخفاء الرموز عشوائيًا كلما قمت بالتنقل خلال البيانات:

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="tf")
```
</tf>
</frameworkcontent>

## تدريب

<frameworkcontent>
<pt>
<Tip>

إذا لم تكن على دراية بضبط نموذج باستخدام [`Trainer`، فراجع الدليل الأساسي هنا](../training#train-with-pytorch-trainer)

</Tip>

أنت الآن مستعد لبدء تدريب نموذجك! قم بتحميل DistilRoBERTa باستخدام [`AutoModelForMaskedLM`]:

```py
>>> from transformers import AutoModelForMaskedLM

>>> model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
```

في هذه المرحلة، هناك ثلاث خطوات فقط:

1. حدد فرط معلمات التدريب الخاصة بك في [`TrainingArguments`]. المعلمة المطلوبة الوحيدة هي "output_dir" والتي تحدد مكان حفظ نموذجك. يمكنك دفع هذا النموذج إلى Hub عن طريق تعيين "push_to_hub=True" (يجب أن تكون مسجلاً الدخول إلى Hugging Face لتحميل نموذجك).
2. مرر فرط معلمات التدريب إلى [`Trainer`] جنبًا إلى جنب مع النموذج ومجموعات البيانات ومجمع البيانات.
3. استدعاء [`~Trainer.train`] لضبط نموذجك.

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_eli5_mlm_model",
...     eval_strategy="epoch"،
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

بمجرد اكتمال التدريب، استخدم طريقة [`~transformers.Trainer.evaluate`] لتقييم نموذجك والحصول على احتمالية حدوثه:

```py
>>> import math

>>> eval_results = trainer.evaluate()
>>> print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
Perplexity: 8.76
```

ثم شارك نموذجك على Hub باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
<Tip>

إذا لم تكن على دراية بضبط نموذج باستخدام Keras، فراجع الدليل الأساسي هنا [../training#train-a-tensorflow-model-with-keras]!

</Tip>
لضبط نموذج في TensorFlow، ابدأ بإعداد دالة محسن ومعدل تعلم وجدول زمني ومفرط معلمات التدريب:

```py
>>> from transformers import create_optimizer, AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

ثم يمكنك تحميل DistilRoBERTa باستخدام [`TFAutoModelForMaskedLM`]:

```py
>>> from transformers import TFAutoModelForMaskedLM

>>> model = TFAutoModelForMaskedLM. from_pretrained("distilbert/distilroberta-base")
```

قم بتحويل مجموعات البيانات الخاصة بك إلى تنسيق "tf.data.Dataset" باستخدام [`~transformers.TFPreTrainedModel.prepare_tf_dataset`]:

```py
>>> tf_train_set = model.prepare_tf_dataset(
...     lm_dataset["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```py
>>> tf_train_set = model.prepare_tf_dataset(
...     lm_dataset["train"],
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_test_set = model.prepare_tf_dataset(
...     lm_dataset["test"],
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

قم بتكوين النموذج للتدريب باستخدام ["compile"](https://keras.io/api/models/model_training_apis/#compile-method). لاحظ أن جميع نماذج Transformers تحتوي على دالة خسارة افتراضية ذات صلة بالمهمة، لذلك لا تحتاج إلى تحديد واحدة ما لم ترغب في ذلك:

```py
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer) # لا توجد وسيطة دالة الخسارة!
```

يمكن القيام بذلك عن طريق تحديد المكان الذي ستدفع فيه نموذجك وبرنامج الترميز الخاص بك في [`~transformers.PushToHubCallback`]:

```py
>>> from transformers.keras_callbacks import PushToHubCallback

>>> callback = PushToHubCallback(
...     output_dir="my_awesome_eli5_mlm_model"،
...     tokenizer=tokenizer,
... )
```

أخيرًا، أنت مستعد لبدء تدريب نموذجك! استدعاء ["fit"](https://keras.io/api/models/model_training_apis/#fit-method) مع مجموعات البيانات التدريبية والتحقق من صحتها، وعدد العصور