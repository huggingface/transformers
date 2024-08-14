# ضبط نموذج مُدرب مسبقًا

هناك فوائد كبيرة لاستخدام نموذج مُدرب مسبقًا. فهو يقلل من تكاليف الحوسبة، وبصمتك الكربونية، ويتيح لك استخدام أحدث النماذج دون الحاجة إلى تدريبها من الصفر. توفر مكتبة 🤗 Transformers إمكانية الوصول إلى آلاف النماذج المُدربة مسبقًا لمجموعة واسعة من المهام. عندما تستخدم نموذجًا مُدربًا مسبقًا، فإنك تقوم بتدريبه على مجموعة بيانات خاصة بمهمتك. يُعرف ذلك بالضبط الدقيق، وهي تقنية تدريب قوية للغاية. في هذا البرنامج التعليمي، سوف تقوم بضبط نموذج مُدرب مسبقًا باستخدام إطار عمل للتعلم العميق من اختيارك:

* ضبط نموذج مُدرب مسبقًا باستخدام 🤗 Transformers [`Trainer`].
* ضبط نموذج مُدرب مسبقًا في TensorFlow باستخدام Keras.
* ضبط نموذج مُدرب مسبقًا في PyTorch الأصلي.

<a id='data-processing'></a>

## إعداد مجموعة بيانات

قبل أن تتمكن من ضبط نموذج مُدرب مسبقًا، قم بتنزيل مجموعة بيانات وإعدادها للتدريب. أظهر البرنامج التعليمي السابق كيفية معالجة البيانات للتدريب، والآن لديك الفرصة لاختبار تلك المهارات!

ابدأ بتحميل مجموعة بيانات [Yelp Reviews](https://huggingface.co/datasets/yelp_review_full):

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("yelp_review_full")
>>> dataset["train"][100]
{'label': 0,
 'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}
```

كما تعلم الآن، تحتاج إلى أداة تعلم علامات الترقيم لمعالجة النص وتضمين استراتيجية للتعامل مع أي أطوال متسلسلة متغيرة. لمعالجة مجموعة البيانات الخاصة بك في خطوة واحدة، استخدم طريقة 🤗 Datasets [`map`](https://huggingface.co/docs/datasets/process#map) لتطبيق دالة معالجة مسبقة على مجموعة البيانات بأكملها:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


>>> def tokenize_function(examples):
...     return tokenizer(examples["text"], padding="max_length", truncation=True)
>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


>>> def tokenize_function(examples):
...     return tokenizer(examples["text"], padding="max_length", truncation=True)


>>> tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

إذا كنت ترغب، يمكنك إنشاء مجموعة فرعية أصغر من مجموعة البيانات الكاملة لضبطها لتقليل الوقت الذي تستغرقه:

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

<a id='trainer'></a>

## التدريب

في هذه المرحلة، يجب عليك اتباع القسم الذي يتوافق مع الإطار الذي تريد استخدامه. يمكنك استخدام الروابط
في شريط التنقل الأيمن للقفز إلى الإطار الذي تريده - وإذا كنت تريد إخفاء كل المحتوى لإطار معين،
فاستخدم الزر في الركن العلوي الأيمن من كتلة الإطار!

<frameworkcontent>
<pt>
<Youtube id="nvBXf7s7vTI"/>

## التدريب باستخدام PyTorch Trainer

تقدم مكتبة 🤗 Transformers فئة [`Trainer`] مُحسّنة لتدريب نماذج 🤗 Transformers، مما يسهل بدء التدريب دون الحاجة إلى كتابة حلقة التدريب الخاصة بك يدويًا. تدعم واجهة برمجة تطبيقات [`Trainer`] مجموعة واسعة من خيارات التدريب والميزات مثل التسجيل، وتراكم التدرجات، والدقة المختلطة.

ابدأ بتحميل نموذجك وتحديد عدد التصنيفات المتوقعة. من بطاقة مجموعة بيانات Yelp Review [dataset card](https://huggingface.co/datasets/yelp_review_full#data-fields)، تعرف أنه يوجد خمسة تصنيفات:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
```

<Tip>

سترى تحذيرًا بشأن بعض أوزان النموذج المُدرب مسبقًا التي لم يتم استخدامها وبعض الأوزان التي تم تهيئتها بشكل عشوائي. لا تقلق، هذا أمر طبيعي تمامًا! يتم التخلص من رأس النموذج المُدرب مسبقًا لشبكة BERT، ويتم استبداله برأس تصنيف مُهيأ بشكل عشوائي. سوف تقوم بضبط رأس النموذج الجديد هذا على مهمة تصنيف التسلسل، ونقل معرفة النموذج المُدرب مسبقًا إليه.

</Tip>

### فرط المعالم

بعد ذلك، قم بإنشاء فئة [`TrainingArguments`] والتي تحتوي على جميع فرط المعالم التي يمكنك ضبطها بالإضافة إلى أعلام لتنشيط خيارات التدريب المختلفة. بالنسبة لهذا البرنامج التعليمي، يمكنك البدء بفرط المعالم الافتراضية [hyperparameters](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)، ولكن لا تتردد في تجربتها للعثور على الإعدادات المثلى.

حدد مكان حفظ نقاط الفحص من تدريبك:

```py
>>> from transformers import TrainingArguments

>>> training_args = TrainingArguments(output_dir="test_trainer")
```

### التقييم

لا يقوم [`Trainer`] تلقائيًا بتقييم أداء النموذج أثناء التدريب. ستحتاج إلى تمرير دالة إلى [`Trainer`] لحساب الإبلاغ عن المقاييس. توفر مكتبة [🤗 Evaluate](https://huggingface.co/docs/evaluate/index) دالة [`accuracy`](https://huggingface.co/spaces/evaluate-metric/accuracy) بسيطة يمكنك تحميلها باستخدام الدالة [`evaluate.load`] (راجع هذا [الدليل السريع](https://huggingface.co/docs/evaluate/a_quick_tour) لمزيد من المعلومات):

```py
>>> import numpy as np
>>> import evaluate

>>> metric = evaluate.load("accuracy")
```

قم بالاتصال بـ [`~evaluate.compute`] على `metric` لحساب دقة تنبؤاتك. قبل تمرير تنبؤاتك إلى `compute`، تحتاج إلى تحويل logits إلى تنبؤات (تذكر أن جميع نماذج 🤗 Transformers تعيد logits):

```py
>>> def compute_metrics(eval_pred):
...     logits، labels = eval_pred
...     التوقعات = np.argmax (logits، المحور = -1)
...     return metric.compute (التنبؤات = التوقعات، المراجع = العلامات)
```

إذا كنت ترغب في مراقبة مقاييس التقييم الخاصة بك أثناء الضبط الدقيق، فحدد معلمة `eval_strategy` في حجج التدريب الخاصة بك للإبلاغ عن المقياس في نهاية كل حقبة:

```py
>>> from transformers import TrainingArguments, Trainer

>>> training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
```

### المدرب

قم بإنشاء كائن [`Trainer`] باستخدام نموذجك، وحجج التدريب، ومجموعات البيانات التدريبية والاختبارية، ودالة التقييم:

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

ثم قم بضبط نموذجك عن طريق استدعاء [`~transformers.Trainer.train`]:

```py
>>> trainer.train()
```
</pt>
<tf>
<a id='keras'></a>

<Youtube id="rnTGBy2ax1c"/>

## تدريب نموذج TensorFlow باستخدام Keras

يمكنك أيضًا تدريب نماذج 🤗 Transformers في TensorFlow باستخدام واجهة برمجة تطبيقات Keras!

### تحميل البيانات لـ Keras

عندما تريد تدريب نموذج 🤗 Transformers باستخدام واجهة برمجة تطبيقات Keras، فأنت بحاجة إلى تحويل مجموعة البيانات الخاصة بك إلى تنسيق يفهمه
Keras. إذا كانت مجموعة البيانات الخاصة بك صغيرة، فيمكنك ببساطة تحويلها بالكامل إلى مصفوفات NumPy وإرسالها إلى Keras.
دعونا نجرب ذلك أولاً قبل أن نقوم بأي شيء أكثر تعقيدًا.

أولاً، قم بتحميل مجموعة بيانات. سنستخدم مجموعة بيانات CoLA من معيار [GLUE benchmark](https://huggingface.co/datasets/glue)،
نظرًا لأنه مهمة تصنيف نص ثنائي بسيطة، وسنأخذ فقط قسم التدريب الآن.

```py
from datasets import load_dataset

dataset = load_dataset("glue"، "cola")
dataset = dataset ["train"] # خذ فقط قسم التدريب الآن
```

بعد ذلك، قم بتحميل أداة تعلم علامات الترقيم وقم بترميز البيانات كمصفوفات NumPy. لاحظ أن التصنيفات هي بالفعل قائمة من 0 و 1،
لذا يمكننا ببساطة تحويل ذلك مباشرة إلى مصفوفة NumPy بدون ترميز!

```py
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
tokenized_data = tokenizer(dataset["sentence"], return_tensors="np", padding=True)
# Tokenizer returns a BatchEncoding, but we convert that to a dict for Keras
tokenized_data = dict(tokenized_data)

labels = np.array(dataset["label"])  # Label is already an array of 0 and 1
```

أخيرًا، قم بتحميل وتجميع وتناسب النموذج. لاحظ أن نماذج Transformers تحتوي جميعها على دالة خسارة ذات صلة بالمهمة بشكل افتراضي، لذا فأنت لست بحاجة إلى تحديد واحدة ما لم ترغب في ذلك:

```py
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam

# تحميل وتجميع النموذج الخاص بنا
model = TFAutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased")
# معدلات التعلم المنخفضة أفضل غالبًا لضبط النماذج الدقيقة
model.compile(optimizer=Adam(3e-5)) # لا توجد دالة خسارة!

model.fit(tokenized_data, labels)
```

<Tip>

أنت لست مضطرًا لتمرير دالة خسارة إلى نماذجك عند تجميعها! تختار نماذج Hugging Face تلقائيًا
دالة خسارة مناسبة لمهمتها وهندسة نموذجها إذا تُركت هذه الحجة فارغة. يمكنك دائمًا
تجاوز ذلك عن طريق تحديد دالة خسارة بنفسك إذا كنت تريد ذلك!

</Tip>

يعمل هذا النهج بشكل رائع لمجموعات البيانات الصغيرة، ولكن بالنسبة لمجموعات البيانات الأكبر، فقد تجد أنه يصبح مشكلة. لماذا؟
لأن المصفوفة المرمزة والتصنيفات يجب أن يتم تحميلها بالكامل في الذاكرة، ولأن NumPy لا يتعامل مع
المصفوفات "المسننة"، لذا يجب أن يتم ترميز كل عينة بالطول الأقصى للعينة في مجموعة البيانات بأكملها. سيؤدي ذلك إلى زيادة حجم المصفوفة لديك، وستبطئ جميع رموز الترقيم من عملية التدريب أيضًا!

### تحميل البيانات كـ tf.data.Dataset

إذا كنت تريد تجنب إبطاء التدريب، فيمكنك تحميل بياناتك كـ `tf.data.Dataset` بدلاً من ذلك. على الرغم من أنه يمكنك كتابة خط أنابيب `tf.data` الخاص بك إذا كنت تريد، إلا أن لدينا طريقتين مختصرتين للقيام بذلك:
- [`~TFPreTrainedModel.prepare_tf_dataset`]: هذه هي الطريقة التي نوصي بها في معظم الحالات. نظرًا لأنه طريقة
على نموذجك، فيمكنه فحص النموذج لتحديد الأعمدة القابلة للاستخدام كمدخلات للنموذج تلقائيًا،
واستبعاد الأعمدة الأخرى لإنشاء مجموعة بيانات أبسط وأكثر كفاءة.
- [`~datasets.Dataset.to_tf_dataset`]: هذه الطريقة أكثر أساسية، وهي مفيدة عندما تريد التحكم بدقة في كيفية
إنشاء مجموعة البيانات الخاصة بك، عن طريق تحديد أعمدة `columns` و `label_cols` المحددة التي سيتم تضمينها.

قبل أن تتمكن من استخدام [`~TFPreTrainedModel.prepare_tf_dataset`]، ستحتاج إلى إضافة نواتج أداة تعلم علامات الترقيم إلى مجموعة البيانات الخاصة بك كأعمدة، كما هو موضح في
عينة التعليمات البرمجية التالية:

```py
def tokenize_dataset (data):
# ستتم إضافة مفاتيح القاموس الذي تمت إعادته كأعمدة إلى مجموعة البيانات
return tokenizer(data["text"])


dataset = dataset.map(tokenize_dataset)
```

تذكر أن مجموعات بيانات Hugging Face يتم تخزينها على القرص بشكل افتراضي، لذا فلن يؤدي ذلك إلى تضخيم استخدام الذاكرة لديك! بمجرد إضافة الأعمدة، يمكنك بث الدفعات من مجموعة البيانات وإضافة الترميز إلى كل دفعة، مما يقلل بشكل كبير من عدد رموز الترقيم مقارنة بترميز مجموعة البيانات بأكملها.


```py
>>> tf_dataset = model.prepare_tf_dataset(dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer)
```

لاحظ أنه في عينة التعليمات البرمجية أعلاه، تحتاج إلى تمرير أداة تعلم علامات الترقيم إلى `prepare_tf_dataset` حتى تتمكن من ترميز الدفعات بشكل صحيح أثناء تحميلها.
إذا كانت جميع العينات في مجموعة البيانات الخاصة بك بنفس الطول ولم يكن الترميز ضروريًا، فيمكنك تخطي هذا الحجة.
إذا كنت بحاجة إلى القيام بشيء أكثر تعقيدًا من مجرد ترميز العينات (على سبيل المثال، إفساد الرموز للنمذجة اللغوية المُقنعة)،
فيمكنك استخدام حجة `collate_fn` بدلاً من ذلك لتمرير دالة يتم استدعاؤها لتحويل
قائمة العينات إلى دفعة وتطبيق أي معالجة مسبقة تريدها. راجع أمثلةنا [examples](https://github.com/huggingface/transformers/tree/main/examples) أو
[دفاتر الملاحظات](https://huggingface.co/docs/transformers/notebooks) لرؤية هذا النهج في العمل.

بمجرد إنشاء `tf.data.Dataset`، يمكنك تجميع النموذج وتناسبه كما هو الحال من قبل:

```py
model.compile(optimizer=Adam(3e-5))  # No loss argument!

model.fit(tf_dataset)
```

</tf>
</frameworkcontent>

<a id='pytorch_native'></a>
## تدريب في PyTorch الأصلي

<frameworkcontent>
<pt>
<Youtube id="Dh9CL8fyG80"/>

[`Trainer`] يهتم بحلقة التدريب ويسمح لك بضبط نموذج في سطر واحد من التعليمات البرمجية. بالنسبة للمستخدمين الذين يفضلون كتابة حلقة التدريب الخاصة بهم، يمكنك أيضًا ضبط نموذج 🤗 Transformers في PyTorch الأصلي.

في هذه المرحلة، قد تحتاج إلى إعادة تشغيل دفتر الملاحظات الخاص بك أو تنفيذ التعليمات البرمجية التالية لتحرير بعض الذاكرة:

```py
حذف النموذج
حذف المدرب
torch.cuda.empty_cache()
```

بعد ذلك، قم بمعالجة `tokenized_dataset` يدويًا لإعداده للتدريب.

1. إزالة عمود `text` لأن النموذج لا يقبل النص الخام كإدخال:

    ```py
    >>> tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    ```

2. إعادة تسمية عمود `label` إلى `labels` لأن النموذج يتوقع أن يكون الاسم `labels`:

    ```py
    >>> tokenized_datasets = tokenized_datasets.rename_column("label"، "labels")
    ```

3. قم بتعيين تنسيق مجموعة البيانات لإرجاع مؤشرات PyTorch بدلاً من القوائم:

    ```py
    >>> tokenized_datasets.set_format("torch")
    ```

بعد ذلك، قم بإنشاء مجموعة فرعية أصغر من مجموعة البيانات كما هو موضح سابقًا لتسريع الضبط الدقيق:

```py
>>> small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
>>> small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

### DataLoader

قم بإنشاء `DataLoader` لمجموعات بيانات التدريب والاختبار الخاصة بك حتى تتمكن من التكرار عبر دفعات البيانات:

```py
>>> from torch.utils.data import DataLoader

>>> train_dataloader = DataLoader(small_train_dataset، shuffle=True، batch_size=8)
>>> eval_dataloader = DataLoader(small_eval_dataset، batch_size=8)
```

قم بتحميل نموذجك مع عدد التصنيفات المتوقعة:

```py
>>> from transformers import AutoModelForSequenceClassification

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased"، num_labels=5)
```

### المحسن ومخطط معدل التعلم

قم بإنشاء محسن ومخطط معدل تعلم لضبط النموذج الدقيق. دعنا نستخدم [`AdamW`](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) المحسن من PyTorch:

```py
>>> from torch.optim import AdamW

>>> optimizer = AdamW(model.parameters()، lr=5e-5)
```

قم بإنشاء مخطط معدل التعلم الافتراضي من [`Trainer`]:

```py
>>> from transformers import get_scheduler

>>> num_epochs = 3
>>> num_training_steps = num_epochs * len(train_dataloader)
>>> lr_scheduler = get_scheduler(
...     name="linear"، optimizer=optimizer، num_warmup_steps=0، num_training_steps=num_training_steps
... )
```

أخيرًا، حدد `device` لاستخدام وحدة معالجة الرسومات (GPU) إذا كان لديك حق الوصول إليها. وإلا، فقد يستغرق التدريب على وحدة المعالجة المركزية (CPU) عدة ساعات بدلاً من دقائق قليلة.

```py
>>> import torch

>>> device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
>>> model.to(device)
```

<Tip>

احصل على وصول مجاني إلى وحدة معالجة رسومات سحابية إذا لم يكن لديك واحدة مع دفتر ملاحظات مستضاف مثل [Colaboratory](https://colab.research.google.com/) أو [SageMaker StudioLab](https://studiolab.sagemaker.aws/).

</Tip>

رائع، الآن أنت مستعد للتدريب! 🥳 

### حلقة التدريب

لمراقبة تقدم التدريب الخاص بك، استخدم مكتبة [tqdm](https://tqdm.github.io/) لإضافة شريط تقدم فوق عدد خطوات التدريب:

```py
>>> from tqdm.auto import tqdm

>>> progress_bar = tqdm(range(num_training_steps))

>>> model.train()
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         batch = {k: v.to(device) for k، v in batch.items()}
...         outputs = model(**batch)
...         loss = outputs.loss
...         loss.backward()

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

### تقييم

تمامًا كما أضفت وظيفة تقييم إلى [`Trainer`]]، تحتاج إلى القيام بنفس الشيء عندما تكتب حلقة التدريب الخاصة بك. ولكن بدلاً من حساب الإبلاغ عن المقياس في نهاية كل حقبة، هذه المرة ستقوم بتجميع جميع الدفعات باستخدام [`~evaluate.add_batch`] وحساب المقياس في النهاية.

```py
>>> import evaluate

>>> metric = evaluate.load("accuracy")
>>> model.eval()
>>> for batch in eval_dataloader:
...     batch = {k: v.to(device) for k، v in batch.items()}
...     with torch.no_grad():
...         outputs = model(**batch)

...     logits = outputs.logits
...     predictions = torch.argmax(logits، dim=-1)
...     metric.add_batch(predictions=predictions، references=batch["labels"])

>>> metric.compute()
```
</pt>
</frameworkcontent>

<a id='additional-resources'></a>

## موارد إضافية

لمزيد من الأمثلة على الضبط الدقيق، راجع:

- [🤗 أمثلة المحولات](https://github.com/huggingface/transformers/tree/main/examples) تتضمن
  النصوص البرمجية لتدريب مهام NLP الشائعة في PyTorch وTensorFlow.

- [🤗 دفاتر ملاحظات المحولات](notebooks) يحتوي على دفاتر ملاحظات مختلفة حول كيفية ضبط نموذج لمهمة محددة في PyTorch وTensorFlow.