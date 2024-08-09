# TAPEX

<Tip warning={true}>

تم وضع هذا النموذج في وضع الصيانة فقط، ولا نقبل أي طلبات سحب جديدة لتغيير شفرته.

إذا واجهتك أي مشكلات أثناء تشغيل هذا النموذج، يرجى إعادة تثبيت الإصدار الأخير الذي يدعم هذا النموذج: v4.30.0.

يمكنك القيام بذلك عن طريق تشغيل الأمر التالي: `pip install -U transformers==4.30.0`.

</Tip>

## نظرة عامة

اقترح نموذج TAPEX في [TAPEX: Table Pre-training via Learning a Neural SQL Executor](https://arxiv.org/abs/2107.07653) بواسطة Qian Liu، Bei Chen، Jiaqi Guo، Morteza Ziyadi، Zeqi Lin، Weizhu Chen، Jian-Guang Lou. يقوم TAPEX بتدريب نموذج BART مسبقًا لحل استعلامات SQL الاصطناعية، وبعد ذلك يمكن ضبطه بدقة للإجابة على أسئلة اللغة الطبيعية المتعلقة ببيانات الجدول، بالإضافة إلى التحقق من حقائق الجدول.

تم ضبط نموذج TAPEX بدقة على العديد من مجموعات البيانات:

- [SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253) (الإجابة على الأسئلة المتسلسلة بواسطة Microsoft)
- [WTQ](https://github.com/ppasupat/WikiTableQuestions) (أسئلة جداول ويكي من جامعة ستانفورد)
- [WikiSQL](https://github.com/salesforce/WikiSQL) (بواسطة Salesforce)
- [TabFact](https://tabfact.github.io/) (بواسطة USCB NLP Lab).

ملخص الورقة هو كما يلي:

> حقق التقدم الأخير في التدريب المسبق لنموذج اللغة نجاحًا كبيرًا من خلال الاستفادة من البيانات النصية غير المنظمة على نطاق واسع. ومع ذلك، لا يزال من الصعب تطبيق التدريب المسبق على بيانات الجدول المنظمة بسبب عدم وجود بيانات جدول كبيرة الحجم وعالية الجودة. في هذه الورقة، نقترح TAPEX لإظهار أن التدريب المسبق للجدول يمكن تحقيقه من خلال تعلم منفذ SQL العصبي عبر مجموعة بيانات اصطناعية، يتم الحصول عليها عن طريق توليف استعلامات SQL القابلة للتنفيذ وإخراج التنفيذ الخاص بها تلقائيًا. يعالج TAPEX تحدي ندرة البيانات من خلال توجيه نموذج اللغة لمحاكاة منفذ SQL على مجموعة البيانات الاصطناعية المتنوعة والكبيرة الحجم وعالية الجودة. نقوم بتقييم TAPEX على أربع مجموعات بيانات مرجعية. وتظهر النتائج التجريبية أن TAPEX يتفوق على أساليب التدريب المسبق للجدول السابقة بهامش كبير ويحقق نتائج جديدة رائدة في المجال في جميع مجموعات البيانات. ويشمل ذلك تحسينات على دقة التسمية الموجهة بالإشراف الضعيف في WikiSQL لتصل إلى 89.5% (+2.3%)، ودقة التسمية في WikiTableQuestions لتصل إلى 57.5% (+4.8%)، ودقة التسمية في SQA لتصل إلى 74.5% (+3.5%)، ودقة TabFact لتصل إلى 84.2% (+3.2%). وعلى حد علمنا، هذه هي أول ورقة بحثية تستغل التدريب المسبق للجدول عبر البرامج القابلة للتنفيذ الاصطناعية وتحقق نتائج جديدة رائدة في المجال في مختلف المهام اللاحقة.

## نصائح الاستخدام

- TAPEX هو نموذج توليدي (seq2seq). يمكنك توصيل أوزان TAPEX مباشرة في نموذج BART.
- يحتوي TAPEX على نقاط تفتيش على المحاور التي يتم تدريبها مسبقًا فقط، أو ضبطها بدقة على WTQ وSQA وWikiSQL وTabFact.
- يتم تقديم الجمل + الجداول إلى النموذج على الشكل التالي: `sentence + " " + linearized table`. تنسيق الجدول الخطي هو كما يلي:
`col: col1 | col2 | col 3 row 1 : val1 | val2 | val3 row 2 : ...`.
- لدى TAPEX برنامج تشغيل خاص به، والذي يسمح بإعداد جميع البيانات للنموذج بسهولة. يمكنك تمرير أطر بيانات Pandas والسلاسل النصية إلى برنامج التشغيل، وسيقوم تلقائيًا بإنشاء `input_ids` و`attention_mask` (كما هو موضح في أمثلة الاستخدام أدناه).

### الاستخدام: الاستدلال

فيما يلي، نوضح كيفية استخدام TAPEX للإجابة على أسئلة الجدول. كما يمكنك أن ترى، يمكنك توصيل أوزان TAPEX مباشرة في نموذج BART.

نستخدم واجهة برمجة التطبيقات التلقائية [auto API]، والتي ستقوم تلقائيًا بإنشاء برنامج تشفير ([`TapexTokenizer`]) ونموذج ([`BartForConditionalGeneration`]) المناسبين لنا، بناءً على ملف تكوين نقطة التفتيش على المحور.

```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> import pandas as pd

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/tapex-large-finetuned-wtq")

>>> # prepare table + question
>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> table = pd.DataFrame.from_dict(data)
>>> question = "how many movies does Leonardo Di Caprio have?"

>>> encoding = tokenizer(table, question, return_tensors="pt")

>>> # let the model generate an answer autoregressively
>>> outputs = model.generate(**encoding)

>>> # decode back to text
>>> predicted_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
>>> print(predicted_answer)
53
```

لاحظ أن [`TapexTokenizer`] يدعم أيضًا الاستدلال الدفعي. وبالتالي، يمكنك توفير دفعة من الجداول/الأسئلة المختلفة، أو دفعة من جدول واحد
وأسئلة متعددة، أو دفعة من استعلام واحد وجداول متعددة. دعونا نوضح هذا:

```python
>>> # prepare table + question
>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> table = pd.DataFrame.from_dict(data)
>>> questions = [
...     "how many movies does Leonardo Di Caprio have?",
...     "which actor has 69 movies?",
...     "what's the first name of the actor who has 87 movies?",
... ]
>>> encoding = tokenizer(table, questions, padding=True, return_tensors="pt")

>>> # let the model generate an answer autoregressively
>>> outputs = model.generate(**encoding)

>>> # decode back to text
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
[' 53', ' george clooney', ' brad pitt']
```

في حالة الرغبة في التحقق من الجدول (أي مهمة تحديد ما إذا كانت الجملة مدعومة أو مفندة من خلال محتويات جدول)، يمكنك إنشاء مثيل لنموذج [`BartForSequenceClassification`]. يحتوي TAPEX على نقاط تفتيش على المحور تم ضبطها بدقة على TabFact، وهي معيار مرجعي مهم للتحقق من حقائق الجدول (يحقق دقة بنسبة 84%). يوضح مثال التعليمات البرمجية أدناه مرة أخرى الاستفادة من واجهة برمجة التطبيقات التلقائية [auto API].

```python
>>> from transformers import AutoTokenizer, AutoModelForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large-finetuned-tabfact")
>>> model = AutoModelForSequenceClassification.from_pretrained("microsoft/tapex-large-finetuned-tabfact")

>>> # prepare table + sentence
>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> table = pd.DataFrame.from_dict(data)
>>> sentence = "George Clooney has 30 movies"

>>> encoding = tokenizer(table, sentence, return_tensors="pt")

>>> # forward pass
>>> outputs = model(**encoding)

>>> # print prediction
>>> predicted_class_idx = outputs.logits[0].argmax(dim=0).item()
>>> print(model.config.id2label[predicted_class_idx])
Refused
```

<Tip>

تتشابه بنية TAPEX مع بنية BART، باستثناء عملية البرمجة. راجع [توثيق BART](bart) للحصول على معلومات حول
فئات التكوين ومعلماتها. يتم توثيق برنامج تشفير TAPEX المحدد أدناه.

</Tip>

## TapexTokenizer

[[autodoc]] TapexTokenizer
- __call__
- save_vocabulary