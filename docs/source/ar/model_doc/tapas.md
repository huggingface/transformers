# TAPAS

## نظرة عامة

اقترح نموذج TAPAS في ورقة "TAPAS: Weakly Supervised Table Parsing via Pre-training" بواسطة جوناثان هيرزيغ وآخرون. وهو نموذج يعتمد على BERT مصمم خصيصًا (ومدرب مسبقًا) للإجابة على الأسئلة حول البيانات الجدولية. مقارنة بـ BERT، يستخدم TAPAS embeddings الموضعية النسبية وله 7 أنواع من الرموز التي تشفر البنية الجدولية. تم التدريب المسبق لـ TAPAS على هدف نمذجة اللغة المقنعة (MLM) على مجموعة بيانات كبيرة تضم ملايين الجداول من ويكيبيديا الإنجليزية والنصوص المقابلة.

بالنسبة للإجابة على الأسئلة، يحتوي TAPAS على رأسين في الأعلى: رأس اختيار الخلية ورأس التجميع، لأداء التجميعات (مثل العد أو الجمع) بين الخلايا المحددة. تم ضبط TAPAS بشكل دقيق على عدة مجموعات بيانات:

- SQA (الإجابة على الأسئلة المتسلسلة من مايكروسوفت)
- WTQ (أسئلة الجدول على ويكي من جامعة ستانفورد)
- WikiSQL (من Salesforce)

يحقق TAPAS حالة فنية على كل من SQA و WTQ، بينما يحقق أداءًا مماثلاً لحالة SOTA على WikiSQL، مع بنية أبسط بكثير.

ملخص الورقة هو كما يلي:

*يُنظر عادةً إلى الإجابة على الأسئلة الطبيعية على الجداول على أنها مهمة تحليل دلالي. للتخفيف من تكلفة جمع الأشكال المنطقية الكاملة، يركز أحد الأساليب الشائعة على الإشراف الضعيف المكون من الدلالات بدلاً من الأشكال المنطقية. ومع ذلك، فإن تدريب برامج التحليل الدلالي من الإشراف الضعيف يطرح صعوبات، بالإضافة إلى ذلك، يتم استخدام الأشكال المنطقية المولدة كخطوة وسيطة فقط قبل استرداد الدلالة. في هذه الورقة، نقدم TAPAS، وهو نهج للإجابة على الأسئلة على الجداول دون توليد أشكال منطقية. يتم تدريب TAPAS من الإشراف الضعيف، ويتوقع الدلالة عن طريق تحديد خلايا الجدول وتطبيق مشغل تجميع مطابق بشكل اختياري على هذا التحديد. يوسع TAPAS بنية BERT لتشفير الجداول كإدخال، ويتم تهيئته من التدريب المشترك الفعال لشرائح النص والجداول المستخرجة من ويكيبيديا، ويتم تدريبه من النهاية إلى النهاية. نجري تجارب على ثلاث مجموعات بيانات مختلفة لتحليل الدلالات، ونجد أن TAPAS يتفوق على نماذج التحليل الدلالي أو ينافسها من خلال تحسين دقة SOTA على SQA من 55.1 إلى 67.2 والأداء على قدم المساواة مع SOTA على WIKISQL وWIKITQ، ولكن مع بنية نموذج أبسط. بالإضافة إلى ذلك، نجد أن التعلم التحويلي، وهو أمر بسيط في إعدادنا، من WIKISQL إلى WIKITQ، ينتج عنه دقة تبلغ 48.7، أي أعلى بـ 4.2 نقطة من SOTA.*

بالإضافة إلى ذلك، قام مؤلفو الورقة بتدريب TAPAS بشكل مسبق للتعرف على استنتاج الجدول، من خلال إنشاء مجموعة بيانات متوازنة من ملايين الأمثلة التدريبية التي تم إنشاؤها تلقائيًا والتي يتم تعلمها في خطوة وسيطة قبل الضبط الدقيق. يطلق مؤلفو TAPAS على هذا التدريب المسبق الإضافي اسم "التدريب المسبق الوسيط" (حيث يتم تدريب TAPAS مسبقًا أولاً على MLM، ثم على مجموعة بيانات أخرى). ووجدوا أن التدريب الوسيط يحسن الأداء بشكل أكبر على SQA، محققًا حالة فنية جديدة، بالإضافة إلى حالة فنية على TabFact، وهي مجموعة بيانات واسعة النطاق تحتوي على 16 ألف جدول من ويكيبيديا لاستنتاج الجدول (مهمة تصنيف ثنائي). لمزيد من التفاصيل، راجع ورقة المتابعة الخاصة بهم: "Understanding tables with intermediate pre-training".

![TAPAS architecture](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tapas_architecture.png)

تمت المساهمة بهذا النموذج من قبل [nielsr]. تمت المساهمة في إصدار TensorFlow من هذا النموذج بواسطة [kamalkraj]. يمكن العثور على الكود الأصلي [هنا](https://github.com/google-research/tapas).

## نصائح الاستخدام

- يستخدم TAPAS embeddings الموضعية النسبية بشكل افتراضي (إعادة تعيين embeddings الموضعية لكل خلية من الجدول). لاحظ أن هذا شيء تمت إضافته بعد نشر ورقة TAPAS الأصلية. وفقًا للمؤلفين، عادة ما يؤدي ذلك إلى أداء أفضل قليلاً، ويتيح لك تشفير تسلسلات أطول دون نفاد embeddings. ينعكس هذا في `reset_position_index_per_cell` معلمة [`TapasConfig`]، والتي يتم تعيينها إلى `True` بشكل افتراضي. تستخدم الإصدارات الافتراضية من النماذج المتوفرة على [hub] جميع embeddings الموضعية النسبية. لا يزال بإمكانك استخدام تلك التي تحتوي على embeddings الموضعية المطلقة عن طريق تمرير حجة إضافية `revision="no_reset"` عند استدعاء طريقة `from_pretrained()`. لاحظ أنه يُنصح عادةً بتبطين الإدخالات على اليمين بدلاً من اليسار.

- يعتمد TAPAS على BERT، لذا فإن "TAPAS-base"، على سبيل المثال، يقابل بنية "BERT-base". بالطبع، سيؤدي "TAPAS-large" إلى أفضل أداء (النتائج المبلغ عنها في الورقة هي من "TAPAS-large"). يتم عرض نتائج النماذج ذات الأحجام المختلفة على [مستودع GitHub الأصلي](https://github.com/google-research/tapas).

- يحتوي TAPAS على نقاط تفتيش تمت معايرتها بدقة على SQA، وهي قادرة على الإجابة على الأسئلة المتعلقة بالجدول في إعداد المحادثة. وهذا يعني أنه يمكنك طرح أسئلة متابعة مثل "ما هو عمره؟" المتعلقة بالسؤال السابق. لاحظ أن عملية التمرير الأمامي لـ TAPAS تختلف قليلاً في حالة الإعداد المحادثي: في هذه الحالة، يجب إدخال كل زوج من الجدول والأسئلة واحدًا تلو الآخر إلى النموذج، بحيث يمكن الكتابة فوق معرّفات نوع الرمز `prev_labels` بواسطة `labels` المتوقعة من النموذج للسؤال السابق. راجع قسم "الاستخدام" لمزيد من المعلومات.

- يشبه TAPAS نموذج BERT وبالتالي يعتمد على هدف نمذجة اللغة المقيدة (MLM). لذلك، فهو فعال في التنبؤ بالرموز المقنعة وفي NLU بشكل عام، ولكنه غير مثالي لتوليد النصوص. النماذج المدربة بهدف نمذجة اللغة السببية (CLM) أفضل في هذا الصدد. لاحظ أنه يمكن استخدام TAPAS كترميز في إطار EncoderDecoderModel، لدمجه مع فك تشفير نصي ذاتي الارتباط مثل GPT-2.
## الاستخدام: الضبط الدقيق

هنا نشرح كيف يمكنك ضبط نموذج [TapasForQuestionAnswering] بدقة على مجموعة البيانات الخاصة بك.

**الخطوة 1: اختر واحدة من 3 طرق يمكنك من خلالها استخدام TAPAS - أو إجراء تجربة**

بشكل أساسي، هناك 3 طرق مختلفة يمكن من خلالها ضبط نموذج [TapasForQuestionAnswering] بدقة، والتي تتوافق مع مجموعات البيانات المختلفة التي تم ضبط نموذج Tapas عليها:

1. SQA: إذا كنت مهتمًا بطرح أسئلة متابعة تتعلق بجدول ما، في إعداد محادثة. على سبيل المثال، إذا كان سؤالك الأول هو "ما اسم الممثل الأول؟" يمكنك بعد ذلك طرح سؤال متابعة مثل "ما عمره؟". هنا، لا تتضمن الأسئلة أي تجميع (جميع الأسئلة هي أسئلة اختيار الخلية).

2. WTQ: إذا لم تكن مهتمًا بطرح أسئلة في إعداد محادثة، ولكن بدلاً من ذلك تريد فقط طرح أسئلة تتعلق بجدول قد يتضمن التجميع، مثل حساب عدد الصفوف أو جمع قيم الخلايا أو حساب متوسطها. يمكنك بعد ذلك، على سبيل المثال، أن تسأل "ما هو العدد الإجمالي للأهداف التي سجلها كريستيانو رونالدو في مسيرته؟". تُعرف هذه الحالة أيضًا باسم **الإشراف الضعيف**، حيث يجب على النموذج نفسه أن يتعلم عامل التجميع المناسب (SUM/COUNT/AVERAGE/NONE) بناءً على إجابة السؤال فقط كإشراف.

3. WikiSQL-supervised: تستند مجموعة البيانات هذه إلى WikiSQL مع إعطاء النموذج عامل التجميع الصحيح أثناء التدريب. يُطلق على هذا أيضًا اسم **الإشراف القوي**. هنا، يكون تعلم عامل التجميع المناسب أسهل بكثير.

ملخص:

| **المهمة**                            | **مجموعة البيانات المثال** | **الوصف**                                                                                         |
|-------------------------------------|---------------------|---------------------------------------------------------------------------------------------------------|
| محادثة                      | SQA                 | محادثي، أسئلة اختيار الخلية فقط                                                           |
| الإشراف الضعيف للتجميع    | WTQ                 | قد تتضمن الأسئلة التجميع، ويجب على النموذج تعلم هذا بناءً على الإجابة فقط كإشراف |
| الإشراف القوي للتجميع  | WikiSQL-supervised  | قد تتضمن الأسئلة التجميع، ويجب على النموذج تعلم هذا بناءً على عامل التجميع الذهبي         |
<frameworkcontent>
<pt>
يمكن تهيئة نموذج باستخدام قاعدة مدربة مسبقًا ورؤوس تصنيف مهيأة بشكل عشوائي من المركز كما هو موضح أدناه.

```py
>>> from transformers import TapasConfig, TapasForQuestionAnswering

>>> # على سبيل المثال، نموذج الحجم الأساسي مع تهيئة SQA الافتراضية
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base")

>>> # أو، نموذج الحجم الأساسي مع تهيئة WTQ
>>> config = TapasConfig.from_pretrained("google/tapas-base-finetuned-wtq")
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)

>>> # أو، نموذج الحجم الأساسي مع تهيئة WikiSQL
>>> config = TapasConfig("google-base-finetuned-wikisql-supervised")
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
```

بالطبع، لا يتعين عليك اتباع إحدى هذه الطرق الثلاث التي تم بها ضبط نموذج TAPAS بدقة. يمكنك أيضًا إجراء تجارب عن طريق تحديد أي معلمات تريد عند تهيئة [`TapasConfig`]، ثم إنشاء [`TapasForQuestionAnswering`] بناءً على تلك التهيئة. على سبيل المثال، إذا كانت لديك مجموعة بيانات تحتوي على أسئلة محادثية وأسئلة قد تتضمن التجميع، فيمكنك القيام بذلك بهذه الطريقة. إليك مثال:

```py
>>> from transformers import TapasConfig, TapasForQuestionAnswering

>>> # يمكنك تهيئة رؤوس التصنيف بأي طريقة تريدها (راجع وثائق TapasConfig)
>>> config = TapasConfig(num_aggregation_labels=3, average_logits_per_cell=True)
>>> # تهيئة نموذج الحجم الأساسي المُدرب مسبقًا برؤوس التصنيف المخصصة لدينا
>>> model = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
```
</pt>
<tf>
يمكن تهيئة نموذج باستخدام قاعدة مدربة مسبقًا ورؤوس تصنيف مهيأة بشكل عشوائي من المركز كما هو موضح أدناه. تأكد من تثبيت اعتماد [tensorflow_probability](https://github.com/tensorflow/probability):

```py
>>> from transformers import TapasConfig, TFTapasForQuestionAnswering

>>> # على سبيل المثال، نموذج الحجم الأساسي مع تهيئة SQA الافتراضية
>>> model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base")

>>> # أو، نموذج الحجم الأساسي مع تهيئة WTQ
>>> config = TapasConfig.from_pretrained("google/tapas-base-finetuned-wtq")
>>> model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)

>>> # أو، نموذج الحجم الأساسي مع تهيئة WikiSQL
>>> config = TapasConfig("google-base-finetuned-wikisql-supervised")
>>> model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
```

بالطبع، لا يتعين عليك اتباع إحدى هذه الطرق الثلاث التي تم بها ضبط نموذج TAPAS بدقة. يمكنك أيضًا إجراء تجارب عن طريق تحديد أي معلمات تريد عند تهيئة [`TapasConfig`]، ثم إنشاء [`TFTapasForQuestionAnswering`] بناءً على تلك التهيئة. على سبيل المثال، إذا كانت لديك مجموعة بيانات تحتوي على أسئلة محادثية وأسئلة قد تتضمن التجميع، فيمكنك القيام بذلك بهذه الطريقة. إليك مثال:

```py
>>> from transformers import TapasConfig, TFTapasForQuestionAnswering

>>> # يمكنك تهيئة رؤوس التصنيف بأي طريقة تريدها (راجع وثائق TapasConfig)
>>> config = TapasConfig(num_aggregation_labels=3, average_logits_per_cell=True)
>>> # تهيئة نموذج الحجم الأساسي المُدرب مسبقًا برؤوس التصنيف المخصصة لدينا
>>> model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)
```
</tf>
</frameworkcontent>

ما يمكنك القيام به أيضًا هو البدء من نقطة ضبط دقيقة جاهزة. تجدر الإشارة هنا إلى أن نقطة الضبط الدقيق الجاهزة على WTQ بها بعض المشكلات بسبب فقدان L2 الذي يعد هشًا إلى حد ما. راجع [هنا](https://github.com/google-research/tapas/issues/91#issuecomment-735719340) للحصول على مزيد من المعلومات.

لعرض قائمة بجميع نقاط التحقق من Tapas المُدربة مسبقًا والمضبوطة بدقة المتوفرة في مركز HuggingFace، راجع [هنا](https://huggingface.co/models?search=tapas).

**الخطوة 2: قم بإعداد بياناتك بتنسيق SQA**

ثانيًا، بغض النظر عما اخترته أعلاه، يجب عليك إعداد مجموعة البيانات الخاصة بك بتنسيق [SQA](https://www.microsoft.com/en-us/download/details.aspx?id=54253). هذا التنسيق عبارة عن ملف TSV/CSV مع الأعمدة التالية:

- `id`: اختياري، معرف زوج الجدول والسؤال، لأغراض مسك الدفاتر.
- `annotator`: اختياري، معرف الشخص الذي قام بتعليق زوج الجدول والسؤال، لأغراض مسك الدفاتر.
- `position`: رقم صحيح يشير إلى ما إذا كان السؤال هو الأول أو الثاني أو الثالث... المتعلق بالجدول. مطلوب فقط في حالة الإعداد المحادثي (SQA). لا تحتاج إلى هذا العمود في حالة اختيار WTQ/WikiSQL-supervised.
- `question`: سلسلة نصية
- `table_file`: سلسلة نصية، اسم ملف csv يحتوي على البيانات الجدولية
- `answer_coordinates`: قائمة من زوج واحد أو أكثر (كل زوج هو إحداثيات خلية، أي زوج صف وعمود يشكل جزءًا من الإجابة)
- `answer_text`: قائمة من سلسلة نصية واحدة أو أكثر (كل سلسلة هي قيمة خلية تشكل جزءًا من الإجابة)
- `aggregation_label`: فهرس عامل التجميع. مطلوب فقط في حالة الإشراف القوي للتجميع (حالة WikiSQL-supervised)
- `float_answer`: الإجابة الرقمية للسؤال، إذا كان هناك (np.nan إذا لم يكن هناك). مطلوب فقط في حالة الإشراف الضعيف للتجميع (مثل WTQ وWikiSQL)

يجب أن تكون الجداول نفسها موجودة في مجلد، مع وجود كل جدول في ملف csv منفصل. لاحظ أن مؤلفي خوارزمية TAPAS استخدموا نصوص تحويل مع بعض المنطق التلقائي لتحويل مجموعات البيانات الأخرى (WTQ، WikiSQL) إلى تنسيق SQA. يوضح المؤلف ذلك [هنا](https://github.com/google-research/tapas/issues/50#issuecomment-705465960). يمكن العثور على تحويل لنص البرنامج النصي هذا يعمل مع تنفيذ HuggingFace [هنا](https://github.com/NielsRogge/tapas_utils). من المثير للاهتمام أن نصوص التحويل هذه ليست مثالية (يتم ملء حقلي `answer_coordinates` و`float_answer` بناءً على `answer_text`)، مما يعني أن نتائج WTQ وWikiSQL يمكن أن تتحسن بالفعل.

**الخطوة 3: قم بتحويل بياناتك إلى تنسورات باستخدام TapasTokenizer**

<frameworkcontent>
<pt>
ثالثًا، نظرًا لأنك أعددت بياناتك بتنسيق TSV/CSV هذا (وملفات csv المقابلة التي تحتوي على البيانات الجدولية)، فيمكنك بعد ذلك استخدام [`TapasTokenizer`] لتحويل أزواج الجدول والسؤال إلى `input_ids` و`attention_mask` و`token_type_ids` وهكذا. مرة أخرى، بناءً على أي من الحالات الثلاث التي اخترتها أعلاه، يتطلب [`TapasForQuestionAnswering`] مدخلات مختلفة ليتم ضبطها بدقة:

| **المهمة**                           | **المدخلات المطلوبة**                                                                                                 |
|------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| محادثة                     | `input_ids`، `attention_mask`، `token_type_ids`، `labels`                                                           |
| الإشراف الضعيف للتجميع  | `input_ids`، `attention_mask`، `token_type_ids`، `labels`، `numeric_values`، `numeric_values_scale`، `float_answer` |
| الإشراف القوي للتجميع | `input ids`، `attention mask`، `token type ids`، `labels`، `aggregation_labels`                                     |

ينشئ [`TapasTokenizer`] القيم `labels` و`numeric_values` و`numeric_values_scale` بناءً على عمودي `answer_coordinates` و`answer_text` في ملف TSV. القيم `float_answer` و`aggregation_labels` موجودة بالفعل في ملف TSV من الخطوة 2. إليك مثال:

```py
>>> from transformers import TapasTokenizer
>>> import pandas as pd

>>> model_name = "google/tapas-base"
>>> tokenizer = TapasTokenizer.from_pretrained(model_name)

>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> queries = [
...     "What is the name of the first actor?",
...     "How many movies has George Clooney played in?",
...     "What is the total number of movies?",
... ]
>>> answer_coordinates = [[(0, 0)], [(2, 1)], [(0, 1), (1, 1), (2, 1)]]
>>> answer_text = [["Brad Pitt"], ["69"], ["209"]]
>>> table = pd.DataFrame.from_dict(data)
>>> inputs = tokenizer(
...     table=table,
...     queries=queries,
...     answer_coordinates=answer_coordinates,
...     answer_text=answer_text,
...     padding="max_length",
...     return_tensors="pt",
... )
>>> inputs
{'input_ids': tensor([[ ... ]]), 'attention_mask': tensor([[...]]), 'token_type_ids': tensor([[[...]]]),
'numeric_values': tensor([[ ... ]]), 'numeric_values_scale: tensor([[ ... ]]), labels: tensor([[ ... ]])}
```

لاحظ أن [`TapasTokenizer`] يتوقع أن تكون بيانات الجدول **نصًا فقط**. يمكنك استخدام `.astype(str)` على dataframe لتحويله إلى بيانات نصية فقط.

بالطبع، هذا يوضح فقط كيفية تشفير مثال تدريبي واحد. يُنصح بإنشاء برنامج تحميل بيانات لإجراء التكرار على الدفعات:

```py
>>> import torch
>>> import pandas as pd

>>> tsv_path = "your_path_to_the_tsv_file"
>>> table_csv_path = "your_path_to_a_directory_containing_all_csv_files"


>>> class TableDataset(torch.utils.data.Dataset):
...     def __init__(self, data, tokenizer):
...         self.data = data
...         self.tokenizer = tokenizer

...     def __getitem__(self, idx):
...         item = data.iloc[idx]
...         table = pd.read_csv(table_csv_path + item.table_file).astype(
...             str
...         )  # تأكد من جعل بيانات الجدول نصية فقط
...         encoding = self.tokenizer(
...             table=table,
...             queries=item.question,
...             answer_coordinates=item.answer_coordinates,
...             answer_text=item.answer_text,
...             truncation=True,
...             padding="max_length",
...             return_tensors="pt",
...         )
...         # احذف بُعد الدُفعة الذي يضيفه tokenizer بشكل افتراضي
...         encoding = {key: val.squeeze(0) for key, val in encoding.items()}
...         # أضف float_answer المطلوب أيضًا (حالة الإشراف الضعيف للتجميع)
...         encoding["float_answer"] = torch.tensor(item.float_answer)
...         return encoding

...     def __len__(self):
...         return len(self.data)


>>> data = pd.read_csv(tsv_path, sep="\t")
>>> train_dataset = TableDataset(data, tokenizer)
>>> train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
```
</pt>
<tf>
ثالثًا، نظرًا لأنك أعددت بياناتك بتنسيق TSV/CSV هذا (وملفات csv المقابلة التي تحتوي على البيانات الجدولية)، فيمكنك بعد ذلك استخدام [`TapasTokenizer`] لتحويل أزواج الجدول والسؤال إلى `input_ids` و`attention_mask` و`token_type_ids` وهكذا. مرة أخرى، بناءً على أي من الحالات الثلاث التي اخترتها أعلاه، يتطلب [`TFTapasForQuestionAnswering`] مدخلات مختلفة ليتم ضبطها بدقة:

| **المهمة**                           | **المدخلات المطلوبة**                                                                                                 |
|------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| محادثة                     | `input_ids`، `attention_mask`، `token_type_ids`، `labels`                                                           |
| الإشراف الضعيف للتجميع  | `input_ids`، `attention_mask`، `token_type_ids`، `labels`، `numeric_values`، `numeric_values_scale`، `float_answer` |
| الإشراف القوي للتجميع | `input ids`، `attention mask`، `token type ids`، `labels`، `aggregation_labels`                                     |

ينشئ [`TapasTokenizer`] القيم `labels` و`numeric_values` و`numeric_values_scale` بناءً على عمودي `answer_coordinates` و`answer_text` في ملف TSV. القيم `float_answer` و`aggregation_labels` موجودة بالفعل في ملف TSV من الخطوة
## Usage: inference

<frameworkcontent>

<pt>
في هذا القسم، نشرح كيفية استخدام [`TapasForQuestionAnswering`] أو [`TFTapasForQuestionAnswering`] للتنبؤ (أي إجراء تنبؤات على بيانات جديدة). للتنبؤ، يجب توفير `input_ids` و `attention_mask` و `token_type_ids` فقط (والتي يمكن الحصول عليها باستخدام [`TapasTokenizer`]) إلى النموذج للحصول على logits. بعد ذلك، يمكنك استخدام طريقة [`~models.tapas.tokenization_tapas.convert_logits_to_predictions`] لتحويل هذه القيم إلى إحداثيات تنبؤات ومؤشرات تجميع اختيارية.

ومع ذلك، لاحظ أن التنبؤ يختلف باختلاف ما إذا كان الإعداد محادثيًا أم لا. في الإعداد غير المحادثي، يمكن إجراء التنبؤ بالتوازي لجميع أزواج الجدول-السؤال في دفعة. إليك مثال على ذلك:

```py
>>> from transformers import TapasTokenizer, TapasForQuestionAnswering
>>> import pandas as pd

>>> model_name = "google/tapas-base-finetuned-wtq"
>>> model = TapasForQuestionAnswering.from_pretrained(model_name)
>>> tokenizer = TapasTokenizer.from_pretrained(model_name)

>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> queries = [
...     "What is the name of the first actor?",
...     "How many movies has George Clooney played in?",
...     "What is the total number of movies?",
... ]
>>> table = pd.DataFrame.from_dict(data)
>>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
>>> outputs = model(**inputs)
>>> predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
...     inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
... )

>>> # دعنا نطبع النتائج:
>>> id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
>>> aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

>>> answers = []
>>> for coordinates in predicted_answer_coordinates:
...     if len(coordinates) == 1:
...         # خلية واحدة فقط:
...         answers.append(table.iat[coordinates[0]])
...     else:
...         # خلايا متعددة
...         cell_values = []
...         for coordinate in coordinates:
...             cell_values.append(table.iat[coordinate])
...         answers.append(", ".join(cell_values))

>>> display(table)
>>> print("")
>>> for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
...     print(query)
...     if predicted_agg == "NONE":
...         print("Predicted answer: " + answer)
...     else:
...         print("Predicted answer: " + predicted_agg + " > " + answer)
What is the name of the first actor?
Predicted answer: Brad Pitt
How many movies has George Clooney played in?
Predicted answer: COUNT > 69
What is the total number of movies?
Predicted answer: SUM > 87, 53, 69
```
</pt>

<tf>
في هذا القسم، نشرح كيفية استخدام [`TFTapasForQuestionAnswering`] للتنبؤ (أي إجراء تنبؤات على بيانات جديدة). للتنبؤ، يجب توفير `input_ids` و `attention_mask` و `token_type_ids` فقط (والتي يمكن الحصول عليها باستخدام [`TapasTokenizer`]) إلى النموذج للحصول على logits. بعد ذلك، يمكنك استخدام طريقة [`~models.tapas.tokenization_tapas.convert_logits_to_predictions`] لتحويل هذه القيم إلى إحداثيات تنبؤات ومؤشرات تجميع اختيارية.

ومع ذلك، لاحظ أن التنبؤ يختلف باختلاف ما إذا كان الإعداد محادثيًا أم لا. في الإعداد غير المحادثي، يمكن إجراء التنبؤ بالتوازي لجميع أزواج الجدول-السؤال في دفعة. إليك مثال على ذلك:

```py
>>> from transformers import TapasTokenizer, TFTapasForQuestionAnswering
>>> import pandas as pd

>>> model_name = "google/tapas-base-finetuned-wtq"
>>> model = TFTapasForQuestionAnswering.from_pretrained(model_name)
>>> tokenizer = TapasTokenizer.from_pretrained(model_name)

>>> data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
>>> queries = [
...     "What is the name of the first actor?",
...     "How many movies has George Clooney played in?",
...     "What is the total number of movies?",
... ]
>>> table = pd.DataFrame.from_dict(data)
>>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="tf")
>>> outputs = model(**inputs)
>>> predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
...     inputs, outputs.logits, outputs.logits_aggregation
... )

>>> # دعنا نطبع النتائج:
>>> id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
>>> aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

>>> answers = []
>>> for coordinates in predicted_answer_coordinates:
...     if len(coordinates) == 1:
...         # خلية واحدة فقط:
...         answers.append(table.iat[coordinates[0]])
...     else:
...         # خلايا متعددة
...         cell_values = []
...         for coordinate in coordinates:
...             cell_values.append(table.iat[coordinate])
...         answers.append(", ".join(cell_values))

>>> display(table)
>>> print("")
>>> for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
...     print(query)
...     if predicted_agg == "NONE":
...         print("Predicted answer: " + answer)
...     else:
...         print("Predicted answer: " + predicted_agg + " > " + answer)
What is the name of the first actor?
Predicted answer: Brad Pitt
How many movies has George Clooney played in?
Predicted answer: COUNT > 69
What is the total number of movies?
Predicted answer: SUM > 87, 53, 69
```
</tf>

</frameworkcontent>

في حالة الإعداد المحادثي، يجب توفير كل زوج من الجدول-السؤال **بالتسلسل** إلى النموذج، بحيث يمكن الكتابة فوق أنواع رموز `prev_labels` بواسطة التنبؤات `labels` للزوج السابق من الجدول-السؤال. لمزيد من المعلومات، يمكن الرجوع إلى [هذا الدفتر](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TAPAS/Fine_tuning_TapasForQuestionAnswering_on_SQA.ipynb) (لـ PyTorch) و [هذا الدفتر](https://github.com/kamalkraj/Tapas-Tutorial/blob/master/TAPAS/Fine_tuning_TapasForQuestionAnswering_on_SQA.ipynb) (لـ TensorFlow).

## الموارد

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهمة نمذجة اللغة المُقنَّعة](../tasks/masked_language_modeling)

## المخرجات الخاصة بـ TAPAS

[[autodoc]] models.tapas.modeling_tapas.TableQuestionAnsweringOutput

## TapasConfig

[[autodoc]] TapasConfig

## TapasTokenizer

[[autodoc]] TapasTokenizer

- __call__
- convert_logits_to_predictions
- save_vocabulary

<frameworkcontent>

<pt>

## TapasModel

[[autodoc]] TapasModel

- forward

## TapasForMaskedLM

[[autodoc]] TapasForMaskedLM

- forward

## TapasForSequenceClassification

[[autodoc]] TapasForSequenceClassification

- forward

## TapasForQuestionAnswering

[[autodoc]] TapasForQuestionAnswering

- forward

</pt>

<tf>

## TFTapasModel

[[autodoc]] TFTapasModel

- call

## TFTapasForMaskedLM

[[autodoc]] TFTapasForMaskedLM

- call

## TFTapasForSequenceClassification

[[autodoc]] TFTapasForSequenceClassification

- call

## TFTapasForQuestionAnswering

[[autodoc]] TFTapasForQuestionAnswering

- call

</tf>

</frameworkcontent>