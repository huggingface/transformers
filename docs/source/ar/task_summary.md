# ما الذي تستطيع مكتبة 🤗 Transformers القيام به؟

مكتبة 🤗 Transformers هي مجموعة من النماذج المُدرّبة مسبقًا الأفضل في فئتها لمهام معالجة اللغة الطبيعية (NLP)، ورؤية الحاسوب، ومعالجة الصوت والكلام. لا تحتوي المكتبة فقط على نماذج المحولات (Transformer) فحسب، بل تشمل أيضًا نماذج أخرى لا تعتمد على المحولات مثل الشبكات العصبية التلافيفية الحديثة لمهام رؤية الحاسوب. إذا نظرت إلى بعض المنتجات الاستهلاكية الأكثر شيوعًا اليوم، مثل الهواتف الذكية والتطبيقات وأجهزة التلفاز، فمن المحتمل أن تقف وراءها تقنية ما من تقنيات التعلم العميق. هل تريد إزالة جسم من خلفية صورة التقطتها بهاتفك الذكي؟ هذا مثال على مهمة التجزئة البانورامية (Panoptic Segmentation) ( لا تقلق إذا لم تفهم معناها بعد، فسوف نشرحها في الأقسام التالية!).

توفر هذه الصفحة نظرة عامة على مختلف مهام الكلام والصوت ورؤية الحاسوب ومعالجة اللغات الطبيعية المختلفة التي يمكن حلها باستخدام مكتبة 🤗 Transformers في ثلاثة أسطر فقط من التعليمات البرمجية!

## الصوت

تختلف مهام معالجة الصوت والكلام قليلاً عن باقي الوسائط، ويرجع ذلك ببشكل أساسي لأن الصوت كمدخل هو إشارة متصلة. على عكس النص، لا يمكن تقسيم الموجة الصوتية الخام بشكل مرتب في أجزاء منفصلة بالطريقة التي يمكن بها تقسيم الجملة إلى كلمات. وللتغلب على هذا، يتم عادةً أخذ عينات من الإشارة الصوتية الخام على فترات زمنية منتظمة. كلما زاد عدد العينات التي تؤخذ في فترة زمنية معينة، ارتفع معدل أخذ العينات (معدل التردد)، وصار الصوت أقرب إلى مصدر الصوت الأصلي.

قامت الطرق السابقة بمعالجة الصوت لاستخراج الميزات المفيدة منه. أصبح من الشائع الآن البدء بمهام معالجة الصوت والكلام عن طريق تغذية شكل الموجة الصوتية الخام مباشرة في مشفر الميزات (Feature Encoder)  لاستخراج تمثيل صوتي له. وهذا يبسط خطوة المعالجة المسبقة ويسمح للنموذج بتعلم أهم الميزات.

### تصنيف الصوت

تصنيف الصوت (Audio Classification) هو مهمة يتم فيها تصنيف بيانات الصوت الصوت من مجموعة محددة مسبقًا من الفئات. إنه فئة واسعة تضم العديد من التطبيقات المحددة، والتي تشمل:

* تصنيف المشهد الصوتي: وضع علامة على الصوت باستخدام تسمية المشهد ("المكتب"، "الشاطئ"، "الملعب")
* اكتشاف الأحداث الصوتية: وضع علامة على الصوت باستخدام تسمية حدث صوتي ("بوق السيارة"، "صوت الحوت"، "كسر زجاج")
* الوسم: وصنيف صوت يحتوي على أصوات متعددة (أصوات الطيور، وتحديد هوية المتحدث في اجتماع)
* تصنيف الموسيقى: وضع علامة على الموسيقى بتسمية النوع ("ميتال"، "هيب هوب"، "كانتري")

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="audio-classification", model="superb/hubert-base-superb-er")
>>> preds = classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.4532, 'label': 'hap'},
 {'score': 0.3622, 'label': 'sad'},
 {'score': 0.0943, 'label': 'neu'},
 {'score': 0.0903, 'label': 'ang'}]
```

### التعرف التلقائي على الكلام

يقوم التعرف التلقائي على الكلام (ASR) هو عملية تحويل الكلام إلى نص. إنه أحد أكثر المهام الصوتية شيوعًا ويرجع ذلك جزئيًا إلى أن الكلام وسيلة طبيعية للتواصل البشري. واليوم، يتم تضمين أنظمة ASR في منتجات التقنية "الذكية" مثل مكبرات الصوت والهواتف والسيارات. يمكننا أن نطلب من مساعدينا الافتراضيين تشغيل الموسيقى، وضبط التذكيرات، وإخبارنا بأحوال الطقس.
ولكن أحد التحديات الرئيسية التي ساعدت نماذج المحولات (Transformer) في التغلب عليها هو التعامل مع اللغات منخفضة الموارد. فمن خلال التدريب المسبق على كميات كبيرة من بيانات الصوتية، يُمكن ضبط النموذج بدقة (Fine-tuning) باستخدام ساعة واحدة فقط من بيانات الكلام المُوسم في لغة منخفضة الموارد إلى نتائج عالية الجودة مقارنة بأنظمة ASR السابقة التي تم تدريبها على بيانات موسومة أكثر بـ 100 مرة.

```py
>>> from transformers import pipeline

>>> transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")
>>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

## رؤية الحاسب

كانت إحدى أوائل مهام رؤية الحاسب وأنجحها هى التعرف على صور أرقام الرموز البريدية باستخدام [شبكة عصبية تلافيفية (CNN)](glossary#convolution). تتكون الصورة من وحدات بيكسل، ولكل بكسل قيمة رقمية. وهذا يجعل من السهل تمثيل صورة كمصفوفة من قيم البكسل. يصف كل مزيج معين من قيم البكسل ألوان الصورة.

هناك طريقتان عامتان يمكن من خلالهما حل مهام رؤية الحاسب:

1. استخدام الالتفافات (Convolutions) لتعلم الميزات الهرمية للصورة بدءًا من الميزات منخفضة المستوى وصولًا إلى الأشياء المجردة عالية المستوى.
2. تقسيم الصورة إلى أجزاء واستخدام نموذج المحولات (Transformer) ليتعلم تدريجياً كيف ترتبط كل جزء صورة ببعضها البعض لتشكيل صورة. على عكس النهج ا التصاعدي (Bottom-Up) الذي تفضله الشبكات العصبية التلافيفية CNN، هذا يشبه إلى حد ما البدء بصورة ضبابية ثم جعلها أوضح تدريجيًا.

### تصنيف الصور

يقوم تصنيف الصور (Image Classification) بوضع علامة على صورة كاملة من مجموعة محددة مسبقًا من الفئات. مثل معظم مهام التصنيف، هناك العديد من التطبيقات العملية لتصنيف الصور، والتي تشمل:

* الرعاية الصحية: تصنيف الصور الطبية للكشف عن الأمراض أو مراقبة صحة المريض
* البيئة: تصنيف صور الأقمار الصناعية لرصد إزالة الغابات، أو إبلاغ إدارة الأراضي البرية أو اكتشاف حرائق الغابات
* الزراعة: تصنيفر المحاصيل لمراقبة صحة النبات أو صور الأقمار الصناعية لمراقبة استخدام الأراضي
* علم البيئة: تصنيف صور الأنواع الحيوانية أو النباتية لرصد أعداد  الكائنات الحية أو تتبع الأنواع المهددة بالانقراض

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="image-classification")
>>> preds = classifier(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> print(*preds, sep="\n")
{'score': 0.4335, 'label': 'lynx, catamount'}
{'score': 0.0348, 'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'}
{'score': 0.0324, 'label': 'snow leopard, ounce, Panthera uncia'}
{'score': 0.0239, 'label': 'Egyptian cat'}
{'score': 0.0229, 'label': 'tiger cat'}
```

### كشف الأجسام

على عكس تصنيف الصور، يقوم كشف الأجسام (Object Detection) بتحديد عدة أجسام داخل صورة ومواضع هذه الأجسام في صورة (يحددها مربع الإحاطة). بعض تطبيقات كشف الأجسام تشمل:

* المركبات ذاتية القيادة: اكتشاف أجسام المرورية اليومية مثل المركبات الأخرى والمشاة وإشارات المرور
* الاستشعار عن بُعد: مراقبة الكوارث، والتخطيط الحضري، والتنبؤ بالطقس
* اكتشاف العيوب: اكتشاف الشقوق أو الأضرار الهيكلية في المباني، وعيوب التصنيع

```py
>>> from transformers import pipeline

>>> detector = pipeline(task="object-detection")
>>> preds = detector(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"], "box": pred["box"]} for pred in preds]
>>> preds
[{'score': 0.9865,
  'label': 'cat',
  'box': {'xmin': 178, 'ymin': 154, 'xmax': 882, 'ymax': 598}}]
```

### تجزئة الصور

تجزئة الصورة (Image Segmentation)  هي مهمة على مستوى البكسل تقوم بتخصيص كل بكسل في صورة لفئة معينة. إنه يختلف عن كشف الأجسام، والذي يستخدم مربعات الإحاطة (Bounding Boxes)  لتصنيف والتنبؤ بالأجسام في الصورة لأن التجزئة أكثر دقة. يمكن لتجزئة الصور اكتشاف الأجسام على مستوى البكسل. هناك عدة أنواع من تجزئة الصور:

* تجزئة مثيلات (Instance Segmentation): بالإضافة إلى تصنيف فئة كائن، فإنها تُصنّف أيضًا كل مثيل (Instance)  مميز لكائن ("الكلب-1"، "الكلب-2")
* التجزئة  البانورامية (Panoptic Segmentation): مزيج من التجزئة الدلالية (Semantic Segmentation) وتجزئة المثيلات؛ فهو تُصنّف كل بكسل مع فئة دلالية **و** كل مثيل مميز لكائن

تُعد مهام تجزئة الصور مفيدة في المركبات ذاتية القيادة على إنشاء خريطة على مستوى البكسل للعالم من حولها حتى تتمكن من التنقل بأمان حول المشاة والمركبات الأخرى. كما أنها مفيدة للتصوير الطبي، حيث يمكن للدقة العالية لهذ المهمة أن تساعد في تحديد الخلايا غير الطبيعية أو خصائص الأعضاء. يمكن أيضًا استخدام تجزئة الصور في التجارة الإلكترونية لتجربة الملابس افتراضيًا أو إنشاء تجارب الواقع المُعزز من خلال تراكب الأجسام في العالم الحقيقي من خلال الكاميرا الهاتف الخاصة بك.

```py
>>> from transformers import pipeline

>>> segmenter = pipeline(task="image-segmentation")
>>> preds = segmenter(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> print(*preds, sep="\n")
{'score': 0.9879, 'label': 'LABEL_184'}
{'score': 0.9973, 'label': 'snow'}
{'score': 0.9972, 'label': 'cat'}
```

### تقدير العمق

يقوم تقدير العمق (Depth Estimation) بالتنبؤ بمسافة كل بكسل في صورة من الكاميرا. تُعد هذه المهمة لرؤية الحاسب هذه مهمة بشكل خاص لفهم وإعادة بناء المشهد. فعلى سبيل المثال، في السيارات ذاتية القيادة، تحتاج المركبات إلى فهم مدى بُعد الأجسام مثل المشاة ولافتات المرور والمركبات الأخرى لتجنب العقبات والاصطدامات. تساعد معلومات العمق أيضًا في بناء التمثيلات ثلاثية الأبعاد من الصور ثنائية الأبعاد ويمكن استخدامها لإنشاء تمثيلات ثلاثية الأبعاد عالية الجودة للهياكل البيولوجية أو المباني.

هناك نهجان لتقدير العمق:

* التصوير المجسم (Stereo): يتم تقدير العمق عن طريق مقارنة صورتين لنفس الصورة من زوايا مختلفة قليلاً.
* التصوير الأحادي (Monocular): يتم تقدير العمق من صورة واحدة.

```py
>>> from transformers import pipeline

>>> depth_estimator = pipeline(task="depth-estimation")
>>> preds = depth_estimator(
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
... )
```

## معالجة اللغات الطبيعية

تُعد مهام معالجة اللغة الطبيعية (NLP) من بين أكثر أنواع المهام شيوعًا نظرًا لأن النص هو وسيلة طبيعية لنا للتواصل. ولكي يتمكن النموذج من فهم النص، يجب أولًا تحويله إلى صيغة رقمية. وهذا يعني تقسيم سلسلة النص إلى كلمات أو مقاطع كلمات منفصلة (رموز - Tokens)، ثم تحويل هذه الرموز إلى أرقام. ونتيجة لذلك، يمكنك تمثيل سلسلة من النص كتسلسل من الأرقام، وبمجرد حصولك على تسلسل من الأرقام، يمكن إدخاله إلى نموذج لحل جميع أنواع مهام معالجة اللغة الطبيعية!

### تصنيف النصوص

تمامًا مثل مهام التصنيف في أي مجال آخر، يقوم تصنيف  النصوص (Text Classification)  بتصنيف سلسلة نصية يمكن أن تكون جملة أو فقرة أو مستند) إلى فئة محددة مسبقًا. هناك العديد من التطبيقات العملية لتصنيف النصوص، والتي تشمل:

* تحليل المشاعر (Sentiment Analysis): تصنيف النص وفقًا لمعيار معين مثل `الإيجابية` أو `السلبية` والتي يمكن أن تُعلم وتدعم عملية صنع القرار في مجالات مثل السياسة والتمويل والتسويق
* تصنيف المحتوى (Content Classification): تصنيف النص وفقًا لبعض الموضوعات للمساعدة في تنظيم وتصفية المعلومات في الأخبار وموجزات الوسائط الاجتماعية (`الطقس`، `الرياضة`، `التمويل`، إلخ).

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="sentiment-analysis")
>>> preds = classifier("Hugging Face is the best thing since sliced bread!")
>>> preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
>>> preds
[{'score': 0.9991, 'label': 'POSITIVE'}]
```

### تصنيف الرموز

في أي مهمة من مهام معالجة اللغة الطبيعية NLP، تتم معالجة النص مسبقًا عن طريق تقسيمه إلى كلمات أو مقاطع كلمات فردية تُعرف باسم  [الرموز](glossary#token). يقوم تصنيف الرموز (Token Classification) بتخصيص تصنيف لكل رمز من مجموعة محددة مسبقًا من التصنيفات.

هناك نوعان شائعان من تصنيف الرموز:

* التعرف على الكيانات المسماة (NER):  تصنيف الرموز وفقًا لفئة الكيان مثل المنظمة أو الشخص أو الموقع أو التاريخ. يعد NER شائعًا بشكل خاص في الإعدادات الطبية الحيوية، حيث يُمكنه تصنيف الجينات والبروتينات وأسماء الأدوية.
* ترميز الأجزاء اللغوية (POS): تصنيف الرموز وفقًا للدورها النحوي مثل الاسم أو الفعل أو الصفة. POS مفيد لمساعدة أنظمة الترجمة على فهم كيفية اختلاف كلمتين متطابقتين نحويًا  (مثل كلمة "عَلَمَ" كاسم و "عَلِمَ" كفعل).

```py
>>> from transformers import pipeline

>>> classifier = pipeline(task="ner")
>>> preds = classifier("Hugging Face is a French company based in New York City.")
>>> preds = [
...     {
...         "entity": pred["entity"],
...         "score": round(pred["score"], 4),
...         "index": pred["index"],
...         "word": pred["word"],
...         "start": pred["start"],
...         "end": pred["end"],
...     }
...     for pred in preds
... ]
>>> print(*preds, sep="\n")
{'entity': 'I-ORG', 'score': 0.9968, 'index': 1, 'word': 'Hu', 'start': 0, 'end': 2}
{'entity': 'I-ORG', 'score': 0.9293, 'index': 2, 'word': '##gging', 'start': 2, 'end': 7}
{'entity': 'I-ORG', 'score': 0.9763, 'index': 3, 'word': 'Face', 'start': 8, 'end': 12}
{'entity': 'I-MISC', 'score': 0.9983, 'index': 6, 'word': 'French', 'start': 18, 'end': 24}
{'entity': 'I-LOC', 'score': 0.999, 'index': 10, 'word': 'New', 'start': 42, 'end': 45}
{'entity': 'I-LOC', 'score': 0.9987, 'index': 11, 'word': 'York', 'start': 46, 'end': 50}
{'entity': 'I-LOC', 'score': 0.9992, 'index': 12, 'word': 'City', 'start': 51, 'end': 55}
```
### الإجابة على الأسئلة

تُعدّ مهمة الإجابة عن الأسئلة (Question Answering) مهمة أخرى على مستوى الرموز (Token-Level) تُرجع إجابة لسؤال ما، وقد تعتمد هذه الإجابة على سياق (في النطاق المفتوح - Open-Domain) أو لا تعتمد على سياق (في النطاق المغلق - Closed-Domain). تحدث هذه المهمة عندما نسأل مساعدًا افتراضيًا عن شيء ما، مثل معرفة ما إذا كان مطعمٌ ما مفتوحًا. يمكن أن تُقدّم هذه المهمة أيضًا دعمًا للعملاء أو دعمًا تقنيًا، كما تُساعد محركات البحث في استرجاع المعلومات ذات الصلة التي نبحث عنها.

هناك نوعان شائعان من الإجابة على الأسئلة:

* الاستخراجية (Extractive): بالنظر إلى سؤال وسياق مُعيّن، فإن الإجابة هي مقطع نصيّ مُستخرج من السياق الذي يُحلّله النموذج.
* التجريدية (Abstractive): بالنظر إلى سؤال وسياق مُعيّن، يتم إنشاء الإجابة من السياق؛ يتعامل نهج [`Text2TextGenerationPipeline`] مع هذا النهج بدلاً من [`QuestionAnsweringPipeline`] الموضح أدناه


```py
>>> from transformers import pipeline

>>> question_answerer = pipeline(task="question-answering")
>>> preds = question_answerer(
...     question="What is the name of the repository?",
...     context="The name of the repository is huggingface/transformers",
... )
>>> print(
...     f"score: {round(preds['score'], 4)}, start: {preds['start']}, end: {preds['end']}, answer: {preds['answer']}"
... )
score: 0.9327, start: 30, end: 54, answer: huggingface/transformers
```

### التلخيص

ينشئ التلخيص (Summarization) نسخة مختصرة من نص طويل مع محاولة الحفاظ على معظم معنى النص الأصلي. التلخيص هو مهمة تسلسل إلى تسلسل(Sequence-to-Sequence)؛؛ فهو تُنتج تسلسلًا نصيًا أقصر من النص المُدخل. هناك الكثير من المستندات الطويلة التي يمكن تلخيصها لمساعدة القراء على فهم النقاط الرئيسية بسرعة. مشاريع القوانين والوثائق القانونية والمالية وبراءات الاختراع والأوراق العلمية هي مجرد أمثلة قليلة للوثائق التي يمكن تلخيصها لتوفير وقت القراء وخدمة كمساعد للقراءة.

مثل الإجابة على الأسئلة، هناك نوعان من التلخيص:

* الاستخراجية (Extractive): تحديد واستخراج أهم الجمل من النص الأصلي
* التجريدي (Abstractive): إنشاء ملخص مستهدف (الذي قد يتضمن كلمات جديدة غير موجودة في النص الأصلي) انطلاقًا من النص الأصلي؛ يستخدم نهج التلخيص التجريدي [`SummarizationPipeline`]

```py
>>> from transformers import pipeline

>>> summarizer = pipeline(task="summarization")
>>> summarizer(
...     "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles."
... )
[{'summary_text': ' The Transformer is the first sequence transduction model based entirely on attention . It replaces the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention . For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers .'}]
```

### الترجمة

تحوّل الترجمة تسلسل نص بلغة إلى لغة أخرى. من المهم مساعدة الأشخاص من خلفيات مختلفة على التواصل مع بعضهم البعض، ومساعدة المحتوى على الوصول إلى جمهور أوسع، وحتى أن يكون أداة تعليمية لمساعدة الأشخاص على تعلم لغة جديدة. إلى جانب التلخيص، تعد الترجمة مهمة من نوع تسلسل إلى تسلسل، حيث يتلقى النموذج تسلسلًا مُدخلًا ويُعيد تسلسلًا مُخرَجًا مُستهدفًا.

في الأيام الأولى، كانت نماذج الترجمة في الغالب أحادية اللغة، ولكن مؤخرًا، كان هناك اهتمام متزايد بالنماذج متعددة اللغات التي يمكنها الترجمة بين العديد من أزواج اللغات.

```py
>>> from transformers import pipeline

>>> text = "translate English to French: Hugging Face is a community-based open-source platform for machine learning."
>>> translator = pipeline(task="translation", model="google-t5/t5-small")
>>> translator(text)
[{'translation_text': "Hugging Face est une tribune communautaire de l'apprentissage des machines."}]
```

### نمذجة اللغة

نمذجة اللغة (Language Modeling) هي مهمة التنبؤ بالكلمة التالية في تسلسل نصي. لقد أصبح مهمة NLP شائعة للغاية لأن النموذج اللغوي المسبق التدريب يمكن أن يتم ضبطه بشكل دقيق للعديد من مهام الأخرى. في الآونة الأخيرة، كان هناك الكثير من الاهتمام بنماذج اللغة الكبيرة (LLMs) التي توضح التعلم من الصفر أو من عدد قليل من الأمثلة (Zero-shot or Few-shot Learning). وهذا يعني أن النموذج يمكنه حل المهام التي لم يتم تدريبه عليها بشكل صريح! يمكن استخدام نماذج اللغة لإنشاء نص سلس ومقنع، على الرغم من أنه يجب أن تكون حذرًا لأن النص قد لا يكون دائمًا دقيقًا.

هناك نوعان من نمذجة اللغة:

* السببية(Causal): هدف النموذج هو التنبؤ بالرمز (Token)  التالي في التسلسل، ويتم إخفاء الرموز المستقبلية (Masking).

```py
>>> from transformers import pipeline

>>> prompt = "Hugging Face is a community-based open-source platform for machine learning."
>>> generator = pipeline(task="text-generation")
>>> generator(prompt)  # doctest: +SKIP
```

* المقنّع (Masked): هدف النموذج هو التنبؤ برمز مُخفيّ ضمن التسلسل مع الوصول الكامل إلى الرموز  الأخرى في التسلسل

```py
>>> text = "Hugging Face is a community-based open-source <mask> for machine learning."
>>> fill_mask = pipeline(task="fill-mask")
>>> preds = fill_mask(text, top_k=1)
>>> preds = [
...     {
...         "score": round(pred["score"], 4),
...         "token": pred["token"],
...         "token_str": pred["token_str"],
...         "sequence": pred["sequence"],
...     }
...     for pred in preds
... ]
>>> preds
[{'score': 0.2236,
  'token': 1761,
  'token_str': ' platform',
  'sequence': 'Hugging Face is a community-based open-source platform for machine learning.'}]
```
  
## متعدد الوسائط:

تتطلب المهام متعددة الوسائط (Multimodal) من النموذج معالجة وسائط بيانات متعددة (نص أو صورة أو صوت أو فيديو) لحل مشكلة معينة. يعد وصف الصورة (Image Captioning) مثالاً على مهمة متعددة الوسائط حيث يأخذ النموذج صورة كمدخل وينتج تسلسل نصيًا يصف الصورة أو بعض خصائصها.

على الرغم من أن النماذج متعددة الوسائط تعمل مع أنواع أو وسائط بيانات مختلفة، إلا أن خطوات المعالجة المسبقة تساعد النموذج داخليًا على تحويل جميع أنواع البيانات إلى متجهات تضمين (Embeddings) (متجهات أو قوائم من الأرقام التي تحتوي على معلومات ذات معنى حول البيانات). بالنسبة لمهمة مثل وصف الصورة، يتعلم النموذج العلاقات بين متجهات تضمين الصور ومتجهات تضمين النص.

### الإجابة على أسئلة المستندات:

الإجابة على أسئلة المستندات  (Document Question Answering) هي مهمة تقوم بالإجابة على أسئلة اللغة الطبيعية من مستند مُعطى. على عكس مهمة الإجابة على الأسئلة على مستوى الرموز (Token-Level) التي تأخذ نصًا كمدخل، فإن الإجابة على أسئلة المستندات تأخذ صورة لمستند كمدخل بالإضافة إلى سؤال هذا حول المستند وتعيد الإجابة. يمكن استخدام الإجابة على أسئلة المستندات لتفسير المستندات المُنسّقة واستخراج المعلومات الرئيسية منها. في المثال أدناه، يمكن استخراج المبلغ الإجمالي والمبلغ المُسترد من إيصال الدفع..

```py
>>> from transformers import pipeline
>>> from PIL import Image
>>> import requests

>>> url = "https://huggingface.co/datasets/hf-internal-testing/example-documents/resolve/main/jpeg_images/2.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> doc_question_answerer = pipeline("document-question-answering", model="magorshunov/layoutlm-invoices")
>>> preds = doc_question_answerer(
...     question="ما هو المبلغ الإجمالي؟",
...     image=image,
... )
>>> preds
[{'score': 0.8531, 'answer': '17,000', 'start': 4, 'end': 4}]
```

نأمل أن تكون هذه الصفحة قد زودتك ببعض المعلومات الأساسية حول جميع أنواع المهام في كل طريقة وأهمية كل منها العملية. في القسم التالي، ستتعلم كيف تعمل مكتبة 🤗 Transformers لحل هذه المهام.