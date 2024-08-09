# PatchTSMixer

## نظرة عامة

اقتُرح نموذج PatchTSMixer في ورقة بحثية بعنوان [TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting](https://arxiv.org/pdf/2306.09364.pdf) بواسطة Vijay Ekambaram وArindam Jati وNam Nguyen وPhanwadee Sinthong وJayant Kalagnanam.

PatchTSMixer هو نهج نمذجة سلسلة زمنية خفيف الوزن يعتمد على بنية MLP-Mixer. وفي هذا التنفيذ من HuggingFace، نوفر قدرات PatchTSMixer لتسهيل المزج الخفيف عبر الرقع والقنوات والميزات المخفية لنمذجة السلاسل الزمنية متعددة المتغيرات بشكل فعال. كما يدعم العديد من آليات الانتباه بدءًا من الانتباه المبسط المحكوم إلى كتل الانتباه الذاتي الأكثر تعقيدًا التي يمكن تخصيصها وفقًا لذلك. يمكن تدريب النموذج مسبقًا ثم استخدامه لاحقًا في مهام مختلفة مثل التنبؤ والتصنيف والرجعية.

الملخص من الورقة هو كما يلي:

*TSMixer هو بنية عصبية خفيفة الوزن تتكون حصريًا من وحدات متعددة الطبقات (MLP) مصممة للتنبؤ والتعلم التمثيلي متعدد المتغيرات على السلاسل الزمنية المرقعة. يستلهم نموذجنا الإلهام من نجاح نماذج MLP-Mixer في رؤية الكمبيوتر. نحن نبرهن على التحديات التي ينطوي عليها تكييف رؤية MLP-Mixer للسلاسل الزمنية ونقدم مكونات تم التحقق من صحتها تجريبيًا لتحسين الدقة. ويشمل ذلك طريقة تصميم جديدة تتمثل في ربط رؤوس التسوية عبر الإنترنت مع العمود الفقري لـ MLP-Mixer، لنمذجة خصائص السلاسل الزمنية بشكل صريح مثل التسلسل الهرمي والارتباطات بين القنوات. كما نقترح نهجًا هجينًا لنمذجة القنوات للتعامل بفعالية مع تفاعلات القنوات الضجيج والتعميم عبر مجموعات بيانات متنوعة، وهو تحدٍ شائع في طرق مزج قنوات الرقع الموجودة. بالإضافة إلى ذلك، تم تقديم آلية اهتمام محكومة بسيطة في العمود الفقري لإعطاء الأولوية للميزات المهمة. من خلال دمج هذه المكونات الخفيفة، نحسن بشكل كبير من قدرة التعلم لهياكل MLP البسيطة، متجاوزين نماذج المحول المعقدة باستخدام الحد الأدنى من الاستخدام الحسابي. علاوة على ذلك، تمكن التصميم النمطي لـ TSMixer من التوافق مع كل من طرق التعلم الخاضعة للإشراف والتعلم الذاتي المقنع، مما يجعله كتلة بناء واعدة لنماذج الأساس الخاصة بالسلاسل الزمنية. يتفوق TSMixer على أحدث نماذج MLP و Transformer في التنبؤ بهامش كبير يتراوح بين 8-60%. كما يتفوق على أحدث المعايير القوية لنماذج Patch-Transformer (بنسبة 1-2%) مع تقليل كبير في الذاكرة ووقت التشغيل (2-3X).*

تمت المساهمة بهذا النموذج من قبل [ajati](https://huggingface.co/ajati)، [vijaye12](https://huggingface.co/vijaye12)، [gsinthong](https://huggingface.co/gsinthong)، [namctin](https://huggingface.co/namctin)، [wmgifford](https://huggingface.co/wmgifford)، [kashif](https://huggingface.co/kashif).

## مثال على الاستخدام

توضح مقتطفات الأكواد أدناه كيفية تهيئة نموذج PatchTSMixer بشكل عشوائي. النموذج متوافق مع [Trainer API](../trainer.md).

```python
from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction
from transformers import Trainer, TrainingArguments,

config = PatchTSMixerConfig(context_length = 512, prediction_length = 96)
model = PatchTSMixerForPrediction(config)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=valid_dataset)
trainer.train()
results = trainer.evaluate(test_dataset)
```

## نصائح الاستخدام

يمكن أيضًا استخدام النموذج لتصنيف السلاسل الزمنية والرجعية الزمنية. راجع فئات [`PatchTSMixerForTimeSeriesClassification`] و [`PatchTSMixerForRegression`] على التوالي.

## الموارد

- يمكن العثور على منشور مدونة يشرح PatchTSMixer بالتفصيل [هنا](https://huggingface.co/blog/patchtsmixer). يمكن أيضًا فتح المدونة في Google Colab.

## PatchTSMixerConfig

[[autodoc]] PatchTSMixerConfig

## PatchTSMixerModel

[[autodoc]] PatchTSMixerModel

- forward

## PatchTSMixerForPrediction

[[autodoc]] PatchTSMixerForPrediction

- forward

## PatchTSMixerForTimeSeriesClassification

[[autodoc]] PatchTSMixerForTimeSeriesClassification

- forward

## PatchTSMixerForPretraining

[[autodoc]] PatchTSMixerForPretraining

- forward

## PatchTSMixerForRegression

[[autodoc]] PatchTSMixerForRegression

- forward