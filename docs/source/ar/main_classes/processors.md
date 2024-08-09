# معالجات

يمكن أن تعني المعالجات أمرين مختلفين في مكتبة المحولات:

- الكائنات التي تقوم بمعالجة المدخلات مسبقًا للنماذج متعددة الوسائط مثل [Wav2Vec2](../model_doc/wav2vec2) (الكلام والنص) أو [CLIP](../model_doc/clip) (النص والرؤية)

- الكائنات المهملة التي كانت تستخدم في الإصدارات القديمة من المكتبة لمعالجة البيانات مسبقًا لـ GLUE أو SQUAD.

## المعالجات متعددة الوسائط

سيحتاج أي نموذج متعدد الوسائط إلى كائن لتشفير أو فك تشفير البيانات التي تجمع بين عدة طرائق (من بين النص، والرؤية، والصوت). تتم معالجة ذلك بواسطة كائنات تسمى المعالجات، والتي تجمع بين كائنين أو أكثر من كائنات المعالجة مثل المعالجات (للنمط النصي)، ومعالجات الصور (للرؤية)، ومستخلصات الميزات (للصوت).

ترث هذه المعالجات من فئة الأساس التالية التي تنفذ وظائف الحفظ والتحميل:

[[autodoc]] ProcessorMixin

## المعالجات المهملة

تتبع جميع المعالجات نفس البنية التي تتمثل في:

[`~data.processors.utils.DataProcessor`]. يعيد المعالج قائمة من

[`~data.processors.utils.InputExample`]. يمكن تحويل هذه

[`~data.processors.utils.InputExample`] إلى

[`~data.processors.utils.InputFeatures`] لتغذية النموذج.

[[autodoc]] data.processors.utils.DataProcessor

[[autodoc]] data.processors.utils.InputExample

[[autodoc]] data.processors.utils.InputFeatures

## GLUE

[تقييم الفهم اللغوي العام (GLUE)](https://gluebenchmark.com/) هو معيار مرجعي يقيم أداء النماذج عبر مجموعة متنوعة من مهام فهم اللغة الطبيعية. تم إصداره جنبًا إلى جنب مع الورقة [GLUE: معيار مرجعي للمهام المتعددة ومنصة تحليل لفهم اللغة الطبيعية](https://openreview.net/pdf?id=rJ4km2R5t7)

تستضيف هذه المكتبة ما مجموعه 10 معالجات للمهام التالية: MRPC، MNLI، MNLI (غير متطابقة)، CoLA، SST2، STSB، QQP، QNLI، RTE، وWNLI.

هذه المعالجات هي:

- [`~data.processors.utils.MrpcProcessor`]

- [`~data.processors.utils.MnliProcessor`]

- [`~data.processors.utils.MnliMismatchedProcessor`]

- [`~data.processors.utils.Sst2Processor`]

- [`~data.processors.utils.StsbProcessor`]

- [`~data.processors.utils.QqpProcessor`]

- [`~data.processors.utils.QnliProcessor`]

- [`~data.processors.utils.RteProcessor`]

- [`~data.processors.utils.WnliProcessor`]

بالإضافة إلى ذلك، يمكن استخدام الطريقة التالية لتحميل القيم من ملف بيانات وتحويلها إلى قائمة من

[`~data.processors.utils.InputExample`].

[[autodoc]] data.processors.glue.glue_convert_examples_to_features

## XNLI

[مجموعة بيانات NLI متعددة اللغات (XNLI)](https://www.nyu.edu/projects/bowman/xnli/) هي معيار مرجعي يقيم جودة التمثيلات اللغوية متعددة اللغات. XNLI هي مجموعة بيانات مستمدة من مصادر جماهيرية بناءً على [*MultiNLI*](http://www.nyu.edu/projects/bowman/multinli/): يتم وضع علامات على أزواج النصوص بوسم الاستتباع النصي للغة الطبيعية لـ 15 لغة مختلفة (بما في ذلك اللغات عالية الموارد مثل اللغة الإنجليزية واللغات منخفضة الموارد مثل السواحيلية).

تم إصداره جنبًا إلى جنب مع الورقة [XNLI: تقييم التمثيلات اللغوية متعددة اللغات](https://arxiv.org/abs/1809.05053)

تستضيف هذه المكتبة المعالج لتحميل بيانات XNLI:

- [`~data.processors.utils.XnliProcessor`]

يرجى ملاحظة أنه نظرًا لتوفر العلامات الذهبية على مجموعة الاختبار، يتم إجراء التقييم على مجموعة الاختبار.

يوجد مثال على استخدام هذه المعالجات في البرنامج النصي [run_xnli.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification/run_xnli.py).

## SQuAD

[مجموعة بيانات Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer//) هي معيار مرجعي يقيم أداء النماذج على الإجابة على الأسئلة. هناك إصداران متاحان، v1.1 وv2.0. تم إصدار الإصدار الأول (v1.1) جنبًا إلى جنب مع الورقة [SQuAD: 100,000+ أسئلة لفهم قراءة النص](https://arxiv.org/abs/1606.05250). تم إصدار الإصدار الثاني (v2.0) إلى جانب الورقة [اعرف ما لا تعرفه: أسئلة غير قابلة للإجابة لـ SQuAD](https://arxiv.org/abs/1806.03822).

تستضيف هذه المكتبة معالجًا لكل من الإصدارين:

### المعالجات

هذه المعالجات هي:

- [`~data.processors.utils.SquadV1Processor`]

- [`~data.processors.utils.SquadV2Processor`]

يرث كلاهما من الفئة المجردة [`~data.processors.utils.SquadProcessor`]

[[autodoc]] data.processors.squad.SquadProcessor

- الكل

بالإضافة إلى ذلك، يمكن استخدام الطريقة التالية لتحويل أمثلة SQuAD إلى

[`~data.processors.utils.SquadFeatures`] التي يمكن استخدامها كمدخلات للنموذج.

[[autodoc]] data.processors.squad.squad_convert_examples_to_features

يمكن استخدام هذه المعالجات وكذلك الطريقة المذكورة أعلاه مع الملفات التي تحتوي على البيانات وكذلك مع حزمة *tensorflow_datasets*. تُقدم الأمثلة أدناه.

### مثال الاستخدام

فيما يلي مثال على استخدام المعالجات وكذلك طريقة التحويل باستخدام ملفات البيانات:

```python
# تحميل معالج الإصدار الثاني
processor = SquadV2Processor()
examples = processor.get_dev_examples(squad_v2_data_dir)

# تحميل معالج الإصدار الأول
processor = SquadV1Processor()
examples = processor.get_dev_examples(squad_v1_data_dir)

features = squad_convert_examples_to_features(
examples=examples,
tokenizer=tokenizer,
max_seq_length=max_seq_length,
doc_stride=args.doc_stride,
max_query_length=max_query_length,
is_training=not evaluate,
)
```

يُعد استخدام *tensorflow_datasets* سهلاً مثل استخدام ملف البيانات:

```python
# tensorflow_datasets فقط لمعالجة الإصدار الأول من Squad.
tfds_examples = tfds.load("squad")
examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)

features = squad_convert_examples_to_features(
examples=examples,
tokenizer=tokenizer,
max_seq_length=max_seq_length,
doc_stride=args.doc_stride,
max_query_length=max_query_length,
is_training=not evaluate,
)
```

يوجد مثال آخر على استخدام هذه المعالجات في البرنامج النصي [run_squad.py](https://github.com/huggingface/transformers/tree/main/examples/legacy/question-answering/run_squad.py).