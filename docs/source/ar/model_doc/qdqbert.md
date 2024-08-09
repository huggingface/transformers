# QDQBERT

> **ملاحظة:** هذا النموذج في وضع الصيانة فقط، ولا نقبل أي طلبات سحب (PRs) جديدة لتغيير شفرته البرمجية. في حال واجهتك أي مشاكل أثناء تشغيل هذا النموذج، يرجى إعادة تثبيت الإصدار الأخير الذي يدعمه: v4.40.2. يمكنك القيام بذلك عن طريق تشغيل الأمر التالي: `pip install -U transformers==4.40.2`.

## نظرة عامة
يمكن الإشارة إلى نموذج QDQBERT في الورقة البحثية [Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation](https://arxiv.org/abs/2004.09602) من تأليف هاو وو، وباتريك جود، وشياوجي تشانغ، وميخائيل إيسايف، وبوليوس ميسيكيفيشيوس.

مقدمة الورقة البحثية هي كما يلي:

> "يمكن لتقنيات التكميم أن تقلل من حجم الشبكات العصبية العميقة وتحسن من زمن الاستدلال وسرعة المعالجة من خلال الاستفادة من التعليمات البرمجية الصحيحة عالية الإنتاجية. في هذه الورقة، نراجع الجوانب الرياضية لمعلمات التكميم ونقيم خياراتها على مجموعة واسعة من نماذج الشبكات العصبية لمجالات تطبيق مختلفة، بما في ذلك الرؤية والخطاب واللغة. نركز على تقنيات التكميم التي يمكن تسريعها بواسطة المعالجات التي تحتوي على خطوط أنابيب حسابية صحيحة عالية الإنتاجية. كما نقدم سير عمل للتكميم بـ 8 بتات قادر على الحفاظ على الدقة ضمن 1% من خط الأساس العائم على جميع الشبكات المدروسة، بما في ذلك النماذج التي يصعب تكميمها، مثل MobileNets و BERT-large."

تمت المساهمة بهذا النموذج من قبل [shangz](https://huggingface.co/shangz).

## نصائح الاستخدام

- يضيف نموذج QDQBERT عمليات تكميم زائفة (زوج من عمليات التكميم الخطي/إلغاء التكميم الخطي) إلى (1) مدخلات الطبقة الخطية وأوزانها، (2) مدخلات matmul، (3) مدخلات الإضافة الباقية، في نموذج BERT.

- يتطلب QDQBERT اعتماد [Pytorch Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization). لتثبيته، استخدم الأمر التالي: `pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com`.

- يمكن تحميل نموذج QDQBERT من أي نقطة تفتيش لنموذج BERT من HuggingFace (على سبيل المثال، *google-bert/bert-base-uncased*)، وإجراء التدريب على التكميم الواعي/التكميم بعد التدريب.

- يمكن العثور على مثال كامل لاستخدام نموذج QDQBERT لإجراء التدريب على التكميم الواعي والتكميم بعد التدريب لمهمة SQUAD في [transformers/examples/research_projects/quantization-qdqbert/](examples/research_projects/quantization-qdqbert/).

### تعيين برامج التكميم الافتراضية

يضيف نموذج QDQBERT عمليات تكميم زائفة (زوج من عمليات التكميم الخطي/إلغاء التكميم الخطي) إلى نموذج BERT بواسطة `TensorQuantizer` في [Pytorch Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization). `TensorQuantizer` هو الوحدة النمطية لتكميم المصفوفات، مع `QuantDescriptor` الذي يحدد كيفية تكميم المصفوفة. راجع [دليل مستخدم Pytorch Quantization Toolkit](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html) لمزيد من التفاصيل.

قبل إنشاء نموذج QDQBERT، يجب تعيين `QuantDescriptor` الافتراضي الذي يحدد برامج تكميم المصفوفات الافتراضية.

مثال:

```python
>>> import pytorch_quantization.nn as quant_nn
>>> from pytorch_quantization.tensor_quant import QuantDescriptor

>>> # تم تعيين برنامج التكميم الافتراضي للمصفوفة لاستخدام طريقة المعايرة القصوى
>>> input_desc = QuantDescriptor(num_bits=8, calib_method="max")
>>> # تم تعيين برنامج التكميم الافتراضي للمصفوفة لتكميم القناة لكل وزن
>>> weight_desc = QuantDescriptor(num_bits=8, axis=((0,)))
>>> quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
>>> quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)
```

### المعايرة

المعايرة هي مصطلح لتمرير عينات البيانات إلى برنامج التكميم وتحديد أفضل عوامل المقياس للمصفوفات. بعد إعداد برامج تكميم المصفوفات، يمكنك استخدام المثال التالي لمعايرة النموذج:

```python
>>> # البحث عن TensorQuantizer وتمكين المعايرة
>>> for name, module in model.named_modules():
...     if name.endswith("_input_quantizer"):
...         module.enable_calib()
...         module.disable_quant() # استخدام بيانات الدقة الكاملة للمعايرة

>>> # إدخال عينات البيانات
>>> model(x)
>>> # ...

>>> # الانتهاء من المعايرة
>>> for name, module in model.named_modules():
...     if name.endswith("_input_quantizer"):
...         module.load_calib_amax()
...         module.enable_quant()

>>> # إذا كنت تعمل على GPU، فيجب استدعاء .cuda() مرة أخرى لأن المعايرة ستنشئ مصفوفات جديدة
>>> model.cuda()

>>> # استمر في تشغيل النموذج المكّم
>>> # ...
```

### التصدير إلى ONNX

الهدف من التصدير إلى ONNX هو نشر الاستدلال باستخدام [TensorRT](https://developer.nvidia.com/tensorrt). سيتم تقسيم التكميم الزائف إلى زوج من عمليات QuantizeLinear/DequantizeLinear ONNX. بعد تعيين العضو الثابت لـ TensorQuantizer لاستخدام دالات التكميم الزائفة الخاصة بـ Pytorch، يمكن تصدير النموذج المكّم زائفًا إلى ONNX، اتبع التعليمات الموجودة في [torch.onnx](https://pytorch.org/docs/stable/onnx.html). مثال:

```python
>>> from pytorch_quantization.nn import TensorQuantizer

>>> TensorQuantizer.use_fb_fake_quant = True

>>> # تحميل النموذج المعاير
>>> ...
>>> # تصدير ONNX
>>> torch.onnx.export(...)
```

## الموارد

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهمة تصنيف الرموز](../tasks/token_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهمة نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## QDQBertConfig

[[autodoc]] QDQBertConfig

## QDQBertModel

[[autodoc]] QDQBertModel

- forward

## QDQBertLMHeadModel

[[autodoc]] QDQBertLMHeadModel

- forward

## QDQBertForMaskedLM

[[autodoc]] QDQBertForMaskedLM

- forward

## QDQBertForSequenceClassification

[[autodoc]] QDQBertForSequenceClassification

- forward

## QDQBertForNextSentencePrediction

[[autodoc]] QDQBertForNextSentencePrediction

- forward

## QDQBertForMultipleChoice

[[autodoc]] QDQBertForMultipleChoice

- forward

## QDQBertForTokenClassification

[[autodoc]] QDQBertForTokenClassification

- forward

## QDQBertForQuestionAnswering

[[autodoc]] QDQBertForQuestionAnswering

- forward