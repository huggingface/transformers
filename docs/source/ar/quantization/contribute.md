# المساهمة بطريقة تحجيم كمي جديدة

يدعم Transformers ويدمج العديد من طرق التحجيم الكمي مثل QLoRA وGPTQ وLLM.int8 وAWQ. ومع ذلك، هناك نهج أخرى للتحجيم غير مدمجة بعد. لتسهيل إضافة طرق التحجيم الكمي هذه واستخدامها مع نماذج Transformers، يجب استخدام فئة [`HfQuantizer`]. تم تصميم فئة [`HfQuantizer`] كفئة مساعدة داخلية لإضافة طريقة تحجيم بدلاً من شيء تطبقه على كل وحدة PyTorch.

سيوضح هذا الدليل كيفية دمج طريقة تحجيم كمي جديدة مع فئة [`HfQuantizer`].

## المتطلبات

قبل دمج طريقة تحجيم كمي جديدة في Transformers، تأكد من أن الطريقة التي تحاول إضافتها تلبي المتطلبات الأساسية التالية. تدعم طرق التحجيم الكمي التي يمكن تشغيلها حاليًا مع وحدات PyTorch فقط.

- طريقة التحجيم الكمي متاحة من خلال حزمة Python التي يمكن لأي شخص تثبيتها عبر pip (من الجيد أيضًا إذا كان بإمكانك تثبيت الحزمة من المصدر فقط). من الناحية المثالية، يتم تضمين نوى مسبقة التجميع في حزمة pip.
- يمكن تشغيل الطريقة على الأجهزة الشائعة الاستخدام (وحدة المعالجة المركزية، وحدة معالجة الرسوميات، ...).
- يتم لف الطريقة في `nn.Module` (على سبيل المثال، `Linear8bitLt`، `Linear4bit`)، ويجب أن يكون للطبقة الخطية المحجّمة التعريف التالي:

```py
class Linear4bit(nn.Module):
    def __init__(self, ...):
        ...
    
    def forward(self, x):
        return my_4bit_kernel(x, self.weight, self.bias)
```

بهذه الطريقة، يمكن تحجيم نماذج Transformers بسهولة عن طريق استبدال بعض مثيلات `nn.Linear` بالفصل المستهدف.

- يجب أن تكون طريقة التحجيم قابلة للتسلسل. يمكنك حفظ الأوزان المحجّمة محليًا أو دفعها إلى Hub.
- تأكد من أن الحزمة التي تحتوي على نوى التحجيم/البدائية مستقرة (بدون تغييرات متكررة وكاسرة).

بالنسبة لبعض طرق التحجيم الكمي، فقد تتطلب "التحجيم المسبق" للنماذج من خلال معايرة البيانات (مثل AWQ). في هذه الحالة، نفضل فقط دعم الاستدلال في Transformers والسماح لمكتبة الجهات الخارجية التي تحتفظ بها مجتمع ML بالتعامل مع تحجيم النموذج نفسه.

## إنشاء فئة HFQuantizer جديدة

1. قم بإنشاء فئة تكوين تحجيم جديدة داخل [src/transformers/utils/quantization_config.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/utils/quantization_config.py) وتأكد من عرض تكوين التحجيم الجديد داخل كائن المبادرة الرئيسي لـ Transformers عن طريق إضافته إلى [`_import_structure`](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/__init__.py#L1088) في [src/transformers/__init__.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/__init__.py).
2. قم بإنشاء ملف جديد داخل [src/transformers/quantizers/](https://github.com/huggingface/transformers/tree/abbffc4525566a48a9733639797c812301218b83/src/transformers/quantizers) يسمى `quantizer_your_method.py`، واجعله يرث من [src/transformers/quantizers/base.py::HfQuantizer](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/quantizers/base.py#L28). تأكد من إضافة المحجّم الجديد وتكوين التحجيم في التخطيط التلقائي للتحجيم في [src/transformers/quantizers/auto.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/quantizers/auto.py).

3. حدد سمات الفئة التالية/طرق الخصائص لطريقة التحجيم الكمي الخاصة بك:

* `requires_calibration`: ما إذا كانت طريقة التحجيم تتطلب عملية معايرة البيانات. إذا تم تعيينه على `True`، فيمكنك فقط دعم الاستدلال (مع الأوزان المحجّمة) وليس الاستدلال والتحجيم.
* `required_packages`: قائمة من السلاسل من الحزم المطلوبة لاستخدام الأوزان المحجّمة. قد تحتاج إلى تحديد بعض طرق المساعدة الجديدة مثل `is_auto_awq_available` في [transformers/src/utils/import_utils.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/utils/import_utils.py).
* `requires_parameters_quantization`: مطلوب فقط إذا كانت طريقة التحجيم الخاصة بك تتطلب اهتمامًا إضافيًا بكائن `nn.Parameter` الأساسي. على سبيل المثال، يستخدم bitsandbytes `Params4bit` و`Int8Param`، والذي يتطلب بعض الاهتمام الإضافي عند تحجيم النموذج. تقوم معظم طرق التحجيم الحديثة بتعبئة الأوزان int2/int4 داخل أوزان `torch.uint8`، لذا يجب ألا تكون هذه العلامة مطلوبة حقًا (يتم تعيينها افتراضيًا على `False`).
* `is_serializable`: طريقة خاصية لتحديد ما إذا كانت الطريقة قابلة للتسلسل أم لا.
* `is_trainable`: طريقة خاصية لتحديد ما إذا كان يمكن ضبط نماذج دقيقة أعلى طريقة التحجيم الكمي (مع أو بدون نهج PEFT).

4. اكتب أساليب `validate_environment` و`update_torch_dtype`. يتم استدعاء هذه الطرق قبل إنشاء النموذج المحجّم لضمان استخدام المستخدمين للتكوين الصحيح. يمكنك الاطلاع على كيفية القيام بذلك في المحجّمات الأخرى.

5. اكتب طريقة `_process_model_before_weight_loading`. في Transformers، يتم تهيئة النماذج المحجّمة أولاً على الجهاز "الميتا" قبل تحميل الأوزان. وهذا يعني أن طريقة `_process_model_before_weight_loading` تهتم بتشغيل هيكل النموذج لاستبدال بعض الوحدات (مثل `nn.Linear`) بالوحدات المستهدفة (وحدات التحجيم). يمكنك تحديد منطق استبدال الوحدة أو أي طريقة مساعدة أخرى عن طريق إنشاء ملف جديد في [transformers/src/integrations/](https://github.com/huggingface/transformers/tree/abbffc4525566a48a9733639797c812301218b83/src/transformers/integrations) وتعريض الطرق ذات الصلة في ملف `__init__.py` لهذا المجلد. نقطة الانطلاق الأفضل ستكون إلقاء نظرة على طرق التحجيم الكمي الأخرى مثل [quantizer_awq.py](https://github.com/huggingface/transformers/blob/abbffc4525566a48a9733639797c812301218b83/src/transformers/quantizers/quantizer_awq.py).

6. اكتب طريقة `_process_model_after_weight_loading`. تمكّن هذه الطريقة من تنفيذ ميزات إضافية تتطلب تشغيل النموذج بعد تحميل الأوزان.

7. قم بتوثيق كل شيء! تأكد من توثيق طريقة التحجيم الكمي الخاصة بك عن طريق إضافة ملف جديد ضمن `docs/source/en/quantization` وإضافة صف جديد في الجدول في `docs/source/en/quantization/overview.md`.

8. أضف الاختبارات! يجب عليك إضافة الاختبارات عن طريق إضافة الحزمة أولاً في Dockerfile الليلي الخاص بنا داخل `docker/transformers-quantization-latest-gpu` ثم إضافة ملف اختبار جديد في `tests/quantization/xxx`. لا تتردد في التحقق من كيفية تنفيذه لطرق التحجيم الكمي الأخرى.