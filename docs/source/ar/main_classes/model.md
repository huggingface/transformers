# النماذج

تنفذ الفئات الأساسية [PreTrainedModel] و [TFPreTrainedModel] و [FlaxPreTrainedModel] الأساليب الشائعة لتحميل/حفظ نموذج إما من ملف أو دليل محلي، أو من تكوين نموذج مُدرب مسبقًا يوفره المكتبة (تم تنزيله من مستودع HuggingFace AWS S3).

كما تنفذ [PreTrainedModel] و [TFPreTrainedModel] أيضًا بعض الأساليب الشائعة بين جميع النماذج للقيام بما يلي:

- تغيير حجم تضمين الرموز المميزة للإدخال عند إضافة رموز جديدة إلى المفردات
- تقليم رؤوس الاهتمام للنموذج.

يتم تحديد الأساليب الأخرى الشائعة لكل نموذج في [~ modeling_utils.ModuleUtilsMixin] (لنماذج PyTorch) و [~ modeling_tf_utils.TFModuleUtilsMixin] (لنماذج TensorFlow) أو للجيل النصي، [~ generation.GenerationMixin] (لنماذج PyTorch)، [~ generation.TFGenerationMixin] (لنماذج TensorFlow) و [~ generation.FlaxGenerationMixin] (لنماذج Flax/JAX).

## PreTrainedModel

[[autodoc]] PreTrainedModel

- push_to_hub
- الكل

## ModuleUtilsMixin

[[autodoc]] modeling_utils.ModuleUtilsMixin

## TFPreTrainedModel

[[autodoc]] TFPreTrainedModel

- push_to_hub
- الكل

## TFModelUtilsMixin

[[autodoc]] modeling_tf_utils.TFModelUtilsMixin

## FlaxPreTrainedModel

[[autodoc]] FlaxPreTrainedModel

- push_to_hub
- الكل

## النشر على المنصة

[[autodoc]] utils.PushToHubMixin

## نقاط التفتيش المجزأة

[[autodoc]] modeling_utils.load_sharded_checkpoint