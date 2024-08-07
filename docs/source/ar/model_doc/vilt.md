# ViLT

## نظرة عامة

تم اقتراح نموذج ViLT في ورقة "ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision" من قبل Wonjae Kim و Bokyung Son و Ildoo Kim. يدمج ViLT embeddings النصية في محول رؤية (ViT)، مما يسمح له بتصميم بسيط للتعليم المسبق للرؤية واللغة (VLP).

ملخص الورقة هو كما يلي:

*"حسّن التعليم المسبق للرؤية واللغة (VLP) الأداء في العديد من مهام الرؤية واللغة المشتركة. تعتمد الأساليب الحالية لـ VLP بشكل كبير على عمليات استخراج ميزات الصور، والتي يتضمن معظمها الإشراف على المنطقة (مثل اكتشاف الأشياء) والهندسة المعمارية ذات المحولات التلافيفية (مثل ResNet). على الرغم من تجاهلها في الأدبيات، إلا أننا نجد أنها تسبب مشكلات من حيث (1) الكفاءة/السرعة، حيث يتطلب استخراج ميزات الإدخال ببساطة الكثير من العمليات الحسابية أكثر من خطوات التفاعل متعدد الوسائط؛ و (2) القوة التعبيرية، حيث أنها محدودة بالقوة التعبيرية للمُضمِّن المرئي ومفرداته المرئية المحددة مسبقًا. في هذه الورقة، نقدم نموذج VLP بسيطًا، وهو محول الرؤية واللغة (ViLT)، أحادي المعنى بمعنى أن معالجة المدخلات المرئية مبسطة بشكل كبير بنفس الطريقة الخالية من المحولات التلافيفية التي نعالج بها المدخلات النصية. نُظهر أن ViLT أسرع بعشرات المرات من نماذج VLP السابقة، ومع ذلك فهو يحقق أداءً تنافسيًا أو أفضل في مهام التدفق السفلي."*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vilt_architecture.jpg" alt="drawing" width="600"/>

<small>هندسة ViLT. مأخوذة من <a href="https://arxiv.org/abs/2102.03334">الورقة الأصلية</a>.</small>

تمت المساهمة بهذا النموذج من قبل [nielsr](https://huggingface.co/nielsr). يمكن العثور على الكود الأصلي [هنا](https://github.com/dandelin/ViLT).

## نصائح الاستخدام

- أسرع طريقة للبدء مع ViLT هي التحقق من [دفاتر الملاحظات التوضيحية](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ViLT)
(التي تعرض كل من الاستدلال والضبط الدقيق على البيانات المخصصة).

- ViLT هو نموذج يأخذ كل من `pixel_values` و`input_ids` كإدخال. يمكن للمرء استخدام [`ViltProcessor`] لتحضير البيانات للنموذج.
يغلف هذا المعالج معالج صورة (لنمط الصورة) ومعالج بيانات (للنمط اللغوي) في واحد.

- تم تدريب ViLT باستخدام صور بأحجام مختلفة: يقوم المؤلفون بإعادة حجم الحافة الأقصر للصور المدخلة إلى 384 وتحديد الحافة الأطول إلى
أقل من 640 مع الحفاظ على نسبة العرض إلى الارتفاع. للسماح بتشغيل دفعات الصور، يستخدم المؤلفون `pixel_mask` الذي يشير
أي قيم بكسل حقيقية وأي منها هو التعبئة. [`ViltProcessor`] يقوم بإنشاء هذا تلقائيًا لك.

- تصميم ViLT مشابه جدًا لتصميم محول رؤية قياسي (ViT). الفرق الوحيد هو أن النموذج يتضمن
طبقات تضمين إضافية للنمط اللغوي.

- إصدار PyTorch من هذا النموذج متاح فقط في الإصدار 1.10 من PyTorch والإصدارات الأحدث.

## ViltConfig

[[autodoc]] ViltConfig

## ViltFeatureExtractor

[[autodoc]] ViltFeatureExtractor

- __call__

## ViltImageProcessor

[[autodoc]] ViltImageProcessor

- preprocess

## ViltProcessor

[[autodoc]] ViltProcessor

- __call__

## ViltModel

[[autodoc]] ViltModel

- forward

## ViltForMaskedLM

[[autodoc]] ViltForMaskedLM

- forward

## ViltForQuestionAnswering

[[autodoc]] ViltForQuestionAnswering

- forward

## ViltForImagesAndTextClassification

[[autodoc]] ViltForImagesAndTextClassification

- forward

## ViltForImageAndTextRetrieval

[[autodoc]] ViltForImageAndTextRetrieval

- forward

## ViltForTokenClassification

[[autodoc]] ViltForTokenClassification

- forward