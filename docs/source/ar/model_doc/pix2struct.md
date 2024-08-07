# Pix2Struct

## نظرة عامة

اقترح نموذج Pix2Struct في [Pix2Struct: Parsing Screenshot as Pretraining for Visual Language Understanding](https://arxiv.org/abs/2210.03347) بواسطة Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, Kristina Toutanova.

الملخص من الورقة هو ما يلي:

> اللغة المرئية موجودة في كل مكان - تتراوح المصادر من الكتب المدرسية مع الرسوم البيانية إلى صفحات الويب مع الصور والجداول، إلى تطبيقات الجوال مع الأزرار والنموذج. ربما بسبب هذا التنوع، اعتمد العمل السابق عادةً على وصفات خاصة بالمجال مع مشاركة محدودة للبيانات الأساسية، وبنى النماذج، والأهداف. نقدم Pix2Struct، وهو نموذج image-to-text مسبق التدريب لفهم اللغة المرئية البحتة، والذي يمكن ضبطه دقيقًا على المهام التي تحتوي على لغة مرئية. يتم تدريب Pix2Struct مسبقًا من خلال تعلم كيفية تحليل لقطات الشاشة المقنعة لصفحات الويب إلى HTML مبسط. يوفر الويب، بثرائه بالعناصر المرئية التي تنعكس بشكل نظيف في بنية HTML، مصدرًا كبيرًا لبيانات التدريب المسبق المناسبة لتنوع المهام اللاحقة. بداهة، يشمل هذا الهدف إشارات التدريب المسبق الشائعة مثل التعرف البصري على الحروف، ونمذجة اللغة، ووصف الصور. بالإضافة إلى استراتيجية التدريب المسبق الجديدة، نقدم تمثيل إدخال متغير الدقة وتكاملًا أكثر مرونة لإدخالات اللغة والرؤية، حيث يتم عرض مطالبات اللغة مثل الأسئلة مباشرةً أعلى صورة الإدخال. لأول مرة، نُظهر أن نموذجًا مسبق التدريب واحدًا يمكن أن يحقق نتائج متميزة في ستة من أصل تسع مهام عبر أربعة مجالات: المستندات، والرسوم التوضيحية، وواجهات المستخدم، والصور الطبيعية.

### نصائح:

تم ضبط Pix2Struct دقيقًا على مجموعة متنوعة من المهام ومجموعات البيانات، بدءًا من وصف الصور، والإجابة على الأسئلة المرئية (VQA) عبر مدخلات مختلفة (الكتب، والمخططات، والرسوم البيانية العلمية)، ووصف مكونات واجهة المستخدم، وما إلى ذلك. يمكن العثور على القائمة الكاملة في الجدول 1 من الورقة.

لذلك، نوصي باستخدام هذه النماذج للمهام التي تم ضبطها دقيقًا عليها. على سبيل المثال، إذا كنت تريد استخدام Pix2Struct لوصف واجهة المستخدم، فيجب استخدام النموذج الذي تم ضبطه دقيقًا على مجموعة بيانات واجهة المستخدم. إذا كنت تريد استخدام Pix2Struct لوصف الصور، فيجب استخدام النموذج الذي تم ضبطه دقيقًا على مجموعة بيانات وصف الصور الطبيعية، وهكذا.

إذا كنت تريد استخدام النموذج لأداء وصف النص الشرطي، فتأكد من استخدام المعالج باستخدام `add_special_tokens=False`.

تمت المساهمة بهذا النموذج بواسطة [ybelkada](https://huggingface.co/ybelkada).

يمكن العثور على الكود الأصلي [هنا](https://github.com/google-research/pix2struct).

## الموارد

- [دفتر ضبط دقيق](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_pix2struct.ipynb)
- [جميع النماذج](https://huggingface.co/models?search=pix2struct)

## Pix2StructConfig

[[autodoc]] Pix2StructConfig

- from_text_vision_configs

## Pix2StructTextConfig

[[autodoc]] Pix2StructTextConfig

## Pix2StructVisionConfig

[[autodoc]] Pix2StructVisionConfig

## Pix2StructProcessor

[[autodoc]] Pix2StructProcessor

## Pix2StructImageProcessor

[[autodoc]] Pix2StructImageProcessor

- preprocess

## Pix2StructTextModel

[[autodoc]] Pix2StructTextModel

- forward

## Pix2StructVisionModel

[[autodoc]] Pix2StructVisionModel

- forward

## Pix2StructForConditionalGeneration

[[autodoc]] Pix2StructForConditionalGeneration

- forward