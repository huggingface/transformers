# EfficientFormer

<Tip warning={true}>
هذا النموذج في وضع الصيانة فقط، ولا نقبل أي PRs جديدة لتغيير شفرته.
إذا واجهتك أي مشكلات أثناء تشغيل هذا النموذج، يرجى إعادة تثبيت الإصدار الأخير الذي يدعم هذا النموذج: v4.40.2.
يمكنك القيام بذلك عن طريق تشغيل الأمر التالي: `pip install -U transformers==4.40.2`.
</Tip>

## نظرة عامة

اقترح نموذج EfficientFormer في ورقة بحثية بعنوان [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)
من قبل ياني لي، وغينغ يوان، ويانغ وين، وإريك هو، وجورجيوس إيفانجيليديس، وسيرجي تولياكوف، ويانزهي وانغ، وجيان رين. ويقترح EfficientFormer محولًا متسقًا للأبعاد يمكن تشغيله على الأجهزة المحمولة لمهام التنبؤ الكثيف مثل تصنيف الصور، والكشف عن الأشياء، والتجزئة الدلالية.

ملخص الورقة البحثية هو كما يلي:

*أظهرت محولات الرؤية (ViT) تقدمًا سريعًا في مهام الرؤية الحاسوبية، وحققت نتائج واعدة في مختلف المعايير.
ومع ذلك، بسبب العدد الهائل من المعلمات وتصميم النموذج، مثل آلية الانتباه، فإن النماذج المستندة إلى ViT تكون عادةً أبطأ بعدة مرات من الشبكات الاقتصادية المبنية على التلافيف. وبالتالي، فإن نشر ViT للتطبيقات في الوقت الفعلي صعب بشكل خاص، خاصة على الأجهزة ذات الموارد المحدودة مثل الأجهزة المحمولة. وتحاول الجهود الأخيرة تقليل تعقيد الحساب في ViT من خلال البحث المعماري للشبكة أو التصميم الهجين مع كتلة MobileNet، ولكن سرعة الاستدلال لا تزال غير مرضية. وهذا يطرح سؤالًا مهمًا: هل يمكن أن تعمل المحولات بسرعة MobileNet مع تحقيق أداء عالٍ؟
للإجابة على هذا السؤال، نقوم أولاً بإعادة النظر في بنية الشبكة والمشغلين المستخدمين في النماذج المستندة إلى ViT وتحديد التصميمات غير الفعالة.
بعد ذلك، نقدم محولًا متسقًا للأبعاد (بدون كتل MobileNet) كنموذج تصميم.
وأخيرًا، نقوم بتنفيذ التخسيس القائم على الكمون للحصول على سلسلة من النماذج النهائية التي يطلق عليها EfficientFormer.
وتظهر التجارب المستفيضة تفوق EfficientFormer في الأداء والسرعة على الأجهزة المحمولة.
ويحقق نموذجنا الأسرع، EfficientFormer-L1، دقة أعلى بنسبة 79.2٪ على ImageNet-1K مع زمن استدلال يبلغ 1.6 مللي ثانية فقط على
iPhone 12 (المترجم باستخدام CoreML)، وهو ما يعادل سرعة MobileNetV2×1.4 (1.6 مللي ثانية، 74.7٪ أعلى)، ويحصل أكبر نموذج لدينا،
EfficientFormer-L7، على دقة 83.3٪ مع زمن استجابة 7.0 مللي ثانية فقط. ويؤكد عملنا أنه يمكن للمحولات المصممة بشكل صحيح
الوصول إلى زمن استجابة منخفض للغاية على الأجهزة المحمولة مع الحفاظ على الأداء العالي.*

تمت المساهمة بهذا النموذج من قبل [novice03](https://huggingface.co/novice03) و [Bearnardd](https://huggingface.co/Bearnardd).
يمكن العثور على الشفرة الأصلية [هنا](https://github.com/snap-research/EfficientFormer). تمت إضافة إصدار TensorFlow من هذا النموذج بواسطة [D-Roberts](https://huggingface.co/D-Roberts).

## موارد التوثيق

- [دليل مهام تصنيف الصور](../tasks/image_classification)

## EfficientFormerConfig

[[autodoc]] EfficientFormerConfig

## EfficientFormerImageProcessor

[[autodoc]] EfficientFormerImageProcessor

- معالجة مسبقة

<frameworkcontent>
<pt>

## EfficientFormerModel

[[autodoc]] EfficientFormerModel

- forword

## EfficientFormerForImageClassification

[[autodoc]] EfficientFormerForImageClassification

- forword

## EfficientFormerForImageClassificationWithTeacher

[[autodoc]] EfficientFormerForImageClassificationWithTeacher

- forword

</pt>
<tf>

## TFEfficientFormerModel

[[autodoc]] TFEfficientFormerModel

- call

## TFEfficientFormerForImageClassification

[[autodoc]] TFEfficientFormerForImageClassification

- call

## TFEfficientFormerForImageClassificationWithTeacher

[[autodoc]] TFEfficientFormerForImageClassificationWithTeacher

- call

</tf>
</frameworkcontent>