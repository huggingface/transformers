# EfficientNet

## نظرة عامة
تم اقتراح نموذج EfficientNet في ورقة [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) بواسطة Mingxing Tan و Quoc V. Le. تعد EfficientNets عائلة من نماذج تصنيف الصور، والتي تحقق دقة عالية، مع كونها أصغر وأسرع من النماذج السابقة.

ملخص الورقة البحثية هو كما يلي:

*تُطور شبكات العصبونات التلافيفية (ConvNets) عادةً بميزانية موارد ثابتة، ثم يتم توسيعها للحصول على دقة أفضل إذا كانت هناك موارد إضافية متاحة. في هذه الورقة، نقوم بدراسة منهجية لقياس النموذج وتحديد أن الموازنة الدقيقة بين عمق الشبكة وعرضها ودقتها يمكن أن تؤدي إلى أداء أفضل. وبناءً على هذه الملاحظة، نقترح طريقة جديدة لقياس النطاق الذي يوسع جميع أبعاد العمق/العرض/الدقة باستخدام معامل مركب بسيط ولكنه فعال للغاية. نثبت فعالية هذه الطريقة في توسيع MobileNets و ResNet.

وللمضي قدما، نستخدم البحث المعماري العصبي لتصميم شبكة أساس جديدة وتوسيعها للحصول على عائلة من النماذج، تسمى EfficientNets، والتي تحقق دقة وكفاءة أفضل بكثير من ConvNets السابقة. على وجه الخصوص، يحقق نموذجنا EfficientNet-B7 دقة أعلى بنسبة 84.3٪ في تصنيف الصور ImageNet، بينما يكون أصغر 8.4 مرة وأسرع 6.1 مرة في الاستدلال من أفضل ConvNet موجود. كما أن نماذج EfficientNets الخاصة بنا تنتقل بشكل جيد وتحقق دقة عالية على مجموعات بيانات التعلم بالانتقال CIFAR-100 (91.7٪) و Flowers (98.8٪) وخمس مجموعات بيانات أخرى للتعلم بالانتقال، مع عدد أقل من المعلمات.*

تمت المساهمة بهذا النموذج من قبل [adirik](https://huggingface.co/adirik). يمكن العثور على الكود الأصلي [هنا](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet).

## EfficientNetConfig

[[autodoc]] EfficientNetConfig

## EfficientNetImageProcessor

[[autodoc]] EfficientNetImageProcessor

- preprocess

## EfficientNetModel

[[autodoc]] EfficientNetModel

- forward

## EfficientNetForImageClassification

[[autodoc]] EfficientNetForImageClassification

- forward