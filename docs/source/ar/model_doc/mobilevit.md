# MobileViT

## نظرة عامة

اقترح Sachin Mehta و Mohammad Rastegari نموذج MobileViT في [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178). ويقدم MobileViT طبقة جديدة تحل محل المعالجة المحلية في التحويلات الضمنية بمعالجة عالمية باستخدام المحولات.

الملخص المستخرج من الورقة هو كما يلي:

> "تعد الشبكات العصبية التلافيفية الخفيفة (CNNs) هي المعيار الفعلي لمهام الرؤية على الأجهزة المحمولة. تسمح لهم تحيزاتهم الفراغية بتعلم التمثيلات باستخدام عدد أقل من المعلمات عبر مهام الرؤية المختلفة. ومع ذلك، هذه الشبكات محلية مكانياً. لتعلم التمثيلات العالمية، تم اعتماد محولات الرؤية القائمة على الاهتمام الذاتي (ViTs). على عكس CNNs، تعد ViTs ثقيلة الوزن. في هذه الورقة، نسأل السؤال التالي: هل من الممكن الجمع بين نقاط القوة في CNNs و ViTs لبناء شبكة خفيفة الوزن ومنخفضة الكمون لمهام الرؤية على الأجهزة المحمولة؟ ولهذه الغاية، نقدم MobileViT، وهو محول رؤية خفيف الوزن وعام الأغراض للأجهزة المحمولة. يقدم MobileViT منظورًا مختلفًا للمعالجة العالمية للمعلومات باستخدام المحولات، أي المحولات كتحويلات. تظهر نتائجنا أن MobileViT يتفوق بشكل كبير على الشبكات القائمة على CNN و ViT عبر مهام ومجموعات بيانات مختلفة. على مجموعة بيانات ImageNet-1k، يحقق MobileViT دقة أعلى بنسبة 78.4٪ باستخدام حوالي 6 ملايين معلمة، وهو ما يمثل زيادة في الدقة بنسبة 3.2٪ و 6.2٪ عن MobileNetv3 (القائم على CNN) و DeIT (القائم على ViT) لعدد مماثل من المعلمات. في مهمة كشف الأجسام MS-COCO، يكون MobileViT أكثر دقة بنسبة 5.7٪ من MobileNetv3 لعدد مماثل من المعلمات."

ساهم [matthijs](https://huggingface.co/Matthijs) بهذا النموذج. ساهم [sayakpaul](https://huggingface.co/sayakpaul) بنسخة TensorFlow من النموذج. يمكن العثور على الكود والأوزان الأصلية [هنا](https://github.com/apple/ml-cvnets).

## نصائح الاستخدام

- يشبه MobileViT شبكات CNN أكثر من نموذج المحول. فهو لا يعمل على بيانات التسلسل، ولكن على دفعات من الصور. على عكس ViT، لا توجد تضمينات. ينتج نموذج العمود الفقري خريطة ميزات. يمكنك اتباع [هذا البرنامج التعليمي](https://keras.io/examples/vision/mobilevit) للحصول على مقدمة خفيفة الوزن.

- يمكن للمرء استخدام [`MobileViTImageProcessor`] لتحضير الصور للنموذج. لاحظ أنه إذا قمت بمعالجة مسبقة خاصة بك، فإن نقاط التحقق المسبقة التدريب تتوقع أن تكون الصور في ترتيب بكسل BGR (ليس RGB).

- تم تدريب نقاط التحقق لتصنيف الصور المتاحة مسبقًا على [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k) (يشار إليها أيضًا باسم ILSVRC 2012، وهي مجموعة من 1.3 مليون صورة و1000 فئة).

- يستخدم نموذج التجزئة رأس [DeepLabV3](https://arxiv.org/abs/1706.05587). تم تدريب نقاط التحقق المتاحة لتجزئة الصور الدلالية مسبقًا على [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).

- كما يوحي الاسم، تم تصميم MobileViT لتقديم أداء وكفاءة على الهواتف المحمولة. تتوافق إصدارات TensorFlow من نماذج MobileViT تمامًا مع [TensorFlow Lite](https://www.tensorflow.org/lite).

يمكنك استخدام الكود التالي لتحويل نقطة تحقق MobileViT (سواء كانت تصنيف الصور أو التجزئة الدلالية) لتوليد نموذج TensorFlow Lite:

```py
from transformers import TFMobileViTForImageClassification
import tensorflow as tf

model_ckpt = "apple/mobilevit-xx-small"
model = TFMobileViTForImageClassification.from_pretrained(model_ckpt)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()
tflite_filename = model_ckpt.split("/")[-1] + ".tflite"
with open(tflite_filename, "wb") as f:
    f.write(tflite_model)
```

سيكون النموذج الناتج حوالي **MB** مما يجعله مناسبًا للتطبيقات المحمولة حيث يمكن أن تكون الموارد وعرض النطاق الترددي للشبكة محدودة.

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (مشار إليها برمز 🌎) لمساعدتك في البدء باستخدام MobileViT.

<PipelineTag pipeline="image-classification"/>

- [`MobileViTForImageClassification`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) هذا.

- راجع أيضًا: [دليل مهمة تصنيف الصور](../tasks/image_classification)

**التجزئة الدلالية**

- [دليل مهام التجزئة الدلالية](../tasks/semantic_segmentation)

إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فالرجاء فتح طلب سحب وسنراجعه! يجب أن يوضح المورد بشكل مثالي شيء جديد بدلاً من تكرار مورد موجود.

## MobileViTConfig

[[autodoc]] MobileViTConfig

## MobileViTFeatureExtractor

[[autodoc]] MobileViTFeatureExtractor

- __call__

- post_process_semantic_segmentation

## MobileViTImageProcessor

[[autodoc]] MobileViTImageProcessor

- preprocess

- post_process_semantic_segmentation

<frameworkcontent>
<pt>

## MobileViTModel

[[autodoc]] MobileViTModel

- forward

## MobileViTForImageClassification

[[autodoc]] MobileViTForImageClassification

- forward

## MobileViTForSemanticSegmentation

[[autodoc]] MobileViTForSemanticSegmentation

- forward

</pt>
<tf>

## TFMobileViTModel

[[autodoc]] TFMobileViTModel

- call

## TFMobileViTForImageClassification

[[autodoc]] TFMobileViTForImageClassification

- call

## TFMobileViTForSemanticSegmentation

[[autodoc]] TFMobileViTForSemanticSegmentation

- call

</tf>
</frameworkcontent>