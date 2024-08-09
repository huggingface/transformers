# Swin Transformer

## نظرة عامة

تم اقتراح محول Swin في [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) بواسطة Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.

الملخص من الورقة هو ما يلي:

*تقدم هذه الورقة محول رؤية جديد، يسمى Swin Transformer، والذي يمكن أن يعمل كعمود فقري للأغراض العامة في مجال الرؤية الحاسوبية. تنشأ التحديات في تكييف المحول من اللغة إلى الرؤية من الاختلافات بين المجالين، مثل الاختلافات الكبيرة في مقاييس الكيانات المرئية ودقة بكسلات الصور مقارنة بالكلمات في النص. لمعالجة هذه الاختلافات، نقترح محولًا هرميًا يتم حساب تمثيله باستخدام نوافذ منزاحة. يجلب نظام النوافذ المنزاحة كفاءة أكبر من خلال الحد من حساب الاهتمام الذاتي إلى نوافذ محلية غير متداخلة مع السماح أيضًا بالاتصال عبر النوافذ. يتميز هذا التصميم الهرمي بالمرونة في النمذجة على نطاقات مختلفة وله تعقيد حسابي خطي فيما يتعلق بحجم الصورة. تجعل هذه الخصائص من محول Swin متوافقًا مع مجموعة واسعة من مهام الرؤية، بما في ذلك تصنيف الصور (87.3 دقة أعلى-1 على ImageNet-1K) ومهام التنبؤ الكثيفة مثل اكتشاف الأجسام (58.7 AP للصندوق و51.1 AP للقناع على COCO test-dev) والتجزئة الدلالية (53.5 mIoU على ADE20K val). تتفوق أداءه على الحالة السابقة للفن بهامش كبير يبلغ +2.7 AP للصندوق و+2.6 AP للقناع على COCO، و+3.2 mIoU على ADE20K، مما يثبت الإمكانات الكبيرة للنماذج المستندة إلى المحول كعمود فقري للرؤية. كما أثبت التصميم الهرمي ونهج النافذة المنزاحة فائدتهما لتصميمات MLP بالكامل.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/swin_transformer_architecture.png" alt="drawing" width="600"/>

<small>هندسة محول Swin. مأخوذة من <a href="https://arxiv.org/abs/2102.03334">الورقة الأصلية</a>.</small>

تمت المساهمة بهذا النموذج من قبل [novice03](https://huggingface.co/novice03). تمت المساهمة في إصدار Tensorflow من هذا النموذج بواسطة [amyeroberts](https://huggingface.co/amyeroberts). يمكن العثور على الكود الأصلي [هنا](https://github.com/microsoft/Swin-Transformer).

## نصائح الاستخدام

- يقوم Swin بتبطين المدخلات التي تدعم أي ارتفاع وعرض للإدخال (إذا كان قابلاً للقسمة على `32`).
- يمكن استخدام Swin كـ *عمود فقري*. عندما تكون `output_hidden_states = True`، فإنه سيخرج كلاً من `hidden_states` و`reshaped_hidden_states`. تحتوي `reshaped_hidden_states` على شكل `(batch, num_channels, height, width)` بدلاً من `(batch_size, sequence_length, num_channels)`.

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (المشار إليها بـ 🌎) لمساعدتك في البدء باستخدام محول Swin.

<PipelineTag pipeline="image-classification"/>

- [`SwinForImageClassification`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) و[دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) هذا.
- راجع أيضًا: [دليل مهام تصنيف الصور](../tasks/image_classification)

وبالإضافة إلى ذلك:

- [`SwinForMaskedImageModeling`] مدعوم من قبل [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining) هذا.

إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فلا تتردد في فتح طلب سحب وسنراجعه! يجب أن يُظهر المورد المثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

## SwinConfig

[[autodoc]] SwinConfig

<frameworkcontent>
<pt>
 ## SwinModel

[[autodoc]] SwinModel

- forward

 ## SwinForMaskedImageModeling

[[autodoc]] SwinForMaskedImageModeling

- forward

 ## SwinForImageClassification

[[autodoc]] transformers.SwinForImageClassification

- forward

</pt>
<tf>

 ## TFSwinModel

[[autodoc]] TFSwinModel

- call

 ## TFSwinForMaskedImageModeling

[[autodoc]] TFSwinForMaskedImageModeling

- call

 ## TFSwinForImageClassification

[[autodoc]] transformers.TFSwinForImageClassification

- call

</tf>
</frameworkcontent>