# UPerNet

## نظرة عامة

تم اقتراح نموذج UPerNet في [Unified Perceptual Parsing for Scene Understanding](https://arxiv.org/abs/1807.10221) بواسطة Tete Xiao و Yingcheng Liu و Bolei Zhou و Yuning Jiang و Jian Sun. UPerNet هو إطار عام لتجزئة مجموعة واسعة من المفاهيم من الصور بشكل فعال، والاستفادة من أي عمود رؤية مثل [ConvNeXt](convnext) أو [Swin](swin).

المستخلص من الورقة هو ما يلي:

> *يتمكن البشر من إدراك العالم المرئي على مستويات متعددة: فنحن نصنف المشاهد ونكتشف الأشياء داخلها بسهولة، بينما نحدد أيضًا نسيج الأشياء وأسطحها إلى جانب أجزائها التركيبية المختلفة. في هذه الورقة، نقوم بدراسة مهمة جديدة تسمى Unified Perceptual Parsing، والتي تتطلب أنظمة رؤية الآلة للتعرف على أكبر عدد ممكن من المفاهيم المرئية من صورة معينة. تم تطوير إطار عمل متعدد المهام يسمى UPerNet واستراتيجية تدريب للتعلم من التعليقات التوضيحية للصور غير المتجانسة. نقوم بضبط إطار عملنا على Unified Perceptual Parsing ونظهر أنه قادر على تجزئة مجموعة واسعة من المفاهيم من الصور بشكل فعال. يتم تطبيق الشبكات المدربة بشكل أكبر على اكتشاف المعرفة المرئية في المشاهد الطبيعية.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/upernet_architecture.jpg" alt="drawing" width="600"/>

<small>إطار عمل UPerNet. مأخوذة من <a href="https://arxiv.org/abs/1807.10221">الورقة الأصلية</a>.</small>

تمت المساهمة بهذا النموذج من قبل [nielsr](https://huggingface.co/nielsr). يعتمد الكود الأصلي على OpenMMLab's mmsegmentation [here](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/uper_head.py).

## أمثلة الاستخدام

UPerNet هو إطار عام للتجزئة الدلالية. يمكن استخدامه مع أي عمود رؤية، مثل ما يلي:

```py
from transformers import SwinConfig, UperNetConfig, UperNetForSemanticSegmentation

backbone_config = SwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"])

config = UperNetConfig(backbone_config=backbone_config)
model = UperNetForSemanticSegmentation(config)
```

لاستخدام عمود رؤية آخر، مثل [ConvNeXt](convnext)، قم ببساطة بإنشاء مثيل للنموذج مع العمود الفقري المناسب:

```py
from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation

backbone_config = ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"])

config = UperNetConfig(backbone_config=backbone_config)
model = UperNetForSemanticSegmentation(config)
```

لاحظ أن هذا سيقوم بإنشاء جميع أوزان النموذج بشكل عشوائي.

## الموارد

قائمة بموارد Hugging Face الرسمية والمجتمعية (مشار إليها بـ 🌎) لمساعدتك في البدء باستخدام UPerNet.

- يمكن العثور على دفاتر الملاحظات التوضيحية لـ UPerNet [هنا](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/UPerNet).
- يتم دعم [`UperNetForSemanticSegmentation`] بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/semantic_segmentation.ipynb) هذا.
- راجع أيضًا: [دليل مهام التجزئة الدلالية](../tasks/semantic_segmentation)

إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فالرجاء فتح طلب سحب وسنقوم بمراجعته! يجب أن يُظهر المورد بشكل مثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

## UperNetConfig

[[autodoc]] UperNetConfig

## UperNetForSemanticSegmentation

[[autodoc]] UperNetForSemanticSegmentation

- forward