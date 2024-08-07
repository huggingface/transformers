# Mask2Former

## نظرة عامة
اقترح نموذج Mask2Former في [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527) بواسطة Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar. Mask2Former هو إطار موحد لقطاعات panoptic و instance و semantic ويتميز بتحسينات كبيرة في الأداء والكفاءة على [MaskFormer](maskformer).

مقدمة الورقة هي التالية:

*يقوم تجميع الصور بتجزئة البكسلات ذات الدلالات المختلفة، على سبيل المثال، فئة أو عضوية مثيل. كل خيار
تعرف الدلالات مهمة. في حين أن دلاليات كل مهمة تختلف فقط، يركز البحث الحالي على تصميم هندسات متخصصة لكل مهمة. نقدم Masked-attention Mask Transformer (Mask2Former)، وهو تصميم جديد قادر على معالجة أي مهمة تجزئة صور (panoptic أو instance أو semantic). تشمل مكوناته الرئيسية الانتباه المقنع، والذي يستخرج ميزات محلية عن طريق تقييد الانتباه المتقاطع داخل مناطق القناع المتوقعة. بالإضافة إلى تقليل جهد البحث ثلاث مرات على الأقل، فإنه يتفوق على أفضل العمارات المتخصصة بهامش كبير في أربع مجموعات بيانات شائعة. والأهم من ذلك، أن Mask2Former يحدد حالة جديدة لتقسيم الصور الفائقة (57.8 PQ على COCO) وتجزئة مثيلات (50.1 AP على COCO) وتجزئة دلالية (57.7 mIoU على ADE20K).*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/mask2former_architecture.jpg" alt="drawing" width="600"/>

<small> هندسة Mask2Former. مأخوذة من <a href="https://arxiv.org/abs/2112.01527">الورقة الأصلية.</a> </small>

تمت المساهمة بهذا النموذج من قبل [Shivalika Singh](https://huggingface.co/shivi) و [Alara Dirik](https://huggingface.co/adirik). يمكن العثور على الكود الأصلي [هنا](https://github.com/facebookresearch/Mask2Former).

## نصائح الاستخدام

- يستخدم Mask2Former نفس خطوات المعالجة المسبقة والمعالجة اللاحقة مثل [MaskFormer](maskformer). استخدم [`Mask2FormerImageProcessor`] أو [`AutoImageProcessor`] لإعداد الصور والأهداف الاختيارية للنموذج.

- للحصول على التجزئة النهائية، اعتمادًا على المهمة، يمكنك استدعاء [`~Mask2FormerImageProcessor.post_process_semantic_segmentation`] أو [`~Mask2FormerImageProcessor.post_process_instance_segmentation`] أو [`~Mask2FormerImageProcessor.post_process_panoptic_segmentation`]. يمكن حل المهام الثلاث جميعها باستخدام إخراج [`Mask2FormerForUniversalSegmentation`]، ويقبل تجزئة panoptic حجة اختيارية `label_ids_to_fuse` لدمج مثيلات الكائن المستهدف (مثل السماء) معًا.

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (مشار إليها بـ 🌎) لمساعدتك في البدء في استخدام Mask2Former.

- يمكن العثور على دفاتر الملاحظات التوضيحية المتعلقة بالاستدلال + ضبط Mask2Former على بيانات مخصصة [هنا](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Mask2Former).

- يمكن العثور على البرامج النصية لضبط [`Mask2Former`] باستخدام [`Trainer`] أو [Accelerate](https://huggingface.co/docs/accelerate/index) [هنا](https://github.com/huggingface/transformers/tree/main/examples/pytorch/instance-segmentation).

إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فلا تتردد في فتح طلب سحب وسنراجعه.

يجب أن يوضح المورد بشكل مثالي شيء جديد بدلاً من تكرار مورد موجود.

## Mask2FormerConfig

[[autodoc]] Mask2FormerConfig

## مخرجات خاصة بـ MaskFormer

[[autodoc]] models.mask2former.modeling_mask2former.Mask2FormerModelOutput

[[autodoc]] models.mask2former.modeling_mask2former.Mask2FormerForUniversalSegmentationOutput

## Mask2FormerModel

[[autodoc]] Mask2FormerModel

- forword

## Mask2FormerForUniversalSegmentation

[[autodoc]] Mask2FormerForUniversalSegmentation

- forword

## Mask2FormerImageProcessor

[[autodoc]] Mask2FormerImageProcessor

- preprocess 

- encode_inputs

- post_process_semantic_segmentation

- post_process_instance_segmentation

- post_process_panoptic_segmentation