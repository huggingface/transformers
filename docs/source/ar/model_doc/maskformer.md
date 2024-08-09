# MaskFormer

> هذا نموذج تم تقديمه مؤخرًا، لذلك لم يتم اختبار واجهة برمجة التطبيقات الخاصة به بشكل مكثف. قد تكون هناك بعض الأخطاء أو التغييرات الطفيفة التي قد تسبب كسر التعليمات البرمجية في المستقبل. إذا لاحظت شيئًا غريبًا، فقم بإنشاء [قضية على GitHub](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title).

## نظرة عامة

تم اقتراح نموذج MaskFormer في ورقة "Per-Pixel Classification is Not All You Need for Semantic Segmentation" بواسطة Bowen Cheng و Alexander G. Schwing و Alexander Kirillov. يعالج MaskFormer التجزئة الدلالية باستخدام نموذج تصنيف الأقنعة بدلاً من إجراء التصنيف الكلاسيكي على مستوى البكسل.

ملخص الورقة هو كما يلي:

> "تقوم الأساليب الحديثة بشكل نموذجي بصياغة التجزئة الدلالية كمهمة تصنيف لكل بكسل، في حين يتم التعامل مع التجزئة على مستوى المثيل باستخدام تصنيف الأقنعة كبديل. رؤيتنا الأساسية هي: تصنيف الأقنعة عام بما يكفي لحل مهام التجزئة الدلالية ومستوى المثيل بطريقة موحدة باستخدام نفس النموذج والخسارة وإجراء التدريب بالضبط. بناءً على هذه الملاحظة، نقترح MaskFormer، وهو نموذج تصنيف أقنعة بسيط يتوقع مجموعة من الأقنعة الثنائية، وكل منها مرتبط بتوقع تسمية فئة عالمية واحدة. بشكل عام، تبسط طريقة التصنيف القائمة على الأقنعة المقترحة مشهد الأساليب الفعالة لمهام التجزئة الدلالية والكلية، وتحقق نتائج تجريبية ممتازة. على وجه الخصوص، نلاحظ أن MaskFormer يتفوق على خطوط الأساس لتصنيف البكسل عندما يكون عدد الفئات كبيرًا. تتفوق طريقة التصنيف المستندة إلى الأقنعة الخاصة بنا على أحدث نماذج التجزئة الدلالية (55.6 mIoU على ADE20K) والتجزئة الكلية (52.7 PQ على COCO) على حد سواء."

يوضح الشكل أدناه بنية MaskFormer. مأخوذة من [الورقة الأصلية](https://arxiv.org/abs/2107.06278).

![بنية MaskFormer](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/maskformer_architecture.png)

تمت المساهمة بهذا النموذج من قبل [francesco](https://huggingface.co/francesco). يمكن العثور على الكود الأصلي [هنا](https://github.com/facebookresearch/MaskFormer).

## نصائح الاستخدام

- إن فك تشفير المحول في MaskFormer مطابق لفك تشفير [DETR](detr). أثناء التدريب، وجد مؤلفو DETR أنه من المفيد استخدام خسائر مساعدة في فك التشفير، خاصة لمساعدة النموذج على إخراج العدد الصحيح من الكائنات لكل فئة. إذا قمت بتعيين معلمة `use_auxiliary_loss` من [`MaskFormerConfig`] إلى `True`، فسيتم إضافة شبكات عصبية أمامية للتنبؤ وخسائر الهنغاري بعد كل طبقة فك تشفير (مع مشاركة FFNs للبارامترات).

- إذا كنت تريد تدريب النموذج في بيئة موزعة عبر عدة عقد، فيجب عليك تحديث وظيفة `get_num_masks` داخل فئة `MaskFormerLoss` من `modeling_maskformer.py`. عند التدريب على عدة عقد، يجب تعيين هذا إلى متوسط عدد الأقنعة المستهدفة عبر جميع العقد، كما هو موضح في التنفيذ الأصلي [هنا](https://github.com/facebookresearch/MaskFormer/blob/da3e60d85fdeedcb31476b5edd7d328826ce56cc/mask_former/modeling/criterion.py#L169).

- يمكنك استخدام [`MaskFormerImageProcessor`] لتحضير الصور للنموذج والأهداف الاختيارية للنموذج.

- للحصول على التجزئة النهائية، اعتمادًا على المهمة، يمكنك استدعاء [`~MaskFormerImageProcessor.post_process_semantic_segmentation`] أو [`~MaskFormerImageProcessor.post_process_panoptic_segmentation`]. يمكن حل كلتا المهمتين باستخدام إخراج [`MaskFormerForInstanceSegmentation`]، ويقبل التجزئة الكلية حجة `label_ids_to_fuse` الاختيارية لدمج مثيلات الكائن المستهدف (مثل السماء) معًا.

## الموارد

- يمكن العثور على جميع دفاتر الملاحظات التي توضح الاستدلال وكذلك الضبط الدقيق على البيانات المخصصة باستخدام MaskFormer [هنا](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/MaskFormer).

- يمكن العثور على البرامج النصية للضبط الدقيق لـ [`MaskFormer`] باستخدام [`Trainer`] أو [Accelerate](https://huggingface.co/docs/accelerate/index) [هنا](https://github.com/huggingface/transformers/tree/main/examples/pytorch/instance-segmentation).

## مخرجات MaskFormer المحددة

[[autodoc]] models.maskformer.modeling_maskformer.MaskFormerModelOutput

[[autodoc]] models.maskformer.modeling_maskformer.MaskFormerForInstanceSegmentationOutput

## MaskFormerConfig

[[autodoc]] MaskFormerConfig

## MaskFormerImageProcessor

[[autodoc]] MaskFormerImageProcessor

- preprocess
- encode_inputs
- post_process_semantic_segmentation
- post_process_instance_segmentation
- post_process_panoptic_segmentation

## MaskFormerFeatureExtractor

[[autodoc]] MaskFormerFeatureExtractor

- __call__
- encode_inputs
- post_process_semantic_segmentation
- post_process_instance_segmentation
- post_process_panoptic_segmentation

## MaskFormerModel

[[autodoc]] MaskFormerModel

- forward

## MaskFormerForInstanceSegmentation

[[autodoc]] MaskFormerForInstanceSegmentation

- forward