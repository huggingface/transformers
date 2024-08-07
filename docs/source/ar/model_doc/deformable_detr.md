# Deformable DETR

## نظرة عامة

اقتُرح نموذج Deformable DETR في الورقة البحثية [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159) من قبل Xizhou Zhu و Weijie Su و Lewei Lu و Bin Li و Xiaogang Wang و Jifeng Dai.

يعالج Deformable DETR مشكلات بطء التقارب والحد من دقة الحيز المكاني للسمات في النموذج الأصلي [DETR](detr) من خلال الاستفادة من وحدة انتباه قابلة للتشكيل جديدة لا تهتم إلا بمجموعة صغيرة من نقاط المعاينة الرئيسية حول مرجع.

ملخص الورقة البحثية هو كما يلي:

*على الرغم من أن نموذج DETR قد اقترح مؤخرًا للتخلص من الحاجة إلى العديد من المكونات المصممة يدويًا في الكشف عن الأشياء مع إظهار أداء جيد، إلا أنه يعاني من بطء التقارب والحد من دقة الحيز المكاني للسمات، وذلك بسبب قيود وحدات اهتمام المحول في معالجة خرائط سمات الصور. وللتخفيف من هذه المشكلات، نقترح Deformable DETR، الذي تهتم وحدات الاهتمام فيه بنقاط المعاينة الرئيسية الصغيرة حول مرجع. يمكن لـ Deformable DETR تحقيق أداء أفضل من DETR (خاصة على الأجسام الصغيرة) مع تقليل عدد دورات التدريب بمقدار 10 مرات. وتظهر التجارب المستفيضة على مجموعة بيانات COCO فعالية نهجنا.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/deformable_detr_architecture.png"
alt="drawing" width="600"/>

<small> بنية Deformable DETR. مأخوذة من <a href="https://arxiv.org/abs/2010.04159">الورقة البحثية الأصلية</a>.</small>

تمت المساهمة بهذا النموذج من قبل [nielsr](https://huggingface.co/nielsr). ويمكن العثور على الكود الأصلي [هنا](https://github.com/fundamentalvision/Deformable-DETR).

## نصائح الاستخدام

- تدريب Deformable DETR مكافئ لتدريب نموذج [DETR](detr) الأصلي. راجع قسم [الموارد](#الموارد) أدناه لمفكرات الملاحظات التوضيحية.

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (المشار إليها بـ 🌎) لمساعدتك في البدء باستخدام Deformable DETR.

<PipelineTag pipeline="object-detection"/>

- يمكن العثور على مفكرات الملاحظات التوضيحية المتعلقة بالاستنتاج + الضبط الدقيق على مجموعة بيانات مخصصة لـ [`DeformableDetrForObjectDetection`] [هنا](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Deformable-DETR).

- يمكن العثور على النصوص البرمجية للضبط الدقيق لـ [`DeformableDetrForObjectDetection`] باستخدام [`Trainer`] أو [Accelerate](https://huggingface.co/docs/accelerate/index) [هنا](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection).

- راجع أيضًا: [دليل مهمة الكشف عن الأشياء](../tasks/object_detection).

إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فيرجى فتح طلب سحب Pull Request وسنراجعه! ويفضل أن يظهر المورد شيئًا جديدًا بدلاً من تكرار مورد موجود.

## DeformableDetrImageProcessor

[[autodoc]] DeformableDetrImageProcessor

- preprocess

- post_process_object_detection

## DeformableDetrFeatureExtractor

[[autodoc]] DeformableDetrFeatureExtractor

- __call__

- post_process_object_detection

## DeformableDetrConfig

[[autodoc]] DeformableDetrConfig

## DeformableDetrModel

[[autodoc]] DeformableDetrModel

- forward

## DeformableDetrForObjectDetection

[[autodoc]] DeformableDetrForObjectDetection

- forward