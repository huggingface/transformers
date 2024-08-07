# Conditional DETR

## نظرة عامة

تم اقتراح نموذج Conditional DETR في الورقة البحثية [Conditional DETR for Fast Training Convergence](https://arxiv.org/abs/2108.06152) بواسطة Depu Meng وآخرون. يقدم Conditional DETR آلية اهتمام متقاطع مشروطة لتسريع تدريب DETR. ويحقق تقاربًا أسرع من DETR بمعدل 6.7 إلى 10 مرات.

ملخص الورقة البحثية كما يلي:

*"تقدم طريقة DETR التي تم تطويرها مؤخرًا بنية الترميز فك الترميز Transformer إلى كشف الأشياء وتحقيق أداء واعد. في هذه الورقة، نعالج مشكلة التقارب البطيء للتدريب، ونقدم آلية اهتمام متقاطع مشروطة لتسريع تدريب DETR. ويستلهم نهجنا من حقيقة أن الاهتمام المتقاطع في DETR يعتمد بشكل كبير على تضمين المحتوى لتحديد المواقع الأربعة المتطرفة والتنبؤ بالصندوق، مما يزيد من الحاجة إلى تضمينات محتوى عالية الجودة، وبالتالي صعوبة التدريب. ويقوم نهجنا، الذي يسمى Conditional DETR، بتعلم استعلام مكاني مشروط من تضمين فك الترميز للاهتمام المتقاطع متعدد الرؤوس في فك الترميز. وتتمثل الفائدة في أنه من خلال الاستعلام المكاني المشروط، يمكن لكل رأس اهتمام متقاطع أن يركز على نطاق يحتوي على منطقة مميزة، مثل أحد طرفي الكائن أو منطقة داخل صندوق الكائن. وهذا يحد من النطاق المكاني لتحديد المواقع للمناطق المميزة من أجل تصنيف الكائنات وانحدار الصندوق، وبالتالي تقليل الاعتماد على تضمينات المحتوى وتسهيل التدريب. وتظهر النتائج التجريبية أن Conditional DETR يحقق تقاربًا أسرع بمعدل 6.7 مرة لهياكل R50 و R101 الأساسية و 10 مرات لهياكل DC5-R50 و DC5-R101 الأقوى. الكود متاح على https://github.com/Atten4Vis/ConditionalDETR.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/conditional_detr_curve.jpg"
alt="drawing" width="600"/>

<small> يظهر Conditional DETR تقاربًا أسرع بكثير مقارنة بـ DETR الأصلي. مأخوذ من <a href="https://arxiv.org/abs/2108.06152">الورقة البحثية الأصلية</a>.</small>

تمت المساهمة بهذا النموذج بواسطة [DepuMeng](https://huggingface.co/DepuMeng). ويمكن العثور على الكود الأصلي [هنا](https://github.com/Atten4Vis/ConditionalDETR).

## الموارد

- يمكن العثور على البرامج النصية لضبط نموذج [`ConditionalDetrForObjectDetection`] باستخدام [`Trainer`] أو [Accelerate](https://huggingface.co/docs/accelerate/index) [هنا](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection).
- راجع أيضًا: [دليل مهمة كشف الأشياء](../tasks/object_detection).

## ConditionalDetrConfig

[[autodoc]] ConditionalDetrConfig

## ConditionalDetrImageProcessor

[[autodoc]] ConditionalDetrImageProcessor

- preprocess
- post_process_object_detection
- post_process_instance_segmentation
- post_process_semantic_segmentation
- post_process_panoptic_segmentation

## ConditionalDetrFeatureExtractor

[[autodoc]] ConditionalDetrFeatureExtractor

- __call__
- post_process_object_detection
- post_process_instance_segmentation
- post_process_semantic_segmentation
- post_process_panoptic_segmentation

## ConditionalDetrModel

[[autodoc]] ConditionalDetrModel

- forward

## ConditionalDetrForObjectDetection

[[autodoc]] ConditionalDetrForObjectDetection

- forward

## ConditionalDetrForSegmentation

[[autodoc]] ConditionalDetrForSegmentation

- forward