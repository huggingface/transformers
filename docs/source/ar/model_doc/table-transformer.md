# Table Transformer

## نظرة عامة

تم اقتراح نموذج Table Transformer في ورقة بحثية بعنوان [PubTables-1M: Towards comprehensive table extraction from unstructured documents](https://arxiv.org/abs/2110.00061) بواسطة Brandon Smock و Rohith Pesala و Robin Abraham. يقدم المؤلفون مجموعة بيانات جديدة تسمى PubTables-1M، لقياس التقدم في استخراج الجداول من المستندات غير المنظمة، بالإضافة إلى التعرف على بنية الجدول والتحليل الوظيفي. يقوم المؤلفون بتدريب نموذجين [DETR](detr)، أحدهما للكشف عن الجدول والآخر للتعرف على بنية الجدول، يُطلق عليهما Table Transformers.

ملخص الورقة البحثية هو كما يلي:

* "تم مؤخرًا إحراز تقدم كبير في تطبيق التعلم الآلي على مشكلة استنتاج بنية الجدول واستخراجه من المستندات غير المنظمة. ومع ذلك، يظل أحد أكبر التحديات هو إنشاء مجموعات بيانات ذات حقائق أرضية كاملة وغير غامضة على نطاق واسع. ولمعالجة هذا الأمر، نقوم بتطوير مجموعة بيانات جديدة أكثر شمولاً لاستخراج الجداول، تسمى PubTables-1M. تحتوي PubTables-1M على ما يقرب من مليون جدول من المقالات العلمية، وتدعم طرق إدخال متعددة، وتحتوي على معلومات تفصيلية حول الرؤوس ومواقع بنيات الجداول، مما يجعلها مفيدة لطائفة واسعة من أساليب النمذجة. كما يعالج مصدرًا مهمًا لعدم اتساق الحقائق الأرضية الملحوظ في مجموعات البيانات السابقة، والذي يُطلق عليه الإفراط في التجزئة، باستخدام إجراء التوحيد القياسي الجديد. نثبت أن هذه التحسينات تؤدي إلى زيادة كبيرة في أداء التدريب وتقدير أكثر موثوقية لأداء النموذج في التقييم للتعرف على بنية الجدول. علاوة على ذلك، نُظهر أن نماذج الكشف القائمة على المحول التي تم تدريبها على PubTables-1M تنتج نتائج ممتازة لجميع المهام الثلاث للكشف والتعرف على البنية والتحليل الوظيفي دون الحاجة إلى أي تخصيص خاص لهذه المهام."

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/table_transformer_architecture.jpeg"
alt="drawing" width="600"/>

<small> تم توضيح كشف الجدول والتعرف على بنية الجدول. مأخوذة من <a href="https://arxiv.org/abs/2110.00061">الورقة البحثية الأصلية</a>.</small>

قام المؤلفون بإطلاق نموذجين، أحدهما لـ [كشف الجدول](https://huggingface.co/microsoft/table-transformer-detection) في المستندات، والآخر لـ [التعرف على بنية الجدول](https://huggingface.co/microsoft/table-transformer-structure-recognition) (مهمة التعرف على الصفوف والأعمدة الفردية في جدول).

تمت المساهمة بهذا النموذج بواسطة [nielsr](https://huggingface.co/nielsr). يمكن العثور على الكود الأصلي [هنا](https://github.com/microsoft/table-transformer).

## الموارد

<PipelineTag pipeline="object-detection"/>

- يمكن العثور على دفتر ملاحظات توضيحي لنموذج Table Transformer [هنا](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Table%20Transformer).
- اتضح أن إضافة الحشو إلى الصور مهم جدًا للكشف. يمكن العثور على مناقشة مثيرة للاهتمام على GitHub مع ردود من المؤلفين [هنا](https://github.com/microsoft/table-transformer/issues/68).

## TableTransformerConfig

[[autodoc]] TableTransformerConfig

## TableTransformerModel

[[autodoc]] TableTransformerModel

- forward

## TableTransformerForObjectDetection

[[autodoc]] TableTransformerForObjectDetection

- forward