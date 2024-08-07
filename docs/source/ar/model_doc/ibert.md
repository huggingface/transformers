# I-BERT

## نظرة عامة

تم اقتراح نموذج I-BERT في [I-BERT: Integer-only BERT Quantization](https://arxiv.org/abs/2101.01321) بواسطة Sehoon Kim و Amir Gholami و Zhewei Yao و Michael W. Mahoney و Kurt Keutzer. إنه إصدار كمي من RoBERTa يعمل على استنتاج يصل إلى أربعة أضعاف أسرع.

الملخص من الورقة هو كما يلي:

*حققت النماذج المستندة إلى المحول، مثل BERT و RoBERTa، نتائج متميزة في العديد من مهام معالجة اللغات الطبيعية. ومع ذلك، فإن بصمتها الذاكرية، ووقت الاستدلال، واستهلاك الطاقة الخاصة بها تحد من الاستدلال الفعال على الحافة، وحتى في مركز البيانات. في حين أن الكم يمكن أن يكون حلاً قابلاً للتطبيق لهذا، يستخدم العمل السابق على تحويل النماذج الكمية الحساب ذا النقطة العائمة أثناء الاستدلال، والذي لا يمكنه الاستفادة بكفاءة من الوحدات المنطقية ذات الأعداد الصحيحة فقط مثل Tensor Cores من Turing الحديث، أو معالجات ARM التقليدية ذات الأعداد الصحيحة فقط. في هذا العمل، نقترح I-BERT، وهو نظام كمي جديد لنماذج المحول التي تقوم بكمية الاستدلال بالكامل باستخدام الحساب ذو الأعداد الصحيحة فقط. بناءً على طرق التقريب خفيفة الوزن ذات الأعداد الصحيحة فقط للعمليات غير الخطية، على سبيل المثال، GELU و Softmax و Layer Normalization، يقوم I-BERT بأداء استدلال BERT من النهاية إلى النهاية باستخدام الأعداد الصحيحة فقط دون أي حسابات ذات أرقام عائمة. نقيم نهجنا على مهام GLUE لأسفل باستخدام RoBERTa-Base/Large. نُظهر أنه في كلتا الحالتين، يحقق I-BERT دقة مماثلة (وأعلى قليلاً) مقارنة بخط الأساس عالي الدقة. علاوة على ذلك، يظهر تنفيذنا الأولي لـ I-BERT تسريعًا يتراوح من 2.4 إلى 4.0x لاستدلال INT8 على نظام GPU T4 مقارنة باستدلال FP32. تم تطوير الإطار في PyTorch وتم إصداره مفتوح المصدر.*

تمت المساهمة بهذا النموذج من قبل [kssteven](https://huggingface.co/kssteven). يمكن العثور على الكود الأصلي [هنا](https://github.com/kssteven418/I-BERT).

## الموارد

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهمة تصنيف الرموز](../tasks/token_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل المهمة متعددة الخيارات](../tasks/masked_language_modeling)

## IBertConfig

[[autodoc]] IBertConfig

## IBertModel

[[autodoc]] IBertModel

- forword

## IBertForMaskedLM

[[autodoc]] IBertForMaskedLM

- forword

## IBertForSequenceClassification

[[autodoc]] IBertForSequenceClassification

- forword

## IBertForMultipleChoice

[[autodoc]] IBertForMultipleChoice

- forword

## IBertForTokenClassification

[[autodoc]] IBertForTokenClassification

- forword

## IBertForQuestionAnswering

[[autodoc]] IBertForQuestionAnswering

- forword