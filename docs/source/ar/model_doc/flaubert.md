# FlauBERT

## نظرة عامة

اقترح نموذج FlauBERT في الورقة البحثية [FlauBERT: Unsupervised Language Model Pre-training for French](https://arxiv.org/abs/1912.05372) بواسطة هانج لي وآخرون. إنه نموذج محول تم تدريبه مسبقًا باستخدام هدف نمذجة اللغة المقنعة (MLM) (مثل BERT).

وملخص الورقة هو كما يلي:

* أصبحت نماذج اللغة خطوة أساسية لتحقيق نتائج متقدمة في العديد من مهام معالجة اللغات الطبيعية المختلفة. من خلال الاستفادة من الكمية الهائلة من النصوص غير الموسومة المتاحة الآن، توفر هذه النماذج طريقة فعالة لتدريب التمثيلات الكلامية المستمرة التي يمكن ضبطها الدقيق لمهام تالية، إلى جانب سياقها على مستوى الجملة. وقد تم إثبات ذلك على نطاق واسع للغة الإنجليزية باستخدام التمثيلات السياقية (Dai and Le, 2015; Peters et al., 2018; Howard and Ruder, 2018; Radford et al., 2018; Devlin et al., 2019; Yang et al., 2019b). في هذه الورقة، نقدم ونشارك FlauBERT، وهو نموذج تم تعلمه على مجموعة بيانات فرنسية كبيرة ومتنوعة للغاية. تم تدريب نماذج ذات أحجام مختلفة باستخدام الحاسوب العملاق الجديد CNRS (المركز الوطني الفرنسي للبحث العلمي) Jean Zay. نطبق نماذج اللغة الفرنسية الخاصة بنا على مهام NLP متنوعة (تصنيف النصوص، إعادة الصياغة، الاستدلال اللغوي الطبيعي، التحليل، تحديد المعنى الدلالي للكلمة) ونظهر أنها في معظم الوقت تتفوق على أساليب التدريب الأخرى. يتم مشاركة إصدارات مختلفة من FlauBERT وكذلك بروتوكول تقييم موحد لمهام المصب، يسمى FLUE (تقييم فهم اللغة الفرنسية)، مع مجتمع البحث لإجراء المزيد من التجارب القابلة للتكرار في معالجة اللغة الطبيعية الفرنسية.

تمت المساهمة بهذا النموذج من قبل [formiel](https://huggingface.co/formiel). يمكن العثور على الكود الأصلي [هنا](https://github.com/getalp/Flaubert).

النصائح:

- مثل RoBERTa، بدون التنبؤ بترتيب الجمل (لذلك تم تدريبه فقط على هدف MLM).

## الموارد

- [دليل مهام تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهام تصنيف الرموز](../tasks/token_classification)
- [دليل مهام الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهام نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل مهام الاختيار المتعدد](../tasks/multiple_choice)

## FlaubertConfig

[[autodoc]] FlaubertConfig

## FlaubertTokenizer

[[autodoc]] FlaubertTokenizer

<frameworkcontent>
<pt>

## FlaubertModel

[[autodoc]] FlaubertModel

- forward

## FlaubertWithLMHeadModel

[[autodoc]] FlaubertWithLMHeadModel

- forward

## FlaubertForSequenceClassification

[[autodoc]] FlaubertForSequenceClassification

- forward

## FlaubertForMultipleChoice

[[autodoc]] FlaubertForMultipleChoice

- forward

## FlaubertForTokenClassification

[[autodoc]] FlaubertForTokenClassification

- forward

## FlaubertForQuestionAnsweringSimple

[[autodoc]] FlaubertForQuestionAnsweringSimple

- forward

## FlaubertForQuestionAnswering

[[autodoc]] FlaubertForQuestionAnswering

- forward

</pt>
<tf>

## TFFlaubertModel

[[autodoc]] TFFlaubertModel

- call

## TFFlaubertWithLMHeadModel

[[autodoc]] TFFlaubertWithLMHeadModel

- call

## TFFlaubertForSequenceClassification

[[autodoc]] TFFlaubertForSequenceClassification

- call

## TFFlaubertForMultipleChoice

[[autodoc]] TFFlaubertForMultipleChoice

- call

## TFFlaubertForTokenClassification

[[autodoc]] TFFlaubertForTokenClassification

- call

## TFFlaubertForQuestionAnsweringSimple


[[autodoc]] TFFlaubertForQuestionAnsweringSimple

- call

</tf>
</frameworkcontent>