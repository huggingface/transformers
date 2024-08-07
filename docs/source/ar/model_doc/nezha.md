# Nezha

<Tip warning={true}>
تم وضع هذا النموذج في وضع الصيانة فقط، ولن نقبل أي طلبات سحب (PRs) جديدة لتغيير شفرته. إذا واجهتك أي مشكلات أثناء تشغيل هذا النموذج، يرجى إعادة تثبيت الإصدار الأخير الذي يدعم هذا النموذج: v4.40.2. يمكنك القيام بذلك عن طريق تشغيل الأمر التالي: `pip install -U transformers==4.40.2`.
</Tip>

## نظرة عامة
اقترح Junqiu Wei وآخرون نموذج Nezha في ورقة البحث [NEZHA: Neural Contextualized Representation for Chinese Language Understanding](https://arxiv.org/abs/1909.00204). فيما يلي الملخص المستخرج من الورقة:

*حققت نماذج اللغة المعالجة مسبقًا نجاحات كبيرة في مختلف مهام فهم اللغة الطبيعية (NLU) بسبب قدرتها على التقاط المعلومات السياقية العميقة في النص من خلال المعالجة المسبقة على نصوص ضخمة. في هذا التقرير الفني، نقدم ممارستنا في معالجة النماذج اللغوية المسماة NEZHA (التمثيل السياقي العصبي لفهم اللغة الصينية) على النصوص الصينية وضبطها الدقيق لمهام NLU الصينية. يعتمد الإصدار الحالي من NEZHA على BERT مع مجموعة من التحسينات المثبتة، والتي تشمل الترميز الموضعي النسبي الوظيفي كنظام ترميز موضعي فعال، واستراتيجية القناع لكلمة كاملة، والمعالجة الدقيقة المختلطة، ومُحَسِّن LAMB عند تدريب النماذج. وتظهر النتائج التجريبية أن NEZHA يحقق أفضل الأداء عند ضبطه الدقيق على عدة مهام صينية تمثيلية، بما في ذلك التعرف على الكيانات المسماة (NER في صحيفة الشعب اليومية)، ومطابقة الجمل (LCQMC)، وتصنيف المشاعر الصينية (ChnSenti)، والاستدلال اللغوي الطبيعي (XNLI).*

تمت المساهمة بهذا النموذج من قبل [sijunhe](https://huggingface.co/sijunhe). يمكن العثور على الشفرة الأصلية [هنا](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-PyTorch).

## الموارد
- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهمة تصنيف العلامات](../tasks/token_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## NezhaConfig
[[autodoc]] NezhaConfig

## NezhaModel
[[autodoc]] NezhaModel
- forward

## NezhaForPreTraining
[[autodoc]] NezhaForPreTraining
- forward

## NezhaForMaskedLM
[[autodoc]] NezhaForMaskedLM
- forward

## NezhaForNextSentencePrediction
[[autodoc]] NezhaForNextSentencePrediction
- forward

## NezhaForSequenceClassification
[[autodoc]] NezhaForSequenceClassification
- forward

## NezhaForMultipleChoice
[[autodoc]] NezhaForMultipleChoice
- forward

## NezhaForTokenClassification
[[autodoc]] NezhaForTokenClassification
- forward

## NezhaForQuestionAnswering
[[autodoc]] NezhaForQuestionAnswering
- forward