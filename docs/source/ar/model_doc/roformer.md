# RoFormer

## نظرة عامة
تم اقتراح نموذج RoFormer في [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864v1.pdf) بواسطة Jianlin Su و Yu Lu و Shengfeng Pan و Bo Wen و Yunfeng Liu.

مقدمة الورقة هي التالية:

*يوفر الترميز الموضعي في بنية المحول إشرافًا لنموذج الاعتماد بين العناصر في مواضع مختلفة في التسلسل. نحن نبحث في طرق مختلفة لترميز المعلومات الموضعية في نماذج اللغة المستندة إلى المحول ونقترح تنفيذًا جديدًا يسمى Rotary Position Embedding (RoPE). يقوم RoPE المقترح بترميز المعلومات الموضعية المطلقة باستخدام مصفوفة دوران ويتضمن بشكل طبيعي اعتماد الموضع النسبي الصريح في صيغة الاهتمام الذاتي. ومن الجدير بالذكر أن RoPE يأتي مع خصائص قيمة مثل مرونة التوسع إلى أي أطوال تسلسل، وتناقص الاعتماد بين الرموز مع زيادة المسافات النسبية، والقدرة على تجهيز الاهتمام الذاتي الخطي مع ترميز الموضع النسبي. ونتيجة لذلك، يحقق المحول المعزز مع ترميز الموضع الدوار، أو RoFormer، أداءً متفوقًا في المهام التي تحتوي على نصوص طويلة. نحن نقدم التحليل النظري إلى جانب بعض نتائج التجارب الأولية على البيانات الصينية. سيتم قريبًا تحديث التجربة الجارية لبيانات المعيار باللغة الإنجليزية.*

تمت المساهمة بهذا النموذج من قبل [junnyu](https://huggingface.co/junnyu). يمكن العثور على الكود الأصلي [هنا](https://github.com/ZhuiyiTechnology/roformer).

## نصائح الاستخدام
RoFormer هو نموذج ترميز ذاتي على غرار BERT مع ترميزات موضعية دوارة. وقد أظهرت الترميزات الموضعية الدوارة تحسنًا في الأداء في مهام التصنيف التي تحتوي على نصوص طويلة.

## الموارد
- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهمة تصنيف الرموز](../tasks/token_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهمة نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## RoFormerConfig

[[autodoc]] RoFormerConfig

## RoFormerTokenizer

[[autodoc]] RoFormerTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## RoFormerTokenizerFast

[[autodoc]] RoFormerTokenizerFast

- build_inputs_with_special_tokens

<frameworkcontent>
<pt>

## RoFormerModel

[[autodoc]] RoFormerModel

- forward

## RoFormerForCausalLM

[[autodoc]] RoFormerForCausalLM

- forward

## RoFormerForMaskedLM

[[autodoc]] RoFormerForMaskedLM

- forward

## RoFormerForSequenceClassification

[[autodoc]] RoFormerForSequenceClassification

- forward

## RoFormerForMultipleChoice

[[autodoc]] RoFormerForMultipleChoice

- forward

## RoFormerForTokenClassification

[[autodoc]] RoFormerForTokenClassification

- forward

## RoFormerForQuestionAnswering

[[autodoc]] RoFormerForQuestionAnswering

- forward

</pt>
<tf>

## TFRoFormerModel

[[autodoc]] TFRoFormerModel

- call

## TFRoFormerForMaskedLM

[[autodoc]] TFRoFormerForMaskedLM

- call

## TFRoFormerForCausalLM

[[autodoc]] TFRoFormerForCausalLM

- call

## TFRoFormerForSequenceClassification

[[autodoc]] TFRoFormerForSequenceClassification

- call

## TFRoFormerForMultipleChoice


[[autodoc]] TFRoFormerForMultipleChoice

- call

## TFRoFormerForTokenClassification

[[autodoc]] TFRoFormerForTokenClassification

- call

## TFRoFormerForQuestionAnswering

[[autodoc]] TFRoFormerForQuestionAnswering

- call

</tf>
<jax>

## FlaxRoFormerModel

[[autodoc]] FlaxRoFormerModel

- __call__

## FlaxRoFormerForMaskedLM

[[autodoc]] FlaxRoFormerForMaskedLM

- __call__

## FlaxRoFormerForSequenceClassification

[[autodoc]] FlaxRoFormerForSequenceClassification

- __call__

## FlaxRoFormerForMultipleChoice

[[autodoc]] FlaxRoFormerForMultipleChoice

- __call__

## FlaxRoFormerForTokenClassification

[[autodoc]] FlaxRoFormerForTokenClassification

- __call__

## FlaxRoFormerForQuestionAnswering

[[autodoc]] FlaxRoFormerForQuestionAnswering

- __call__

</jax>
</frameworkcontent>