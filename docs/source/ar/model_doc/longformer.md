# Longformer

## نظرة عامة
تم تقديم نموذج Longformer في ورقة "Longformer: The Long-Document Transformer" بواسطة Iz Beltagy و Matthew E. Peters و Arman Cohan.

مقدمة الورقة البحثية هي التالية:

*لا تستطيع النماذج المستندة إلى المحولات معالجة التسلسلات الطويلة بسبب عملية الاهتمام الذاتي، والتي تزداد تربيعيًا مع طول التسلسل. ولمعالجة هذا القيد، نقدم Longformer بآلية اهتمام تزداد خطيًا مع طول التسلسل، مما يجعل من السهل معالجة المستندات التي تحتوي على آلاف الرموز أو أكثر. وآلية الاهتمام في Longformer هي بديل مباشر لآلية الاهتمام الذاتي القياسية، وتجمع بين الاهتمام المحلي بنافذة والاهتمام العالمي المدفوع بالمهمة. وبناءً على العمل السابق حول محولات التسلسل الطويل، نقيم Longformer على النمذجة اللغوية على مستوى الأحرف ونحقق نتائج متقدمة على text8 و enwik8. وعلى عكس معظم الأعمال السابقة، نقوم أيضًا بتدريب Longformer مسبقًا وضبطها الدقيق على مجموعة متنوعة من المهام النهائية. ويفوق Longformer المُدرب مسبقًا أداء RoBERTa باستمرار في مهام المستندات الطويلة، ويحقق نتائج متقدمة جديدة في WikiHop و TriviaQA.*

تمت المساهمة بهذا النموذج من قبل [beltagy](https://huggingface.co/beltagy). يمكن العثور على كود المؤلفين [هنا](https://github.com/allenai/longformer).

## نصائح الاستخدام
- بما أن Longformer يعتمد على RoBERTa، فإنه لا يحتوي على `token_type_ids`. لا تحتاج إلى الإشارة إلى الجزء الذي ينتمي إليه الرمز. قم فقط بفصل أجزاءك باستخدام رمز الفصل `tokenizer.sep_token` (أو `</s>`).

- نموذج محول يستبدل مصفوفات الاهتمام بمصفوفات مبعثرة لتسريعها. غالبًا ما يكون السياق المحلي (على سبيل المثال، ما هو الرمزان الموجودان على اليسار واليمين؟) كافٍ لاتخاذ إجراء بالنسبة لرمز معين. لا تزال بعض الرموز المدخلة مسبقًا تحصل على اهتمام عالمي، ولكن مصفوفة الاهتمام بها عدد أقل بكثير من المعلمات، مما يؤدي إلى تسريعها. راجع قسم الاهتمام المحلي لمزيد من المعلومات.

## الاهتمام الذاتي Longformer

يستخدم الاهتمام الذاتي Longformer الاهتمام الذاتي لكل من السياق "المحلي" والسياق "العالمي". معظم الرموز لا تحضر إلا "محليًا" لبعضها البعض، مما يعني أن كل رمز يحضر إلى \\(\frac{1}{2} w\\) الرموز السابقة و \\(\frac{1}{2} w\\) الرموز اللاحقة مع \\(w\\) كونها طول النافذة كما هو محدد في `config.attention_window`. لاحظ أن `config.attention_window` يمكن أن يكون من النوع `List` لتحديد طول نافذة مختلف لكل طبقة. يحضر عدد قليل من الرموز المختارة "عالميًا" إلى جميع الرموز الأخرى، كما هو الحال تقليديًا بالنسبة لجميع الرموز في `BertSelfAttention`.

لاحظ أن الرموز التي تحضر "محليًا" و "عالميًا" يتم إسقاطها بواسطة مصفوفات استعلام ومفتاح وقيمة مختلفة. لاحظ أيضًا أن كل رمز يحضر "محليًا" لا يحضر فقط إلى الرموز الموجودة داخل نافذته \\(w\\)، ولكن أيضًا إلى جميع الرموز التي تحضر "عالميًا" بحيث يكون الاهتمام العالمي *متناظرًا*.

يمكن للمستخدم تحديد الرموز التي تحضر "محليًا" والرموز التي تحضر "عالميًا" عن طريق تعيين المصفوفة `global_attention_mask` بشكل مناسب في وقت التشغيل. تستخدم جميع نماذج Longformer المنطق التالي لـ `global_attention_mask`:

- 0: يحضر الرمز "محليًا"،
- 1: يحضر الرمز "عالميًا".

لمزيد من المعلومات، يرجى الرجوع إلى طريقة [`~LongformerModel.forward`].

باستخدام الاهتمام الذاتي Longformer، يمكن تقليل تعقيد الذاكرة والوقت لعملية الضرب في المصفوفة الاستعلامية الرئيسية، والتي تمثل عادةً عنق الزجاجة في الذاكرة والوقت، من \\(\mathcal{O}(n_s \times n_s)\\) إلى \\(\mathcal{O}(n_s \times w)\\)، مع \\(n_s\\) كونها طول التسلسل و \\(w\\) كونها متوسط حجم النافذة. من المفترض أن يكون عدد الرموز التي تحضر "عالميًا" غير مهم مقارنة بعدد الرموز التي تحضر "محليًا".

لمزيد من المعلومات، يرجى الرجوع إلى الورقة [الأصلية](https://arxiv.org/pdf/2004.05150.pdf).

## التدريب

يتم تدريب [`LongformerForMaskedLM`] بنفس طريقة تدريب [`RobertaForMaskedLM`] ويجب استخدامه على النحو التالي:

```python
input_ids = tokenizer.encode("This is a sentence from [MASK] training data", return_tensors="pt")
mlm_labels = tokenizer.encode("This is a sentence from the training data", return_tensors="pt")

loss = model(input_ids, labels=input_ids, masked_lm_labels=mlm_labels)[0]
```

## الموارد

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهمة تصنيف الرموز](../tasks/token_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة المعقدة](../tasks/masked_language_modeling)
- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## LongformerConfig

[[autodoc]] LongformerConfig

## LongformerTokenizer

[[autodoc]] LongformerTokenizer

## LongformerTokenizerFast

[[autodoc]] LongformerTokenizerFast

## المخرجات الخاصة بـ Longformer

[[autodoc]] models.longformer.modeling_longformer.LongformerBaseModelOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerBaseModelOutputWithPooling

[[autodoc]] models.longformer.modeling_longformer.LongformerMaskedLMOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerQuestionAnsweringModelOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerSequenceClassifierOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerMultipleChoiceModelOutput

[[autodoc]] models.longformer.modeling_longformer.LongformerTokenClassifierOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerBaseModelOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerBaseModelOutputWithPooling

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerMaskedLMOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerQuestionAnsweringModelOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerSequenceClassifierOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerMultipleChoiceModelOutput

[[autodoc]] models.longformer.modeling_tf_longformer.TFLongformerTokenClassifierOutput

<frameworkcontent>
<pt>

## LongformerModel

[[autodoc]] LongformerModel

- forward

## LongformerForMaskedLM

[[autodoc]] LongformerForMaskedLM

- forward

## LongformerForSequenceClassification

[[autodoc]] LongformerForSequenceClassification

- forward

## LongformerForMultipleChoice

[[autodoc]] LongformerForMultipleChoice

- forward

## LongformerForTokenClassification

[[autodoc]] LongformerForTokenClassification

- forward

## LongformerForQuestionAnswering

[[autodoc]] LongformerForQuestionAnswering

- forward

</pt>
<tf>

## TFLongformerModel

[[autodoc]] TFLongformerModel

- call

## TFLongformerForMaskedLM

[[autodoc]] TFLongformerForMaskedLM

- call

## TFLongformerForQuestionAnswering

[[autodoc]] TFLongformerForQuestionAnswering

- call

## TFLongformerForSequenceClassification

[[autodoc]] TFLongformerForSequenceClassification

- call

## TFLongformerForTokenClassification

[[autodoc]] TFLongformerForTokenClassification

- call

## TFLongformerForMultipleChoice


[[autodoc]] TFLongformerForMultipleChoice

- call

</tf>
</frameworkcontent>