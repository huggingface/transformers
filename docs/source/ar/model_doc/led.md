# LED

## نظرة عامة
اقترح نموذج LED في "Longformer: The Long-Document Transformer" بواسطة Iz Beltagy و Matthew E. Peters و Arman Cohan.

مقدمة الورقة البحثية هي التالية:

*لا تستطيع النماذج المستندة إلى المحول معالجة التسلسلات الطويلة بسبب عملية الاهتمام الذاتي، والتي تتناسب تربيعيًا مع طول التسلسل. ولمعالجة هذا القيد، نقدم Longformer بآلية اهتمام تتناسب خطيًا مع طول التسلسل، مما يسهل معالجة المستندات التي تحتوي على آلاف الرموز أو أكثر. وآلية الاهتمام Longformer هي بديل مباشر لاهتمام الذات القياسي ويجمع بين الاهتمام المحلي بنافذة والاهتمام العالمي المدفوع بالمهمة. وبناءً على العمل السابق حول محولات التسلسل الطويل، نقيم Longformer على نمذجة اللغة على مستوى الأحرف ونحقق نتائج متقدمة على text8 و enwik8. وعلى عكس معظم الأعمال السابقة، نقوم أيضًا بتدريب Longformer مسبقًا وضبط دقته على مجموعة متنوعة من المهام اللاحقة. تفوق نسخة Longformer المُدربة مسبقًا باستمرار على RoBERTa في مهام المستندات الطويلة وتحقق نتائج متقدمة جديدة في WikiHop و TriviaQA. وأخيرًا، نقدم Longformer-Encoder-Decoder (LED)، وهو متغير Longformer لدعم مهام التسلسل إلى تسلسل التوليدية الخاصة بالمستندات الطويلة، ونثبت فعاليتها في مجموعة بيانات الملخصات arXiv.*

## نصائح الاستخدام
- [`LEDForConditionalGeneration`] هو امتداد لـ
[`BartForConditionalGeneration`] باستبدال طبقة *الاهتمام الذاتي* التقليدية
بطبقة *اهتمام ذاتي مقسم* من *Longformer*. [`LEDTokenizer`] هو مرادف لـ
[`BartTokenizer`].
- يعمل LED بشكل جيد جدًا في مهام *التسلسل إلى تسلسل* طويلة المدى حيث تتجاوز `input_ids` طول 1024 رمزًا.
- يقوم LED بضبط `input_ids` ليكون مضاعفًا لـ `config.attention_window` إذا لزم الأمر. لذلك، يتم تحقيق تسريع صغير عند استخدام [`LEDTokenizer`] مع وسيط `pad_to_multiple_of`.
- يستخدم LED *الاهتمام العالمي* من خلال `global_attention_mask` (راجع
[`LongformerModel`]). بالنسبة للتلخيص، يُنصح بوضع *الاهتمام العالمي* فقط على الرمز الأول
`<s>`. بالنسبة للإجابة على الأسئلة، يُنصح بوضع *الاهتمام العالمي* على جميع رموز السؤال.
- لضبط دقة LED على 16384، يمكن تمكين *التحقق من التدرج* في حالة حدوث أخطاء في الذاكرة أثناء التدريب. يمكن القيام بذلك عن طريق تنفيذ `model.gradient_checkpointing_enable()`.
بالإضافة إلى ذلك، يمكن استخدام علم `use_cache=False`
لإيقاف تشغيل آلية التخزين المؤقت لتوفير الذاكرة.
- LED هو نموذج مع تضمين الموضع المطلق، لذلك يُنصح عادةً بضبط المدخلات على اليمين بدلاً من اليسار.

تمت المساهمة بهذا النموذج بواسطة [patrickvonplaten](https://huggingface.co/patrickvonplaten).

## الموارد
- [دفتر ملاحظات يظهر كيفية تقييم LED](https://colab.research.google.com/drive/12INTTR6n64TzS4RrXZxMSXfrOd9Xzamo?usp=sharing).
- [دفتر ملاحظات يظهر كيفية ضبط دقة LED](https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing).
- [دليل مهمة تصنيف النص](../tasks/sequence_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة الترجمة](../tasks/translation)
- [دليل مهمة الملخص](../tasks/summarization)

## LEDConfig
[[autodoc]] LEDConfig

## LEDTokenizer
[[autodoc]] LEDTokenizer
- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## LEDTokenizerFast
[[autodoc]] LEDTokenizerFast

## المخرجات الخاصة بـ LED
[[autodoc]] models.led.modeling_led.LEDEncoderBaseModelOutput
[[autodoc]] models.led.modeling_led.LEDSeq2SeqModelOutput
[[autodoc]] models.led.modeling_led.LEDSeq2SeqLMOutput
[[autodoc]] models.led.modeling_led.LEDSeq2SeqSequenceClassifierOutput
[[autodoc]] models.led.modeling_led.LEDSeq2SeqQuestionAnsweringModelOutput
[[autodoc]] models.led.modeling_tf_led.TFLEDEncoderBaseModelOutput
[[autodoc]] models.led.modeling_tf_led.TFLEDSeq2SeqModelOutput
[[autodoc]] models.led.modeling_tf_led.TFLEDSeq2SeqLMOutput

<frameworkcontent>
<pt>

## LEDModel
[[autodoc]] LEDModel
- forward

## LEDForConditionalGeneration
[[autodoc]] LEDForConditionalGeneration
- forward

## LEDForSequenceClassification
[[autodoc]] LEDForSequenceClassification
- forward

## LEDForQuestionAnswering
[[autodoc]] LEDForQuestionAnswering
- forward

</pt>
<tf>

## TFLEDModel
[[autodoc]] TFLEDModel
- call

## TFLEDForConditionalGeneration
[[autodoc]] TFLEDForConditionalGeneration
- call

</tf>
</frameworkcontent>