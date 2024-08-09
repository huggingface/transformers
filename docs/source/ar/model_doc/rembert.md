# RemBERT

## نظرة عامة
تم اقتراح نموذج RemBERT في ورقة "إعادة التفكير في اقتران التضمين في نماذج اللغة المُدربة مسبقًا" من قبل هيونغ وون تشونغ، وتيبو ألتويري، وهنري تساي، وميلفين جونسون، وسيباستيان رودر.

ملخص الورقة البحثية هو كما يلي:

*نعيد تقييم الممارسة القياسية لمشاركة الأوزان بين التضمينات المدخلة والمخرجة في نماذج اللغة المُدربة مسبقًا. نُظهر أن التضمينات المنفصلة توفر مرونة أكبر في النمذجة، مما يسمح لنا بتحسين كفاءة تخصيص المعلمات في التضمين المدخل للنماذج متعددة اللغات. من خلال إعادة تخصيص معلمات التضمين المدخل في طبقات المحول، نحقق أداءً أفضل بكثير في مهام فهم اللغة الطبيعية القياسية بنفس عدد المعلمات أثناء الضبط الدقيق. كما نُظهر أن تخصيص سعة إضافية للتضمين المخرج يوفر فوائد للنموذج تستمر خلال مرحلة الضبط الدقيق على الرغم من التخلص من التضمين المخرج بعد التدريب المسبق. ويُظهر تحليلنا أن التضمينات المخرجة الأكبر حجمًا تمنع الطبقات الأخيرة للنموذج من التخصص المفرط في مهمة التدريب المسبق وتشجع تمثيلات المحول على أن تكون أكثر عمومية وقابلية للنقل إلى مهام ولغات أخرى. ومن خلال الاستفادة من هذه النتائج، يمكننا تدريب نماذج تحقق أداءً قويًا في معيار XTREME دون زيادة عدد المعلمات في مرحلة الضبط الدقيق."

## نصائح الاستخدام

بالنسبة للضبط الدقيق، يمكن اعتبار RemBERT إصدارًا أكبر من mBERT مع تحليل لعامل تضمين الطبقة المشابه لـ ALBERT. لا يتم ربط التضمينات في التدريب المسبق، على عكس BERT، مما يمكّن من استخدام تضمينات مدخلة أصغر (محفوظة أثناء الضبط الدقيق) وتضمينات مخرجة أكبر (يتم التخلص منها في الضبط الدقيق). كما أن طريقة التمييز tokenization مشابهة لتلك المستخدمة في Albert بدلاً من BERT.

## الموارد

- [دليل مهام تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهام تصنيف العلامات](../tasks/token_classification)
- [دليل مهام الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهام نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهام نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل مهام الاختيار المتعدد](../tasks/multiple_choice)

## RemBertConfig

[[autodoc]] RemBertConfig

## RemBertTokenizer

[[autodoc]] RemBertTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## RemBertTokenizerFast

[[autodoc]] RemBertTokenizerFast

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

<frameworkcontent>

<pt>

## RemBertModel

[[autodoc]] RemBertModel

- forward

## RemBertForCausalLM

[[autodoc]] RemBertForCausalLM

- forward

## RemBertForMaskedLM

[[autodoc]] RemBertForMaskedLM

- forward

## RemBertForSequenceClassification

[[autodoc]] RemBertForSequenceClassification

- forward

## RemBertForMultipleChoice

[[autodoc]] RemBertForMultipleChoice

- forward

## RemBertForTokenClassification

[[autodoc]] RemBertForTokenClassification

- forward

## RemBertForQuestionAnswering

[[autodoc]] RemBertForQuestionAnswering

- forward

</pt>

<tf>

## TFRemBertModel


[[autodoc]] TFRemBertModel

- call

## TFRemBertForMaskedLM

[[autodoc]] TFRemBertForMaskedLM

- call

## TFRemBertForCausalLM

[[autodoc]] TFRemBertForCausalLM

- call

## TFRemBertForSequenceClassification

[[autodoc]] TFRemBertForSequenceClassification

- call

## TFRemBertForMultipleChoice

[[autodoc]] TFRemBertForMultipleChoice

- call

## TFRemBertForTokenClassification

[[autodoc]] TFRemBertForTokenClassification

- call

## TFRemBertForQuestionAnswering

[[autodoc]] TFRemBertForQuestionAnswering

- call

</tf>

</frameworkcontent>