# ELECTRA

## نظرة عامة

تم اقتراح نموذج ELECTRA في الورقة البحثية [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than
Generators](https://openreview.net/pdf?id=r1xMH1BtvB). ELECTRA هو أسلوب جديد للتعليم المسبق يقوم بتدريب نموذجين من نوع المحول: المولد والمحقق. يتمثل دور المولد في استبدال الرموز في تسلسل، وبالتالي يتم تدريبه كنموذج لغة مقنعة. أما المحقق، وهو النموذج الذي يهمنا، فيحاول تحديد الرموز التي استبدلها المولد في التسلسل.

ملخص الورقة البحثية هو كما يلي:

> تقوم طرق التعليم المسبق لـ MLM (Masked Language Modeling) مثل BERT بإفساد الإدخال عن طريق استبدال بعض الرموز بـ [MASK] ثم تدريب نموذج لإعادة بناء الرموز الأصلية. وعلى الرغم من أنها تنتج نتائج جيدة عند نقلها إلى مهام NLP لأسفل، إلا أنها تتطلب عمومًا كميات كبيرة من الحساب لتكون فعالة. وبدلاً من ذلك، نقترح مهمة تعليم مسبق أكثر كفاءة في العينات تسمى كشف الرمز المُستبدل. بدلاً من قناع الإدخال، يقوم نهجنا بإفساده عن طريق استبدال بعض الرموز ببدائل مقنعة يتم أخذ عينات منها من شبكة مولد صغيرة. ثم، بدلاً من تدريب نموذج يتنبأ بالهويات الأصلية للرموز الفاسدة، نقوم بتدريب نموذج تمييزي يتنبأ بما إذا كان كل رمز في الإدخال الفاسد قد تم استبداله بواسطة عينة من المولد أم لا. تُظهر التجارب الشاملة أن مهمة التعليم المسبق الجديدة هذه أكثر كفاءة من MLM لأن المهمة محددة لجميع رموز الإدخال بدلاً من المجموعة الفرعية الصغيرة التي تم قناعها. ونتيجة لذلك، فإن التمثيلات السياقية التي تعلمها نهجنا تتفوق بشكل كبير على تلك التي تعلمها BERT نظرًا لنفس حجم النموذج والبيانات والحساب. وتكون المكاسب قوية بشكل خاص بالنسبة للنماذج الصغيرة؛ على سبيل المثال، نقوم بتدريب نموذج على وحدة معالجة رسومات واحدة لمدة 4 أيام تتفوق على GPT (تم تدريبه باستخدام 30 ضعف الحساب) في معيار الفهم اللغوي GLUE. كما يعمل نهجنا بشكل جيد على نطاق واسع، حيث يؤدي أداءً مماثلاً لـ RoBERTa و XLNet باستخدام أقل من 1/4 من حساباتهما ويتفوق عليهما عند استخدام نفس الكمية من الحساب.

تمت المساهمة بهذا النموذج من قبل [lysandre](https://huggingface.co/lysandre). يمكن العثور على الكود الأصلي [هنا](https://github.com/google-research/electra).

## نصائح الاستخدام

- ELECTRA هو نهج التعليم المسبق، وبالتالي لا يوجد أي تغييرات تقريبًا في النموذج الأساسي: BERT. التغيير الوحيد هو فصل حجم تضمين وحجم المخفي: حجم التضمين أصغر عمومًا، في حين أن حجم المخفي أكبر. تُستخدم طبقة إسقاط إضافية (خطية) لإسقاط التضمينات من حجم التضمين إلى حجم المخفي. في الحالة التي يكون فيها حجم التضمين هو نفسه حجم المخفي، لا تُستخدم أي طبقة إسقاط.

- ELECTRA هو نموذج محول مُدرب مسبقًا باستخدام نموذج لغة مقنع آخر (صغير). يتم إفساد الإدخالات بواسطة نموذج اللغة هذا، والذي يأخذ نصًا مقنعًا عشوائيًا كإدخال ويقوم بإخراج نص يجب على ELECTRA التنبؤ به والذي هو رمز أصلي والذي تم استبداله. مثل تدريب GAN، يتم تدريب نموذج اللغة الصغيرة لبضع خطوات (ولكن باستخدام النصوص الأصلية كهدف، وليس خداع نموذج ELECTRA كما هو الحال في إعداد GAN التقليدي) ثم يتم تدريب نموذج ELECTRA لبضع خطوات.

- تحتوي نقاط تفتيش ELECTRA المحفوظة باستخدام [تنفيذ Google Research](https://github.com/google-research/electra) على كل من المولد والمحقق. يتطلب برنامج التحويل من المستخدم تسمية النموذج الذي سيتم تصديره إلى الهندسة المعمارية الصحيحة. بمجرد تحويلها إلى تنسيق HuggingFace، يمكن تحميل هذه نقاط التفتيش في جميع نماذج ELECTRA المتاحة، ومع ذلك. وهذا يعني أنه يمكن تحميل المحقق في نموذج [`ElectraForMaskedLM`]، ويمكن تحميل المولد في نموذج [`ElectraForPreTraining`] (سيتم تهيئة رأس التصنيف بشكل عشوائي لأنه غير موجود في المولد).

## الموارد

- [دليل مهام تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهام تصنيف الرموز](../tasks/token_classification)
- [دليل مهام الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهام نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهام نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل مهام الاختيار المتعدد](../tasks/multiple_choice)

## ElectraConfig

[[autodoc]] ElectraConfig

## ElectraTokenizer

[[autodoc]] ElectraTokenizer

## ElectraTokenizerFast

[[autodoc]] ElectraTokenizerFast

## المخرجات الخاصة بـ Electra

[[autodoc]] models.electra.modeling_electra.ElectraForPreTrainingOutput

[[autodoc]] models.electra.modeling_tf_electra.TFElectraForPreTrainingOutput

<frameworkcontent>
<pt>

## ElectraModel

[[autodoc]] ElectraModel

- forward

## ElectraForPreTraining

[[autodoc]] ElectraForPreTraining

- forward

## ElectraForCausalLM

[[autodoc]] ElectraForCausalLM

- forward

## ElectraForMaskedLM

[[autodoc]] ElectraForMaskedLM

- forward

## ElectraForSequenceClassification

[[autodoc]] ElectraForSequenceClassification

- forward

## ElectraForMultipleChoice

[[autodoc]] ElectraForMultipleChoice

- forward

## ElectraForTokenClassification

[[autodoc]] ElectraForTokenClassification

- forward

## ElectraForQuestionAnswering

[[autodoc]] ElectraForQuestionAnswering

- forward

</pt>
<tf>

## TFElectraModel

[[autodoc]] TFElectraModel

- call

## TFElectraForPreTraining

[[autodoc]] TFElectraForPreTraining

- call

## TFElectraForMaskedLM

[[autodoc]] TFElectraForMaskedLM

- call

## TFElectraForSequenceClassification

[[autodoc]] TFElectraForSequenceClassification

- call

## TFElectraForMultipleChoice


[[autodoc]] TFElectraForMultipleChoice

- call

## TFElectraForTokenClassification

[[autodoc]] TFElectraForTokenClassification

- call

## TFElectraForQuestionAnswering

[[autodoc]] TFElectraForQuestionAnswering

- call

</tf>
<jax>

## FlaxElectraModel

[[autodoc]] FlaxElectraModel

- __call__

## FlaxElectraForPreTraining

[[autodoc]] FlaxElectraForPreTraining

- __call__

## FlaxElectraForCausalLM

[[autodoc]] FlaxElectraForCausalLM

- __call__

## FlaxElectraForMaskedLM

[[autodoc]] FlaxElectraForMaskedLM

- __call__

## FlaxElectraForSequenceClassification

[[autodoc]] FlaxElectraForSequenceClassification

- __call__

## FlaxElectraForMultipleChoice

[[autodoc]] FlaxElectraForMultipleChoice

- __call__

## FlaxElectraForTokenClassification

[[autodoc]] FlaxElectraForTokenClassification

- __call__

## FlaxElectraForQuestionAnswering

[[autodoc]] FlaxElectraForQuestionAnswering

- __call__

</jax>
</frameworkcontent>