# RoBERTa-PreLayerNorm

## نظرة عامة

تم اقتراح نموذج RoBERTa-PreLayerNorm في ورقة "fairseq: مجموعة أدوات سريعة وقابلة للتوسيع لوضع نماذج للتسلسل" بواسطة ميلي أوت، وسيرجي إيدونوف، وأليكسي بايفسكي، وأنجيلا فان، وسام جروس، وناثان نج، وديفيد رانجير، ومايكل أولي.

وهو مطابق لاستخدام علم `--encoder-normalize-before` في [fairseq](https://fairseq.readthedocs.io/).

هذا الملخص مأخوذ من الورقة:

> fairseq هي مجموعة أدوات مفتوحة المصدر لوضع نماذج للتسلسل تتيح للباحثين والمطورين تدريب نماذج مخصصة للترجمة، والتلخيص، ونمذجة اللغة، وغيرها من مهام توليد النصوص. وتستند مجموعة الأدوات إلى PyTorch وتدعم التدريب الموزع عبر وحدات معالجة الرسومات (GPUs) والآلات. كما ندعم التدريب والاستنتاج عاليي الدقة المختلطة على وحدات معالجة الرسومات الحديثة.

تمت المساهمة بهذا النموذج من قبل [andreasmaden](https://huggingface.co/andreasmadsen). يمكن العثور على الكود الأصلي [هنا](https://github.com/princeton-nlp/DinkyTrain).

## نصائح الاستخدام

- التنفيذ مطابق لـ [Roberta](roberta) باستثناء أنه بدلاً من استخدام _Add and Norm_، فإنه يستخدم _Norm and Add_. تشير _Add_ و _Norm_ إلى الإضافة وتطبيع الطبقة كما هو موصوف في ورقة "Attention Is All You Need".

- هذا مطابق لاستخدام علم `--encoder-normalize-before` في [fairseq](https://fairseq.readthedocs.io/).

## الموارد

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهمة تصنيف الرموز](../tasks/token_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهمة نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## RobertaPreLayerNormConfig

[[autodoc]] RobertaPreLayerNormConfig

<frameworkcontent>

<pt>

## RobertaPreLayerNormModel

[[autodoc]] RobertaPreLayerNormModel

- forward

## RobertaPreLayerNormForCausalLM

[[autodoc]] RobertaPreLayerNormForCausalLM

- forward

## RobertaPreLayerNormForMaskedLM

[[autodoc]] RobertaPreLayerNormForMaskedLM

- forward

## RobertaPreLayerNormForSequenceClassification

[[autodoc]] RobertaPreLayerNormForSequenceClassification

- forward

## RobertaPreLayerNormForMultipleChoice

[[autodoc]] RobertaPreLayerNormForMultipleChoice

- forward

## RobertaPreLayerNormForTokenClassification

[[autodoc]] RobertaPreLayerNormForTokenClassification

- forward

## RobertaPreLayerNormForQuestionAnswering

[[autodoc]] RobertaPreLayerNormForQuestionAnswering

- forward

</pt>

<tf>

## TFRobertaPreLayerNormModel

[[autodoc]] TFRobertaPreLayerNormModel

- call

## TFRobertaPreLayerNormForCausalLM

[[autodoc]] TFRobertaPreLayerNormForCausalLM

- call

## TFRobertaPreLayerNormForMaskedLM

[[autodoc]] TFRobertaPreLayerNormForMaskedLM

- call

## TFRobertaPreLayerNormForSequenceClassification

[[autodoc]] TFRobertaPreLayerNormForSequenceClassification

- call

## TFRobertaPreLayerNormForMultipleChoice

[[autodoc]] TFRobertaPreLayerNormForMultipleChoice

- call

## TFRobertaPreLayerNormForTokenClassification

[[autodoc]] TFRobertaPreLayerNormForTokenClassification

- call

## TFRobertaPreLayerNormForQuestionAnswering

[[autodoc]] TFRobertaPreLayerNormForQuestionAnswering

- call

</tf>

<jax>

## FlaxRobertaPreLayerNormModel

[[autodoc]] FlaxRobertaPreLayerNormModel

- __call__

## FlaxRobertaPreLayerNormForCausalLM

[[autodoc]] FlaxRobertaPreLayerNormForCausalLM

- __call__

## FlaxRobertaPreLayerNormForMaskedLM

[[autodoc]] FlaxRobertaPreLayerNormForMaskedLM

- __call__

## FlaxRobertaPreLayerNormForSequenceClassification

[[autodoc]] FlaxRobertaPreLayerNormForSequenceClassification

- __call__

## FlaxRobertaPreLayerNormForMultipleChoice

[[autodoc]] FlaxRobertaPreLayerNormForMultipleChoice

- __call__

## FlaxRobertaPreLayerNormForTokenClassification

[[autodoc]] FlaxRobertaPreLayerNormForTokenClassification

- __call__

## FlaxRobertaPreLayerNormForQuestionAnswering

[[autodoc]] FlaxRobertaPreLayerNormForQuestionAnswering

- __call__

</jax>

</frameworkcontent>