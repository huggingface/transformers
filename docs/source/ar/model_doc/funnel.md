# Funnel Transformer

## نظرة عامة

اقتُرح نموذج Funnel Transformer في الورقة البحثية [Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing](https://arxiv.org/abs/2006.03236). وهو نموذج تحويل ثنائي الاتجاه، مثل BERT، ولكن مع عملية تجميع (Pooling) بعد كل كتلة من الطبقات، تشبه إلى حد ما الشبكات العصبية التلافيفية التقليدية (CNN) في رؤية الكمبيوتر.

وملخص الورقة البحثية هو كما يلي:

*مع نجاح اللغة في مرحلة ما قبل التدريب، من المرغوب فيه للغاية تطوير هندسات أكثر كفاءة وقابلة للتطوير بشكل جيد يمكنها استغلال البيانات غير المُعَلَّمَة الوفيرة بتكلفة أقل. ولتحسين الكفاءة، ندرس التكرار المُهْمَل إلى حد كبير في الحفاظ على عرض كامل على مستوى الرمز، خاصة بالنسبة للمهام التي تتطلب فقط عرضًا أحادي المتجه للتسلسل. وبناءً على هذا الحدس، نقترح Funnel-Transformer الذي يضغط تدريجيًا تسلسل حالات المخفية إلى تسلسل أقصر، وبالتالي يقلل تكلفة الحساب. والأهم من ذلك، من خلال إعادة استثمار FLOPs المحفوظة من تقليل الطول في بناء نموذج أعمق أو أوسع، فإننا نحسن بشكل أكبر سعة النموذج. بالإضافة إلى ذلك، ولأداء تنبؤات على مستوى الرمز كما هو مطلوب من قبل أهداف ما قبل التدريب الشائعة، يمكن لـ Funnel-Transformer استعادة تمثيل عميق لكل رمز من التسلسل المخفي المختزل عبر فك الترميز. ومن الناحية التجريبية، وباستخدام عدد مماثل أو أقل من FLOPs، يفوق Funnel-Transformer المحول القياسي في مجموعة واسعة من مهام التنبؤ على مستوى التسلسل، بما في ذلك تصنيف النصوص، وفهم اللغة، والفهم القرائي.*

تمت المساهمة بهذا النموذج من قبل [sgugger](https://huggingface.co/sgugger). يمكن العثور على الكود الأصلي [هنا](https://github.com/laiguokun/Funnel-Transformer).

## نصائح الاستخدام

- بما أن Funnel Transformer يستخدم التجميع (Pooling)، يتغير طول تسلسل الحالات المخفية بعد كل كتلة من الطبقات. بهذه الطريقة، يتم تقسيم طولها إلى النصف، مما يسرع حساب الحالات المخفية التالية. لذلك، فإن طول التسلسل النهائي للنموذج الأساسي هو ربع الطول الأصلي. يمكن استخدام هذا النموذج مباشرة للمهام التي تتطلب فقط ملخص الجملة (مثل تصنيف التسلسل أو الاختيار من متعدد). بالنسبة للمهام الأخرى، يتم استخدام النموذج الكامل؛ يحتوي هذا النموذج الكامل على فك تشفير يقوم بزيادة حجم الحالات المخفية النهائية إلى نفس طول التسلسل مثل الإدخال.

- بالنسبة لمهام مثل التصنيف، لا يمثل ذلك مشكلة، ولكن بالنسبة للمهام مثل نمذجة اللغة المقنعة أو تصنيف الرموز، نحتاج إلى حالة مخفية بنفس طول تسلسل الإدخال الأصلي. في تلك الحالات، يتم زيادة حجم الحالات المخفية النهائية إلى طول تسلسل الإدخال وتمر عبر طبقتين إضافيتين. هذا هو السبب في وجود إصدارين لكل نقطة تفتيش. يحتوي الإصدار الذي يحمل اللاحقة "-base" على الكتل الثلاث فقط، في حين أن الإصدار بدون اللاحقة يحتوي على الكتل الثلاث ورأس زيادة الحجم مع طبقاته الإضافية.

- تتوفر جميع نقاط تفتيش Funnel Transformer بإصدارين، كامل وأساسي. يجب استخدام الإصدارات الأولى لـ [`FunnelModel`]، و [`FunnelForPreTraining`]، و [`FunnelForMaskedLM`]، و [`FunnelForTokenClassification`]، و [`FunnelForQuestionAnswering`]. يجب استخدام الإصدارات الثانية لـ [`FunnelBaseModel`]، و [`FunnelForSequenceClassification`]، و [`FunnelForMultipleChoice`].

## الموارد

- [دليل مهام تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهام تصنيف الرموز](../tasks/token_classification)
- [دليل مهام الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهام نمذجة اللغة المقنعة](../tasks/masked_language_modeling)
- [دليل مهام الاختيار من متعدد](../tasks/multiple_choice)

## FunnelConfig

[[autodoc]] FunnelConfig

## FunnelTokenizer

[[autodoc]] FunnelTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## FunnelTokenizerFast

[[autodoc]] FunnelTokenizerFast

## المخرجات الخاصة بـ Funnel

[[autodoc]] models.funnel.modeling_funnel.FunnelForPreTrainingOutput

[[autodoc]] models.funnel.modeling_tf_funnel.TFFunnelForPreTrainingOutput

<frameworkcontent>
<pt>

## FunnelBaseModel

[[autodoc]] FunnelBaseModel

- forward

## FunnelModel

[[autodoc]] FunnelModel

- forward

## FunnelModelForPreTraining

[[autodoc]] FunnelForPreTraining

- forward

## FunnelForMaskedLM

[[autodoc]] FunnelForMaskedLM

- forward

## FunnelForSequenceClassification

[[autodoc]] FunnelForSequenceClassification

- forward

## FunnelForMultipleChoice

[[autodoc]] FunnelForMultipleChoice

- forward

## FunnelForTokenClassification

[[autodoc]] FunnelForTokenClassification

- forward

## FunnelForQuestionAnswering

[[autodoc]] FunnelForQuestionAnswering

- forward

</pt>
<tf>

## TFFunnelBaseModel

[[autodoc]] TFFunnelBaseModel

- call

## TFFunnelModel

[[autodoc]] TFFunnelModel

- call

## TFFunnelModelForPreTraining

[[autodoc]] TFFunnelForPreTraining

- call

## TFFunnelForMaskedLM

[[autodoc]] TFFunnelForMaskedLM

- call

## TFFunnelForSequenceClassification

[[autodoc]] TFFunnelForSequenceClassification

- call

## TFFunnelForMultipleChoice

[[autodoc]] TFFunnelForMultipleChoice

- call

## TFFunnelForTokenClassification

[[autodoc]] TFFunnelForTokenClassification

- call

## TFFunnelForQuestionAnswering

[[autodoc]] TFFunnelForQuestionAnswering

- call

</tf>
</frameworkcontent>