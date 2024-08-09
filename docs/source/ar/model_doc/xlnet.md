# XLNet

## نظرة عامة

اقترح نموذج XLNet في [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) بواسطة Zhilin Yang، وZihang Dai، وYiming Yang، وJaime Carbonell، وRuslan Salakhutdinov، وQuoc V. Le. XLnet هو امتداد لنموذج Transformer-XL الذي تم تدريبه مسبقًا باستخدام طريقة السياق ثنائي الاتجاه المستندة إلى النمذجة اللغوية السياقية.

مقتطف من الورقة البحثية:

> *مع قدرة النمذجة اللغوية السياقية ثنائية الاتجاه، فإن التدريب المسبق للترميز التلقائي لإزالة التشويش مثل BERT يحقق أداءً أفضل من أساليب التدريب المسبق القائمة على نمذجة اللغة السياقية السياقية. ومع ذلك، فإن BERT، الذي يعتمد على تشويش المدخلات باستخدام الأقنعة، يتجاهل الاعتماد بين المواضع المقنعة ويعاني من عدم الاتساق بين التدريب الدقيق والتدريب المسبق. وفي ضوء هذه المزايا والعيوب، نقترح XLNet، وهي طريقة تدريب مسبق سياقية عامة تتيح (1) تعلم السياقات ثنائية الاتجاه عن طريق تعظيم الاحتمالية المتوقعة على جميع ترتيبات ترتيب العوامل و(2) تتغلب على قيود BERT بفضل صياغتها السياقية. علاوة على ذلك، يدمج XLNet أفكار Transformer-XL، وهو نموذج سياقي سياقي رائد، في التدريب المسبق. ومن الناحية التجريبية، يتفوق XLNet على BERT في 20 مهمة، غالبًا بهامش كبير، بما في ذلك الإجابة على الأسئلة والاستدلال اللغوي الطبيعي وتحليل المشاعر وترتيب المستندات.*

تمت المساهمة بهذا النموذج من قبل [thomwolf](https://huggingface.co/thomwolf). يمكن العثور على الكود الأصلي [هنا](https://github.com/zihangdai/xlnet/).

## نصائح الاستخدام

- يمكن التحكم في نمط الانتباه المحدد أثناء التدريب ووقت الاختبار باستخدام إدخال `perm_mask`.
- بسبب صعوبة تدريب نموذج سياقي ذاتي كامل على ترتيب عوامل مختلف، تم تدريب XLNet مسبقًا باستخدام مجموعة فرعية فقط من الرموز المخرجة كهدف يتم تحديدها باستخدام إدخال `target_mapping`.
- لاستخدام XLNet للترميز التسلسلي (أي ليس في إعداد ثنائي الاتجاه الكامل)، استخدم إدخالات `perm_mask` و`target_mapping` للتحكم في نطاق الانتباه والمخرجات (انظر الأمثلة في *examples/pytorch/text-generation/run_generation.py*).
- XLNet هو أحد النماذج القليلة التي لا يوجد بها حد لطول التسلسل.
- XLNet ليس نموذجًا سياقيًا ذاتيًا تقليديًا ولكنه يستخدم استراتيجية تدريب تستند إلى ذلك. فهو يقوم بترتيب الرموز في الجملة، ثم يسمح للنموذج باستخدام الرموز n الأخيرة للتنبؤ بالرمز n+1. نظرًا لأن كل هذا يتم باستخدام قناع، يتم إدخال الجملة بالفعل في النموذج بالترتيب الصحيح، ولكن بدلاً من قناع الرموز n الأولى لـ n+1، يستخدم XLNet قناعًا يقوم بإخفاء الرموز السابقة في بعض الترتيبات المعطاة لـ 1، ...، طول التسلسل.
- يستخدم XLNet أيضًا نفس آلية التكرار مثل Transformer-XL لبناء الاعتمادية طويلة المدى.

## الموارد

- [دليل مهمة تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهمة تصنيف الرموز](../tasks/token_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## XLNetConfig

[[autodoc]] XLNetConfig

## XLNetTokenizer

[[autodoc]] XLNetTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## XLNetTokenizerFast

[[autodoc]] XLNetTokenizerFast

## مخرجات XLNet المحددة

[[autodoc]] models.xlnet.modeling_xlnet.XLNetModelOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetLMHeadModelOutput

[[autodoc]] models.xlpartumizing_xlnet.XLNetForSequenceClassificationOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForMultipleChoiceOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForTokenClassificationOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringSimpleOutput

[[autodoc]] models.xlnet.modeling_xlnet.XLNetForQuestionAnsweringOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetModelOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetLMHeadModelOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetForSequenceClassificationOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetForMultipleChoiceOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetForTokenClassificationOutput

[[autodoc]] models.xlnet.modeling_tf_xlnet.TFXLNetForQuestionAnsweringSimpleOutput

<frameworkcontent>
<pt>

## XLNetModel

[[autodoc]] XLNetModel

- forward

## XLNetLMHeadModel

[[autodoc]] XLNetLMHeadModel

- forward

## XLNetForSequenceClassification

[[autodoc]] XLNetForSequenceClassification

- forward

## XLNetForMultipleChoice

[[autodoc]] XLNetForMultipleChoice

- forward

## XLNetForTokenClassification

[[autodoc]] XLNetForTokenClassification

- forward

## XLNetForQuestionAnsweringSimple

[[autodoc]] XLNetForQuestionAnsweringSimple

- forward

## XLNetForQuestionAnswering

[[autodoc]] XLNetForQuestionAnswering

- forward

</pt>
<tf>

## TFXLNetModel

[[autodoc]] TFXLNetModel

- call

## TFXLNetLMHeadModel

[[autodoc]] TFXLNetLMHeadModel

- call

## TFXLNetForSequenceClassification

[[autodoc]] TFXLNetForSequenceClassification

- call

## TFLNetForMultipleChoice

[[autodoc]] TFXLNetForMultipleChoice

- call

## TFXLNetForTokenClassification

[[autodoc]] TFXLNetForTokenClassification

- call

## TFXLNetForQuestionAnsweringSimple

[[autodoc]] TFXLNetForQuestionAnsweringSimple

- call

</tf>
</frameworkcontent>