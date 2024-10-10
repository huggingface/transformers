# XLNet

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=xlnet">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-xlnet-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/xlnet-base-cased">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## نظرة عامة

اقترح نموذج XLNet في [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) بواسطة Zhilin Yang، Zihang Dai، Yiming Yang، Jaime Carbonell، Ruslan Salakhutdinov،
Quoc V. Le. XLnet هو امتداد لنموذج Transformer-XL الذي تم تدريبه مسبقًا باستخدام طريقة ذاتية الترجيع لتعلم
سياقات ثنائية الاتجاه عن طريق تعظيم الاحتمالية المتوقعة عبر جميع ترتيبات ترتيب تسلسل الإدخال.

الملخص من الورقة هو ما يلي:

*مع قدرة نمذجة السياقات ثنائية الاتجاه، فإن التدريب المسبق للترميز التلقائي لإزالة التشويش مثل BERT يحقق
أداء أفضل من أساليب التدريب المسبق القائمة على نمذجة اللغة ذاتية الترجيع. ومع ذلك، فإن الاعتماد على
تشويه الإدخال باستخدام الأقنعة، يتجاهل BERT الاعتماد المتبادل بين المواضع المقنعة ويعاني من
الاختلاف بين التدريب المسبق والضبط الدقيق. في ضوء هذه الإيجابيات والسلبيات، نقترح XLNet، وهي طريقة تدريب مسبق ذاتية الترجيع عامة (1) تمكن من تعلم سياقات ثنائية الاتجاه عن طريق تعظيم الاحتمالية المتوقعة عبر جميع
ترتيبات ترتيب العوامل و (2) تتغلب على قيود BERT بفضل صيغتها ذاتية الترجيع. علاوة على ذلك، يدمج XLNet الأفكار من Transformer-XL، وهو نموذج ذاتي الترجيع من أحدث طراز، في
التدريب المسبق. تجريبياً، وبموجب إعدادات التجربة المماثلة، يتفوق XLNet على BERT في 20 مهمة، غالبًا بهامش كبير، بما في ذلك الإجابة على الأسئلة، والاستدلال اللغوي الطبيعي، وتحليل المشاعر، وتصنيف المستندات.

تمت المساهمة بهذا النموذج بواسطة [thomwolf](https://huggingface.co/thomwolf). يمكن العثور على الكود الأصلي [هنا](https://github.com/zihangdai/xlnet/).

## نصائح الاستخدام

- يمكن التحكم في نمط الانتباه المحدد في وقت التدريب ووقت الاختبار باستخدام إدخال `perm_mask`.
- بسبب صعوبة تدريب نموذج ذاتي الترجيع بالكامل على ترتيب عوامل مختلفة، يتم تدريب XLNet مسبقًا
  باستخدام مجموعة فرعية فقط من الرموز المخرجة كهدف يتم تحديدها باستخدام إدخال `target_mapping`.
- لاستخدام XLNet للترميز المتسلسل (أي ليس في إعداد ثنائي الاتجاه بالكامل)، استخدم إدخالات `perm_mask` و
  `target_mapping` للتحكم في فترة الانتباه والمخرجات (انظر الأمثلة في
  *examples/pytorch/text-generation/run_generation.py*)
- XLNet هو أحد النماذج القليلة التي ليس لها حد لطول التسلسل.
- XLNet ليس نموذجًا ذاتي الترجيع تقليديًا ولكنه يستخدم استراتيجية تدريب تستند إلى ذلك. فهو يعيد ترتيب الرموز في الجملة، ثم يسمح للنموذج باستخدام الرموز n الأخيرة للتنبؤ بالرمز n+1. نظرًا لأن كل هذا يتم باستخدام قناع، يتم بالفعل إدخال الجملة في النموذج بالترتيب الصحيح، ولكن بدلاً من إخفاء الرموز n الأولى لـ n+1، يستخدم XLNet قناعًا يخفي الرموز السابقة في بعض الترتيب المعين لـ 1، ..., طول التسلسل.
- يستخدم XLNet أيضًا نفس آلية التكرار مثل Transformer-XL لبناء تبعيات طويلة المدى.

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
