# BigBird

## نظرة عامة
تم اقتراح نموذج BigBird في ورقة "Big Bird: Transformers for Longer Sequences" بواسطة Zaheer وآخرون. BigBird هو محول يعتمد على الانتباه النادر الذي يوسع النماذج القائمة على المحول، مثل BERT، إلى تسلسلات أطول بكثير. بالإضافة إلى الانتباه النادر، يطبق BigBird أيضًا الانتباه العالمي والانتباه العشوائي على التسلسل المدخل. وقد ثبت نظريًا أن تطبيق الانتباه النادر والعالمي والعشوائي يقارب الانتباه الكامل، مع كونه أكثر كفاءة من الناحية الحسابية للتسلسلات الأطول. وكنتيجة لقدرة BigBird على التعامل مع السياق الأطول، فقد أظهر أداءً محسنًا في العديد من مهام معالجة اللغات الطبيعية للوثائق الطويلة، مثل الإجابة على الأسئلة والتلخيص، مقارنةً بـ BERT أو RoBERTa.

ملخص الورقة هو كما يلي:

* "تعد النماذج القائمة على المحولات، مثل BERT، أحد أكثر نماذج التعلم العميق نجاحًا في معالجة اللغات الطبيعية. ولكن أحد قيودها الأساسية هو الاعتماد التربيعي (بشكل رئيسي من حيث الذاكرة) على طول التسلسل بسبب آلية الانتباه الكامل الخاصة بها. ولعلاج ذلك، نقترح BigBird، وهي آلية انتباه نادرة تقلل هذا الاعتماد التربيعي إلى الاعتماد الخطي. ونظهر أن BigBird هو محاكٍ تقريبي عالمي لوظائف التسلسل وهو مكتمل من الناحية النظرية، وبالتالي يحافظ على هذه الخصائص للنموذج الكامل ذو الاعتماد التربيعي. وفي أثناء ذلك، يكشف تحليلنا النظري بعض فوائد وجود رموز عالمية O(1) (مثل CLS)، والتي تحضر التسلسل بالكامل كجزء من آلية الانتباه النادرة. ويمكن لآلية الانتباه النادرة التعامل مع تسلسلات يصل طولها إلى 8 مرات مما كان ممكنًا سابقًا باستخدام أجهزة مماثلة. وكنتيجة لقدرة BigBird على التعامل مع السياق الأطول، فإنه يحسن الأداء بشكل كبير في العديد من مهام معالجة اللغات الطبيعية مثل الإجابة على الأسئلة والتلخيص. كما نقترح تطبيقات جديدة على بيانات الجينوم".

تمت المساهمة بهذا النموذج بواسطة [vasudevgupta]. يمكن العثور على الكود الأصلي [هنا](https://github.com/google-research/bigbird).

## نصائح الاستخدام

- للحصول على شرح تفصيلي لكيفية عمل انتباه BigBird، راجع [منشور المدونة هذا](https://huggingface.co/blog/big-bird).
- يأتي BigBird بتنفيذين: **original_full** و **block_sparse**. بالنسبة لطول التسلسل < 1024، يُنصح باستخدام **original_full** حيث لا توجد فائدة من استخدام انتباه **block_sparse**.
- يستخدم الكود حاليًا حجم نافذة مكون من 3 كتل و2 كتل عالمية.
- يجب أن يكون طول التسلسل قابلاً للقسمة على حجم الكتلة.
- يدعم التنفيذ الحالي فقط **ITC**.
- لا يدعم التنفيذ الحالي **num_random_blocks = 0**
- BigBird هو نموذج مع تضمين موضع مطلق، لذلك يُنصح عادةً بإضافة حشو إلى المدخلات من اليمين بدلاً من اليسار.

## الموارد

- [دليل مهمة تصنيف النص](../tasks/sequence_classification)
- [دليل مهمة تصنيف الرموز](../tasks/token_classification)
- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)
- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)
- [دليل مهمة نمذجة اللغة المعقدة](../tasks/masked_language_modeling)
- [دليل المهمة متعددة الخيارات](../tasks/multiple_choice)

## BigBirdConfig

[[autodoc]] BigBirdConfig

## BigBirdTokenizer

[[autodoc]] BigBirdTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

## BigBirdTokenizerFast

[[autodoc]] BigBirdTokenizerFast

## المخرجات الخاصة بـ BigBird

[[autodoc]] models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput

<frameworkcontent>

<pt>

## BigBirdModel

[[autodoc]] BigBirdModel

- forward

## BigBirdForPreTraining

[[autodoc]] BigBirdForPreTraining

- forward

## BigBirdForCausalLM

[[autodoc]] BigBirdForCausalLM

- forward

## BigBirdForMaskedLM

[[autodoc]] BigBirdForMaskedLM

- forward

## BigBirdForSequenceClassification

[[autodoc]] BigBirdForSequenceClassification

- forward

## BigBirdForMultipleChoice

[[autodoc]] BigBirdForMultipleChoice

- forward

## BigBirdForTokenClassification

[[autodoc]] BigBirdForTokenClassification

- forward

## BigBirdForQuestionAnswering

[[autodoc]] BigBirdForQuestionAnswering

- forward

</pt>

<jax>

## FlaxBigBirdModel

[[autodoc]] FlaxBigBirdModel

- __call__

## FlaxBigBirdForPreTraining

[[autodoc]] FlaxBigBirdForPreTraining

- __call__

## FlaxBigBirdForCausalLM

[[autodoc]] FlaxBigBirdForCausalLM

- __call__

## FlaxBigBirdForMaskedLM

[[autodoc]] FlaxBigBirdForMaskedLM

- __call__

## FlaxBigBirdForSequenceClassification

[[autodoc]] FlaxBigBirdForSequenceClassification

- __call__

## FlaxBigBirdForMultipleChoice

[[autodoc]] FlaxBigBirdForMultipleChoice

- __call__

## FlaxBigBirdForTokenClassification

[[autodoc]] FlaxBigBirdForTokenClassification

- __call__

## FlaxBigBirdForQuestionAnswering

[[autodoc]] FlaxBigBirdForQuestionAnswering

- __call__

</jax>

</frameworkcontent>