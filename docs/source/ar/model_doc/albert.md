# Albert

## نظرة عامة

اقترح نموذج Albert في "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" من قبل Zhenzhong Lan وآخرون. ويقدم تقنيتين لتخفيض المعلمات لخفض استهلاك الذاكرة وزيادة سرعة التدريب لـ BERT:

- تقسيم مصفوفة التضمين إلى مصفوفتين أصغر.
- استخدام طبقات متكررة مقسمة بين المجموعات.

الملخص من الورقة هو كما يلي:

*إن زيادة حجم النموذج عند التدريب المسبق لتمثيلات اللغة الطبيعية يؤدي غالبًا إلى تحسين الأداء في مهام التدفق السفلي. ومع ذلك، في مرحلة ما، تصبح الزيادات الإضافية في النموذج أكثر صعوبة بسبب قيود ذاكرة GPU/TPU، وأوقات التدريب الأطول، وتدهور النموذج غير المتوقع. ولمعالجة هذه المشكلات، نقدم تقنيتين لتخفيض المعلمات لخفض استهلاك الذاكرة وزيادة سرعة التدريب لـ BERT. تقدم الأدلة التجريبية الشاملة أن الطرق التي نقترحها تؤدي إلى نماذج تتناسب بشكل أفضل مقارنة بـ BERT الأصلي. كما نستخدم أيضًا خسارة ذاتية الإشراف تركز على نمذجة الترابط بين الجمل، ونظهر أنها تساعد باستمرار مهام التدفق السفلي مع إدخالات متعددة الجمل. ونتيجة لذلك، يحقق نموذجنا الأفضل نتائج جديدة في معايير GLUE وRACE وSQuAD بينما لديه معلمات أقل مقارنة بـ BERT-large.*

تمت المساهمة بهذا النموذج من قبل [lysandre](https://huggingface.co/lysandre). تمت المساهمة بهذه النسخة من النموذج بواسطة [kamalkraj](https://huggingface.co/kamalkraj). يمكن العثور على الكود الأصلي [هنا](https://github.com/google-research/ALBERT).

## نصائح الاستخدام

- Albert هو نموذج مع تضمينات الموضع المطلق لذلك يُنصح عادةً بإضافة مسافات إلى الإدخالات من اليمين بدلاً من اليسار.

- يستخدم Albert طبقات متكررة مما يؤدي إلى بصمة ذاكرة صغيرة، ومع ذلك، تظل التكلفة الحسابية مماثلة لبنية BERT-like بنفس عدد الطبقات المخفية لأنه يجب عليه المرور عبر نفس عدد الطبقات (المتكررة).

- حجم التضمين E مختلف عن حجم المخفي H مبرر لأن التضمينات مستقلة عن السياق (تمثل مصفوفة التضمين مصفوفة واحدة لكل رمز)، في حين أن الحالات المخفية تعتمد على السياق (تمثل حالة مخفية واحدة تسلسلًا من الرموز) لذا فمن المنطقي أن يكون H >> E. أيضًا، مصفوفة التضمين كبيرة حيث أنها V x E (V هو حجم المفردات). إذا كان E < H، فستكون المعلمات أقل.

- تنقسم الطبقات إلى مجموعات تشترك في المعلمات (لتوفير الذاكرة).

يتم استبدال التنبؤ بالجملة التالية بالتنبؤ بترتيب الجملة: في الإدخالات، لدينا جملتان A وB (متتاليتان) ونقوم إما بإدخال A متبوعة بـ B أو B متبوعة بـ A. يجب على النموذج التنبؤ بما إذا تم تبديلهما أم لا.

تمت المساهمة بهذا النموذج من قبل [lysandre](https://huggingface.co/lysandre). تمت المساهمة بنسخة النموذج هذه من قبل [kamalkraj](https://huggingface.co/kamalkraj). يمكن العثور على الكود الأصلي [هنا](https://github.com/google-research/ALBERT).

## الموارد

تتكون الموارد المقدمة في الأقسام التالية من قائمة بموارد Hugging Face الرسمية وموارد المجتمع (مشار إليها بـ 🌎) لمساعدتك في البدء باستخدام AlBERT. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فيرجى فتح طلب سحب وسنراجعه! يجب أن يثبت المورد بشكل مثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

- [`AlbertForSequenceClassification`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) هذا.

- [`TFAlbertForSequenceClassification`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification) هذا.

- [`FlaxAlbertForSequenceClassification`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb) هذا.

- تحقق من [دليل تصنيف النص](../tasks/sequence_classification) حول كيفية استخدام النموذج.

- [`AlbertForTokenClassification`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) هذا.

- [`TFAlbertForTokenClassification`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb) هذا.

- [`FlaxAlbertForTokenClassification`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification) هذا.

- الفصل [تصنيف الرمز](https://huggingface.co/course/chapter7/2؟fw=pt) من دورة 🤗 Hugging Face.

- تحقق من [دليل تصنيف الرموز](../tasks/token_classification) حول كيفية استخدام النموذج.

- [`AlbertForMaskedLM`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb) هذا.

- [`TFAlbertForMaskedLM`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb) هذا.

- [`FlaxAlbertForMaskedLM`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb) هذا.

- الفصل [نمذجة اللغة المقنعة](https://huggingface.co/course/chapter7/3؟fw=pt) من دورة 🤗 Hugging Face.

- تحقق من [دليل نمذجة اللغة المقنعة](../tasks/masked_language_modeling) حول كيفية استخدام النموذج.

- [`AlbertForQuestionAnswering`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb) هذا.

- [`TFAlbertForQuestionAnswering`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb) هذا.

- [`FlaxAlbertForQuestionAnswering`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering) هذا.

- الفصل [الأسئلة والأجوبة](https://huggingface.co/course/chapter7/7؟fw=pt) من دورة 🤗 Hugging Face.

- تحقق من [دليل الأسئلة والأجوبة](../tasks/question_answering) حول كيفية استخدام النموذج.

**الاختيار من متعدد**

- [`AlbertForMultipleChoice`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb) هذا.

- [`TFAlbertForMultipleChoice`] مدعوم بواسطة [نص البرنامج النصي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb) هذا.

- تحقق من [دليل الاختيار من متعدد](../tasks/multiple_choice) حول كيفية استخدام النموذج.

## AlbertConfig

[[autodoc]] AlbertConfig

## AlbertTokenizer

[[autodoc]] AlbertTokenizer

- build_inputs_with_special_tokens

- get_special_tokens_mask

- create_token_type_ids_from_sequences

- save_vocabulary

## AlbertTokenizerFast

[[autodoc]] AlbertTokenizerFast

## مخرجات Albert المحددة

[[autodoc]] models.albert.modeling_albert.AlbertForPreTrainingOutput

[[autodoc]] models.albert.modeling_tf_albert.TFAlbertForPreTrainingOutput

<frameworkcontent>
<pt>

## AlbertModel

[[autodoc]] AlbertModel

- forward

## AlbertForPreTraining

[[autodoc]] AlbertForPreTraining

- forward

## AlbertForMaskedLM

[[autodoc]] AlbertForMaskedLM

- forward

## AlbertForSequenceClassification

[[autodoc]] AlbertForSequenceClassification

- forward

## AlbertForMultipleChoice

[[autodoc]] AlbertForMultipleChoice

## AlbertForTokenClassification

[[autodoc]] AlbertForTokenClassification

- forward

## AlbertForQuestionAnswering


[[autodoc]] AlbertForQuestionAnswering

- forward

</pt>

<tf>

## TFAlbertModel

[[autodoc]] TFAlbertModel

- call
## TFAlbertForPreTraining

نموذج ألبرت للمرحلة السابقة للتدريب، مصمم لمهام ما قبل التدريب مثل التنبؤ بالجملة التالية.

## TFAlbertForMaskedLM

نموذج ألبرت للغة الماسكة، مصمم لمهام اللغة الماسكة مثل استكمال الجمل.

## TFAlbertForSequenceClassification

نموذج ألبرت لتصنيف التسلسل، مصمم لمهام تصنيف التسلسل مثل تصنيف النوايا في نصوص الدردشة.

## TFAlbertForMultipleChoice

نموذج ألبرت للاختيار المتعدد، مصمم لمهام الاختيار المتعدد مثل الإجابة على الأسئلة ذات الخيارات المتعددة.

## TFAlbertForTokenClassification

نموذج ألبرت لتصنيف الرموز، مصمم لمهام تصنيف الرموز مثل تسمية الكيانات في النص.

## TFAlbertForQuestionAnswering

نموذج ألبرت للإجابة على الأسئلة، مصمم لمهام الإجابة على الأسئلة مثل استرجاع الإجابات من نص معين.

## FlaxAlbertModel

نموذج ألبرت الأساسي المبنى باستخدام Flax، يمكن استخدامه كنقطة بداية لمهام NLP المختلفة.

## FlaxAlbertForPreTraining

نموذج ألبرت للمرحلة السابقة للتدريب، مصمم لمهام ما قبل التدريب مثل التنبؤ بالجملة التالية.

## FlaxAlbertForMaskedLM

نموذج ألبرت للغة الماسكة، مصمم لمهام اللغة الماسكة مثل استكمال الجمل.

## FlaxAlbertForSequenceClassification

نموذج ألبرت لتصنيف التسلسل، مصمم لمهام تصنيف التسلسل مثل تصنيف النوايا في نصوص الدردشة.

## FlaxAlbertForMultipleChoice

نموذج ألبرت للاختيار المتعدد، مصمم لمهام الاختيار المتعدد مثل الإجابة على الأسئلة ذات الخيارات المتعددة.

## FlaxAlbertForTokenClassification

نموذج ألبرت لتصنيف الرموز، مصمم لمهام تصنيف الرموز مثل تسمية الكيانات في النص.

## FlaxAlbertForQuestionAnswering

نموذج ألبرت للإجابة على الأسئلة، مصمم لمهام الإجابة على الأسئلة مثل استرجاع الإجابات من نص معين.