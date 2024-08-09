# XLM-RoBERTa

## نظرة عامة

اقترح نموذج XLM-RoBERTa في ورقة "تعلم التمثيل متعدد اللغات بدون إشراف على نطاق واسع" من قبل أليكسيس كونو، وكارتي كاي خانديلوال، ونيمان جويال، وفيشراف تشاودري، وجيوم ونسيك، وفرانسيسكو غوزمان، وإدوارد جريف، وميل أوت، ولوك زيتلموير، وفيسيلين ستويانوف. وهو مبني على نموذج RoBERTa الذي أصدرته شركة فيسبوك في عام 2019. وهو نموذج لغوي متعدد اللغات كبير، تم تدريبه على 2.5 تيرابايت من بيانات CommonCrawl التي تمت تصفيتها.

ملخص الورقة البحثية هو كما يلي:

*"توضح هذه الورقة أن التدريب المسبق للنماذج اللغوية متعددة اللغات على نطاق واسع يؤدي إلى مكاسب أداء كبيرة لمجموعة واسعة من مهام النقل متعدد اللغات. نقوم بتدريب نموذج محول قائم على لغة مقنعة على مائة لغة، باستخدام أكثر من اثنين تيرابايت من بيانات CommonCrawl التي تمت تصفيتها. يتفوق نموذجنا، الذي يُطلق عليه اسم XLM-R، بشكل كبير على BERT متعدد اللغات (mBERT) في مجموعة متنوعة من المعايير المرجعية متعددة اللغات، بما في ذلك +13.8% متوسط الدقة على XNLI، و +12.3% متوسط F1 score على MLQA، و +2.1% متوسط F1 score على NER. يعمل XLM-R بشكل جيد بشكل خاص على اللغات منخفضة الموارد، حيث يحسن دقة XNLI بنسبة 11.8% للغة السواحيلية و 9.2% للغة الأردية على نموذج XLM السابق. كما نقدم تقييمًا تجريبيًا مفصلاً للعوامل الرئيسية اللازمة لتحقيق هذه المكاسب، بما في ذلك المفاضلة بين (1) النقل الإيجابي وتخفيف السعة و (2) أداء اللغات عالية ومنخفضة الموارد على نطاق واسع. وأخيرًا، نُظهر، لأول مرة، إمكانية النمذجة متعددة اللغات دون التضحية بالأداء لكل لغة؛ XLM-R تنافسية للغاية مع النماذج الأحادية اللغة القوية على معايير GLUE و XNLI. سنقوم بإتاحة كود XLM-R والبيانات والنماذج للجمهور".*

تمت المساهمة بهذا النموذج من قبل [stefan-it](https://huggingface.co/stefan-it). يمكن العثور على الكود الأصلي [هنا](https://github.com/pytorch/fairseq/tree/master/examples/xlmr).

## نصائح الاستخدام

- XLM-RoBERTa هو نموذج متعدد اللغات تم تدريبه على 100 لغة مختلفة. على عكس بعض النماذج متعددة اللغات XLM، فإنه لا يتطلب `lang` tensors لفهم اللغة المستخدمة، ويجب أن يكون قادرًا على تحديد اللغة الصحيحة من معرفات الإدخال.

- يستخدم حيل RoBERTa على نهج XLM، ولكنه لا يستخدم هدف نمذجة الترجمة اللغوية. فهو يستخدم فقط نمذجة اللغة المقنعة على الجمل القادمة من لغة واحدة.

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (مشار إليها بـ 🌎) لمساعدتك في البدء باستخدام XLM-RoBERTa. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فالرجاء فتح طلب سحب Pull Request وسنراجعه! يجب أن يُظهر المورد المثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

- منشور مدونة حول كيفية [ضبط دقة XLM RoBERTa للتصنيف متعدد الفئات مع Habana Gaudi على AWS](https://www.philschmid.de/habana-distributed-training)

- مدعوم [`XLMRobertaForSequenceClassification`] بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb).

- مدعوم [`TFXLMRobertaForSequenceClassification`] بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb).

- مدعوم [`FlaxXLMRobertaForSequenceClassification`] بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb).

- [تصنيف النص](https://huggingface.co/docs/transformers/tasks/sequence_classification) فصل من دليل المهام 🤗 Hugging Face.

- [دليل مهمة تصنيف النص](../tasks/sequence_classification)

- مدعوم [`XLMRobertaForTokenClassification`] بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb).

- مدعوم [`TFXLMRobertaForTokenClassification`] بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb).

- مدعوم [`FlaxXLMRobertaForTokenClassification`] بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification).

- [تصنيف الرموز](https://huggingface.co/course/chapter7/2?fw=pt) فصل من الدورة 🤗 Hugging Face.

- [دليل مهمة تصنيف الرموز](../tasks/token_classification)

- مدعوم [`XLMRobertaForCausalLM`] بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).

- [نمذجة اللغة السببية](https://huggingface.co/docs/transformers/tasks/language_modeling) فصل من دليل المهام 🤗 Hugging Face.

- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)

- مدعوم [`XLMRobertaForMaskedLM`] بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).

- مدعوم [`TFXLMRobertaForMaskedLM`] بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).

- مدعوم [`FlaxXLMRobertaForMaskedLM`] بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb).

- [نمذجة اللغة المقنعة](https://huggingface.co/course/chapter7/3?fw=pt) فصل من الدورة 🤗 Hugging Face.

- [نمذجة اللغة المقنعة](../tasks/masked_language_modeling)

- مدعوم [`XLMRobertaForQuestionAnswering`] بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb).

- مدعوم [`TFXLMRobertaForQuestionAnswering`] بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb).

- مدعوم [`FlaxXLMRobertaForQuestionAnswering`] بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering).

- [الإجابة على الأسئلة](https://huggingface.co/course/chapter7/7?fw=pt) فصل من الدورة 🤗 Hugging Face.

- [دليل مهمة الإجابة على الأسئلة](../tasks/question_answering)

**الاختيار من متعدد**

- مدعوم [`XLMRobertaForMultipleChoice`] بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb).

- مدعوم [`TFXLMRobertaForMultipleChoice`] بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb).

- [دليل المهمة متعدد الخيارات](../tasks/multiple_choice)

🚀 النشر

- منشور مدونة حول كيفية [نشر XLM-RoBERTa عديم الخادم على AWS Lambda](https://www.philschmid.de/multilingual-serverless-xlm-roberta-with-huggingface).

<Tip>

هذا التنفيذ هو نفسه RoBERTa. راجع [توثيق RoBERTa](roberta) للحصول على أمثلة الاستخدام بالإضافة إلى المعلومات المتعلقة بالإدخالات والمخرجات.

</Tip>

## XLMRobertaConfig

[[autodoc]] XLMRobertaConfig

## XLMRobertaTokenizer

[[autodoc]] XLMRobertaTokenizer

- build_inputs_with_special_tokens

- get_special_tokens_mask

- create_token_type_ids_from_sequences

- save_vocabulary

## XLMRobertaTokenizerFast

[[autodoc]] XLMRobertaTokenizerFast

<frameworkcontent>

<pt>

## XLMRobertaModel

[[autodoc]] XLMRobertaModel

- forward

## XLMRobertaForCausalLM

[[autodoc]] XLMRobertaForCausalLM

- forward

## XLMRobertaForMaskedLM

[[autodoc]] XLMRobertaForMaskedLM

- forward

## XLMRobertaForSequenceClassification

[[autodoc]] XLMRobertaForSequenceClassification

- forward

## XLMRobertaForMultipleChoice

[[autodoc]] XLMRobertaForMultipleChoice

- forward

## XLMRobertaForTokenClassification

[[autodoc]] XLMRobertaForTokenClassification

- forward

## XLMRobertaForQuestionAnswering

[[autodoc]] XLMRobertaForQuestionAnswering

- forward

</pt>

<tf>

## TFXLMRobertaModel

[[autodoc]] TFXLMRobertaModel

- call

## TFXLMRobertaForCausalLM

[[autodoc]] TFXLMRobertaForCausalLM

- call

## TFXLMRobertaForMaskedLM

[[autodoc]] TFXLMRobertaForMaskedLM

- call

## TFXLMRobertaForSequenceClassification

[[autodoc]] TFXLMRobertaForSequenceClassification

- call

## TFXLMRobertaForMultipleChoice

[[autodoc]] TFXLMRobertaForMultipleChoice

- call

## TFXLMRobertaForTokenClassification

[[autodoc]] TFXLMRobertaForTokenClassification

- call

## TFXLMRobertaForQuestionAnswering

[[autodoc]] TFXLMRobertaForQuestionAnswering

- call

</tf>

<jax>

## FlaxXLMRobertaModel

[[autodoc]] FlaxXLMRobertaModel


- __call__

## FlaxXLMRobertaForCausalLM

[[autodoc]] FlaxXLMRobertaForCausalLM

- __call__

## FlaxXLMRobertaForMaskedLM

[[autodoc]] FlaxXLMRobertaForMaskedLM

- __call__

## FlaxXLMRobertaForSequenceClassification

[[autodoc]] FlaxXLMRobertaForSequenceClassification

- __call__

## FlaxXLMRobertaForMultipleChoice

[[autodoc]] FlaxXLMRobertaForMultipleChoice

- __call__

## FlaxXLMRobertaForTokenClassification

[[autodoc]] FlaxXLMRobertaForTokenClassification

- __call__

## FlaxXLMRobertaForQuestionAnswering

[[autodoc]] FlaxXLMRobertaForQuestionAnswering

- __call__

</jax>

</frameworkcontent>