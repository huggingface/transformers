# RoBERTa

## نظرة عامة

اقترح نموذج RoBERTa في [RoBERTa: نهج التحسين القوي لتدريب BERT المسبق](https://arxiv.org/abs/1907.11692) بواسطة Yinhan Liu، [Myle Ott](https://huggingface.co/myleott)، Naman Goyal، Jingfei Du، Mandar Joshi، Danqi Chen، Omer Levy، Mike Lewis، Luke Zettlemoyer، Veselin Stoyanov. وهو مبني على نموذج BERT الذي أصدرته جوجل في عام 2018.

يستند النموذج إلى BERT ويعدل المعلمات الأساسية، حيث يزيل هدف التدريب المسبق للجملة التالية ويتدرب على دفعات تعليمية أكبر ومعدلات تعلم أكبر.

المقتطف من الورقة هو ما يلي:

> أدى التدريب المسبق لنموذج اللغة إلى مكاسب أداء كبيرة ولكن المقارنة الدقيقة بين النهج المختلفة أمر صعب. التدريب مكلف من الناحية الحسابية، وغالباً ما يتم على مجموعات بيانات خاصة بأحجام مختلفة، وكما سنظهر، فإن خيارات المعلمة الأساسية لها تأثير كبير على النتائج النهائية. نقدم دراسة استنساخ لتدريب BERT المسبق (Devlin et al.، 2019) التي تقيس بعناية تأثير العديد من المعلمات الأساسية وحجم بيانات التدريب. نجد أن BERT كان مدربًا بشكل كبير، ويمكنه مطابقة أو تجاوز أداء كل نموذج تم نشره بعده. يحقق أفضل نموذج لدينا نتائج متقدمة على GLUE وRACE وSQuAD. تسلط هذه النتائج الضوء على أهمية خيارات التصميم التي تم التغاضي عنها سابقًا، وتثير أسئلة حول مصدر التحسينات المبلغ عنها مؤخرًا. نقوم بإصدار نماذجنا وشفرة المصدر الخاصة بنا.

تمت المساهمة بهذا النموذج من قبل [julien-c](https://huggingface.co/julien-c). يمكن العثور على الكود الأصلي [هنا](https://github.com/pytorch/fairseq/tree/master/examples/roberta).

## نصائح الاستخدام

- هذا التنفيذ هو نفسه [`BertModel`] مع تعديل طفيف على embeddings بالإضافة إلى إعداد لنماذج Roberta المُدربة مسبقًا.

- يستخدم RoBERTa نفس الهندسة المعمارية مثل BERT، ولكنه يستخدم Byte-level BPE كمحلل نحوي (نفس GPT-2) ويستخدم مخطط تدريب مختلف.

- لا يحتوي RoBERTa على `token_type_ids`، لست بحاجة إلى الإشارة إلى الرمز الذي ينتمي إلى الجزء. فقط قم بفصل أجزاءك باستخدام رمز الفصل `tokenizer.sep_token` (أو `</s>`)

- نفس BERT مع حيل التدريب المسبق الأفضل:

- التعتيم الديناميكي: يتم تعتيم الرموز بشكل مختلف في كل حقبة، في حين أن BERT يفعل ذلك مرة واحدة وإلى الأبد

- معًا للوصول إلى 512 رمزًا (لذلك تكون الجمل بترتيب قد يمتد عبر عدة مستندات)

- التدريب باستخدام دفعات أكبر

- استخدام BPE مع البايت كوحدة فرعية وليس الأحرف (بسبب أحرف Unicode)

- [CamemBERT](camembert) عبارة عن غلاف حول RoBERTa. راجع هذه الصفحة للحصول على أمثلة الاستخدام.

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (المشار إليها بـ 🌎) لمساعدتك في البدء باستخدام RoBERTa. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فيرجى فتح طلب سحب وسنراجعه! يجب أن يُظهر المورد بشكل مثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

- مدونة حول [البدء بتحليل المشاعر على تويتر](https://huggingface.co/blog/sentiment-analysis-twitter) باستخدام RoBERTa و [Inference API](https://huggingface.co/inference-api).

- مدونة حول [تصنيف الآراء باستخدام Kili وHugging Face AutoTrain](https://huggingface.co/blog/opinion-classification-with-kili) باستخدام RoBERTa.

- دفتر ملاحظات حول كيفية [ضبط نموذج RoBERTa للتحليل الدلالي](https://colab.research.google.com/github/DhavalTaunk08/NLP_scripts/blob/master/sentiment_analysis_using_roberta.ipynb). 🌎

- [`RobertaForSequenceClassification`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb).

- [`TFRobertaForSequenceClassification`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb).

- [`FlaxRobertaForSequenceClassification`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb).

- دليل مهام التصنيف النصي [](../tasks/sequence_classification)

- [`RobertaForTokenClassification`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb).

- [`TFRobertaForTokenClassification`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb).

- [`FlaxRobertaForTokenClassification`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification).

- فصل [تصنيف الرموز](https://huggingface.co/course/chapter7/2?fw=pt) من دورة 🤗 Hugging Face.

- دليل مهام تصنيف الرموز [](../tasks/token_classification)

- مدونة حول [كيفية تدريب نموذج لغة جديد من الصفر باستخدام Transformers وTokenizers](https://huggingface.co/blog/how-to-train) مع RoBERTa.

- [`RobertaForMaskedLM`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).

- [`TFRobertaForMaskedLM`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).

- [`FlaxRobertaForMaskedLM`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb).

- فصل [نمذجة اللغة المعتمة](https://huggingface.co/course/chapter7/3?fw=pt) من دورة 🤗 Hugging Face.

- دليل مهام نمذجة اللغة المعتمة [](../tasks/masked_language_modeling)

- مدونة حول [تسريع الاستدلال باستخدام Optimum وTransformers Pipelines](https://huggingface.co/blog/optimum-inference) مع RoBERTa للاستجواب.

- [`RobertaForQuestionAnswering`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb).

- [`TFRobertaForQuestionAnswering`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb).

- [`FlaxRobertaForQuestionAnswering`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering).

- فصل [الاستجواب](https://huggingface.co/course/chapter7/7?fw=pt) من دورة 🤗 Hugging Face.

- دليل مهام الاستجواب [](../tasks/question_answering)

**الاختيار من متعدد**

- [`RobertaForMultipleChoice`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb).

- [`TFRobertaForMultipleChoice`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb).

- دليل مهام الاختيار من متعدد [](../tasks/multiple_choice)

## RobertaConfig

[[autodoc]] RobertaConfig

## RobertaTokenizer

[[autodoc]] RobertaTokenizer

- build_inputs_with_special_tokens

- get_special_tokens_mask

- create_token_type_ids_from_sequences

- save_vocabulary

## RobertaTokenizerFast

[[autodoc]] RobertaTokenizerFast

- build_inputs_with_special_tokens

<frameworkcontent>

<pt>

## RobertaModel

[[autodoc]] RobertaModel

- forward

## RobertaForCausalLM

[[autodoc]] RobertaForCausalLM

- forward

## RobertaForMaskedLM

[[autodoc]] RobertaForMaskedLM

- forward

## RobertaForSequenceClassification

[[autodoc]] RobertaForSequenceClassification

- forward

## RobertaForMultipleChoice

[[autodoc]] RobertaForMultipleChoice

- forward

## RobertaForTokenClassification

[[autodoc]] RobertaForTokenClassification

- forward

## RobertaForQuestionAnswering

[[autodoc]] RobertaForQuestionAnswering

- forward

</pt>

<tf>

## TFRobertaModel

[[autodoc]] TFRobertaModel

- call

## TFRobertaForCausalLM

[[autodoc]] TFRobertaForCausalLM

- call

## TFRobertaForMaskedLM

[[autodoc]] TFRobertaForMaskedLM

- call

## TFRobertaForSequenceClassification


[[autodoc]] TFRobertaForSequenceClassification


- call

## TFRobertaForMultipleChoice

[[autodoc]] TFRobertaForMultipleChoice

- call

## TFRobertaForTokenClassification

[[autodoc]] TFRobertaForTokenClassification

- call

## TFRobertaForQuestionAnswering

[[autodoc]] TFRobertaForQuestionAnswering

- call

</tf>

<jax>

## FlaxRobertaModel

[[autodoc]] FlaxRobertaModel

- __call__

## FlaxRobertaForCausalLM

[[autodoc]] FlaxRobertaForCausalLM

- __call__

## FlaxRobertaForMaskedLM

[[autodoc]] FlaxRobertaForMaskedLM

- __call__

## FlaxRobertaForSequenceClassification

[[autodoc]] FlaxRobertaForSequenceClassification

- __call__

## FlaxRobertaForMultipleChoice

[[autodoc]] FlaxRobertaForMultipleChoice

- __call__

## FlaxRobertaForTokenClassification

[[autodoc]] FlaxRobertaForTokenClassification

- __call__

## FlaxRobertaForQuestionAnswering

[[autodoc]] FlaxRobertaForQuestionAnswering

- __call__

</jax>

</frameworkcontent>