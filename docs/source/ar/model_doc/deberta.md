# DeBERTa

## نظرة عامة
اقترح Pengcheng He وآخرون نموذج DeBERTa في ورقتهم البحثية بعنوان "DeBERTa: Decoding-enhanced BERT with Disentangled Attention". وهو مبني على نموذج BERT من Google الذي تم إصداره في عام 2018، ونموذج RoBERTa من Facebook الذي تم إصداره في عام 2019.

يستند DeBERTa إلى نموذج RoBERTa مع إضافة آلية الانتباه المنفصلة (disentangled attention) وتعزيز تدريب فك تشفير القناع (enhanced mask decoder training) باستخدام نصف كمية البيانات المستخدمة في RoBERTa.

وفيما يلي الملخص المستخرج من الورقة البحثية:

"حققت التقدمات الأخيرة في نماذج اللغة العصبية المُدربة مسبقًا تحسينات كبيرة في أداء العديد من مهام معالجة اللغة الطبيعية (NLP). وفي هذه الورقة، نقترح بنية نموذج جديدة تسمى DeBERTa (Decoding-enhanced BERT with disentangled attention) والتي تحسن نماذج BERT وRoBERTa باستخدام تقنيتين جديدتين. الأولى هي آلية الانتباه المنفصلة، حيث يتم تمثيل كل كلمة باستخدام متجهين يشفر أحدهما المحتوى والآخر الموضع، ويتم حساب أوزان الانتباه بين الكلمات باستخدام مصفوفات منفصلة لمحتوياتها ومواضعها النسبية. أما التقنية الثانية، فهي استخدام فك تشفير القناع المعزز ليحل محل طبقة softmax الإخراجية للتنبؤ بالرموز المميزة المقنعة لتدريب النموذج المُسبق. ونُظهر أن هاتين التقنيتين تحسنان بشكل كبير من كفاءة تدريب النموذج المُسبق وأداء المهام اللاحقة. مقارنة بنموذج RoBERTa-Large، يحقق نموذج DeBERTa المُدرب على نصف بيانات التدريب نتائج أفضل باستمرار في مجموعة واسعة من مهام NLP، حيث يحقق تحسينات بنسبة +0.9% في MNLI (من 90.2% إلى 91.1%)، و+2.3% في SQuAD v2.0 (من 88.4% إلى 90.7%)، و+3.6% في RACE (من 83.2% إلى 86.8%). ستكون شفرة DeBERTa والنماذج المُدربة مسبقًا متاحة للجمهور على https://github.com/microsoft/DeBERTa."

تمت المساهمة بهذا النموذج من قبل [DeBERTa] (https://huggingface.co/DeBERTa). وتمت المساهمة في تنفيذ TF 2.0 بواسطة [kamalkraj] (https://huggingface.co/kamalkraj). ويمكن العثور على الشفرة الأصلية [هنا] (https://github.com/microsoft/DeBERTa).

## الموارد
فيما يلي قائمة بموارد Hugging Face الرسمية وموارد المجتمع (مشار إليها بـ 🌎) لمساعدتك في البدء باستخدام DeBERTa. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فيرجى فتح طلب سحب (Pull Request) وسنقوم بمراجعته! ويفضل أن يُظهر المورد شيئًا جديدًا بدلاً من تكرار مورد موجود.

- منشور مدونة حول كيفية [تسريع تدريب النماذج الكبيرة باستخدام DeepSpeed] (https://huggingface.co/blog/accelerate-deepspeed) مع DeBERTa.

- منشور مدونة حول [تعزيز خدمة العملاء باستخدام التعلم الآلي] (https://huggingface.co/blog/supercharge-customer-service-with-machine-learning) مع DeBERTa.

- [`DebertaForSequenceClassification`] مدعوم بواسطة [مثال النص البرمجي هذا] (https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) و [دفتر الملاحظات] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb).

- [`TFDebertaForSequenceClassification`] مدعوم بواسطة [مثال النص البرمجي هذا] (https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification) و [دفتر الملاحظات] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb).

- دليل مهام التصنيف النصي [هنا] (../tasks/sequence_classification).

- [`DebertaForTokenClassification`] مدعوم بواسطة [مثال النص البرمجي هذا] (https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) و [دفتر الملاحظات] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb).

- [`TFDebertaForTokenClassification`] مدعوم بواسطة [مثال النص البرمجي هذا] (https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) و [دفتر الملاحظات] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb).

- فصل [تصنيف الرموز] (https://huggingface.co/course/chapter7/2?fw=pt) من دورة 🤗 Hugging Face Course.

- فصل [ترميز الاقتران البايت ثنائي] (https://huggingface.co/course/chapter6/5?fw=pt) من دورة 🤗 Hugging Face Course.

- دليل مهام تصنيف الرموز [هنا] (../tasks/token_classification).

- [`DebertaForMaskedLM`] مدعوم بواسطة [مثال النص البرمجي هذا] (https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) و [دفتر الملاحظات] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).

- [`TFDebertaForMaskedLM`] مدعوم بواسطة [مثال النص البرمجي هذا] (https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) و [دفتر الملاحظات] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).

- فصل [نمذجة اللغة المقنعة] (https://huggingface.co/course/chapter7/3?fw=pt) من دورة 🤗 Hugging Face Course.

- دليل مهام نمذجة اللغة المقنعة [هنا] (../tasks/masked_language_modeling).

- [`DebertaForQuestionAnswering`] مدعوم بواسطة [مثال النص البرمجي هذا] (https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) و [دفتر الملاحظات] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb).

- [`TFDebertaForQuestionAnswering`] مدعوم بواسطة [مثال النص البرمجي هذا] (https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) و [دفتر الملاحظات] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb).

- فصل [الإجابة على الأسئلة] (https://huggingface.co/course/chapter7/7?fw=pt) من دورة 🤗 Hugging Face Course.

- دليل مهام الإجابة على الأسئلة [هنا] (../tasks/question_answering).

## DebertaConfig

[[autodoc]] DebertaConfig

## DebertaTokenizer

[[autodoc]] DebertaTokenizer

- build_inputs_with_special_tokens

- get_special_tokens_mask

- create_token_type_ids_from_sequences

- save_vocabulary

## DebertaTokenizerFast

[[autodoc]] DebertaTokenizerFast

- build_inputs_with_special_tokens

- create_token_type_ids_from_sequences

<frameworkcontent>

<pt>

## DebertaModel

[[autodoc]] DebertaModel

- forward

## DebertaPreTrainedModel

[[autodoc]] DebertaPreTrainedModel

## DebertaForMaskedLM

[[autodoc]] DebertaForMaskedLM

- forward

## DebertaForSequenceClassification

[[autodoc]] DebertaForSequenceClassification

- forward

## DebertaForTokenClassification

[[autodoc]] DebertaForTokenClassification

- forward

## DebertaForQuestionAnswering

[[autodoc]] DebertaForQuestionAnswering

- forward

</pt>

<tf>

## TFDebertaModel

[[autodoc]] TFDebertaModel

- call

## TFDebertaPreTrainedModel

[[autodoc]] TFDebertaPreTrainedModel

- call

## TFDebertaForMaskedLM

[[autodoc]] TFDebertaForMaskedLM

- call

## TFDebertaForSequenceClassification

[[autodoc]] TFDebertaForSequenceClassification

- call

## TFDebertaForTokenClassification

[[autodoc]] TFDebertaForTokenClassification

- call

## TFDebertaForQuestionAnswering

[[autodoc]] TFDebertaForQuestionAnswering

- call

</tf>

</frameworkcontent>