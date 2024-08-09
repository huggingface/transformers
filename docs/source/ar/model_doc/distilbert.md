# DistilBERT

## نظرة عامة

اقتُرح نموذج DistilBERT في المنشور على المدونة [Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT](https://medium.com/huggingface/distilbert-8cf3380435b5)، وورقة البحث [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108). DistilBERT هو نموذج Transformer صغير وسريع ورخيص وخفيف تم تدريبه عن طريق تقطير BERT base. لديه 40% معلمات أقل من *google-bert/bert-base-uncased*، ويعمل أسرع بنسبة 60% مع الحفاظ على أكثر من 95% من أداء BERT كما تم قياسه على معيار GLUE لفهم اللغة.

الملخص من الورقة هو ما يلي:

> مع انتشار تعلم النقل من النماذج الكبيرة المُدربة مسبقًا في معالجة اللغة الطبيعية (NLP)، لا يزال تشغيل هذه النماذج الكبيرة على الحافة و/أو ضمن ميزانيات التدريب أو الاستدلال المحوسب المقيد يمثل تحديًا. في هذا العمل، نقترح طريقة لتدريب نموذج تمثيل لغوي عام أصغر، يسمى DistilBERT، يمكن بعد ذلك ضبط دقته بدقة مع أداء جيد على مجموعة واسعة من المهام مثل نظرائه الأكبر حجمًا. في حين أن معظم الأعمال السابقة درست استخدام التقطير لبناء نماذج خاصة بمهام معينة، فإننا نستفيد من التقطير المعرفي أثناء مرحلة التدريب المسبق ونظهر أنه من الممكن تقليل حجم نموذج BERT بنسبة 40%، مع الاحتفاظ بنسبة 97% من قدراته على فهم اللغة وكونه أسرع بنسبة 60%. للاستفادة من الانحيازات الاستقرائية التي تعلمتها النماذج الأكبر أثناء التدريب المسبق، نقدم خسارة ثلاثية تجمع بين خسائر النمذجة اللغوية والتقطير والمسافة الكوسينية. إن نموذجنا الأصغر والأسرع والأخف أرخص في التدريب المسبق، ونحن نثبت قدراته على الحسابات داخل الجهاز في تجربة إثبات المفهوم ودراسة مقارنة على الجهاز.

تمت المساهمة بهذا النموذج من قبل [victorsanh](https://huggingface.co/victorsanh). تمت المساهمة بهذه النسخة من النموذج بواسطة [kamalkraj](https://huggingface.co/kamalkraj). يمكن العثور على الكود الأصلي [هنا](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation).

## نصائح الاستخدام

- لا يمتلك DistilBERT `token_type_ids`، لذا لا تحتاج إلى الإشارة إلى الرمز الذي ينتمي إلى أي مقطع. فقط قم بفصل مقاطعك باستخدام رمز الفصل `tokenizer.sep_token` (أو `[SEP]`).

- لا يمتلك DistilBERT خيارات لتحديد مواضع الإدخال (`position_ids` input). يمكن إضافة هذا إذا لزم الأمر، فقط أخبرنا إذا كنت بحاجة إلى هذا الخيار.

- مثل BERT ولكن أصغر. تم تدريبه عن طريق تقطير نموذج BERT المُدرب مسبقًا، مما يعني أنه تم تدريبه للتنبؤ بنفس الاحتمالات مثل النموذج الأكبر. الهدف الفعلي هو مزيج من:

   - العثور على نفس الاحتمالات مثل نموذج المعلم
   - التنبؤ بالرموز المُقنعة بشكل صحيح (ولكن بدون هدف الجملة التالية)
   - تشابه كوسيني بين الحالات المخفية للطالب والمعلم

## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (مشار إليها بـ 🌎) لمساعدتك في البدء باستخدام DistilBERT. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فالرجاء فتح طلب سحب Pull Request وسنراجعه! يجب أن يُظهر المورد بشكل مثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

- منشور مدونة حول [Getting Started with Sentiment Analysis using Python](https://huggingface.co/blog/sentiment-analysis-python) باستخدام DistilBERT.

- منشور مدونة حول كيفية [تدريب DistilBERT باستخدام Blurr للتصنيف التسلسلي](https://huggingface.co/blog/fastai).

- منشور مدونة حول كيفية استخدام [Ray لضبط دقة فرط معلمات DistilBERT](https://huggingface.co/blog/ray-tune).

- منشور مدونة حول كيفية [تدريب DistilBERT باستخدام Hugging Face وAmazon SageMaker](https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face).

- دفتر ملاحظات حول كيفية [ضبط دقة فرط معلمات DistilBERT للتصنيف متعدد التصنيفات](https://colab.research.google.com/github/DhavalTaunk08/Transformers_scripts/blob/master/Transformers_multilabel_distilbert.ipynb). 🌎

- دفتر ملاحظات حول كيفية [ضبط دقة فرط معلمات DistilBERT للتصنيف متعدد الفئات باستخدام PyTorch](https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multiclass_classification.ipynb). 🌎

- دفتر ملاحظات حول كيفية [ضبط دقة فرط معلمات DistilBERT للتصنيف النصي في TensorFlow](https://colab.research.google.com/github/peterbayerle/huggingface_notebook/blob/main/distilbert_tf.ipynb). 🌎

- [`DistilBertForSequenceClassification`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb).

- [`TFDistilBertForSequenceClassification`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb).

- [`FlaxDistilBertForSequenceClassification`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb).

- [دليل مهام التصنيف النصي](../tasks/sequence_classification)

- [`DistilBertForTokenClassification`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb).

- [`TFDistilBertForTokenClassification`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb).

- [`FlaxDistilBertForTokenClassification`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification).

- فصل [التصنيف الرمزي](https://huggingface.co/course/chapter7/2?fw=pt) من دورة 🤗 Hugging Face Course.

- [دليل مهام التصنيف الرمزي](../tasks/token_classification)

- [`DistilBertForMaskedLM`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).

- [`TFDistilBertForMaskedLM`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).

- [`FlaxDistilBertForMaskedLM`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb).

- فصل [النمذجة اللغوية المقنعة](https://huggingface.co/course/chapter7/3?fw=pt) من دورة 🤗 Hugging Face Course.

- [دليل مهام النمذجة اللغوية المقنعة](../tasks/masked_language_modeling)

- [`DistilBertForQuestionAnswering`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb).

- [`TFDistilBertForQuestionAnswering`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb).

- [`FlaxDistilBertForQuestionAnswering`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering).

- فصل [الأسئلة والأجوبة](https://huggingface.co/course/chapter7/7?fw=pt) من دورة 🤗 Hugging Face Course.

- [دليل مهام الأسئلة والأجوبة](../tasks/question_answering)

**الاختيار من متعدد**

- [`DistilBertForMultipleChoice`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb).

- [`TFDistilBertForMultipleChoice`] مدعوم بواسطة [مثال النص البرمجي](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb).

- [دليل مهام الاختيار من متعدد](../tasks/multiple_choice)

⚗️ التحسين

- منشور مدونة حول كيفية [تقطير DistilBERT باستخدام 🤗 Optimum وIntel](https://huggingface.co/blog/intel).

- منشور مدونة حول كيفية [تحسين المحولات للوحدات GPU باستخدام 🤗 Optimum](https://www.philschmid.de/optimizing-transformers-with-optimum-gpu).

- منشور مدونة حول [تحسين المحولات باستخدام Hugging Face Optimum](https://www.philschmid.de/optimizing-transformers-with-optimum).

⚡️ الاستدلال

- منشور مدونة حول كيفية [تسريع استدلال BERT باستخدام Hugging Face Transformers وAWS Inferentia](https://huggingface.co/blog/bert-inferentia-sagemaker) مع DistilBERT.

- منشور مدونة حول [Serverless Inference with Hugging Face's Transformers, DistilBERT and Amazon SageMaker](https://www.philschmid.de/sagemaker-serverless-huggingface-distilbert).

🚀 النشر

- منشور مدونة حول كيفية [نشر DistilBERT على Google Cloud](https://huggingface.co/blog/how-to-deploy-a-pipeline-to-google-clouds).

- منشور مدونة حول كيفية [نشر DistilBERT باستخدام Amazon SageMaker](https://huggingface.co/blog/deploy-hugging-face-models-easily-with-amazon-sagemaker).

- منشور مدونة حول كيفية [نشر BERT باستخدام Hugging Face Transformers وAmazon SageMaker ووحدة Terraform](https://www.philschmid.de/terraform-huggingface-amazon-sagemaker).

## الجمع بين DistilBERT وFlash Attention 2

أولاً، تأكد من تثبيت أحدث إصدار من Flash Attention 2 لتضمين ميزة نافذة الاهتمام المنزلقة.

```bash
pip install -U flash-attn --no-build-isolation
```

تأكد أيضًا من أن لديك أجهزة متوافقة مع Flash-Attention 2. اقرأ المزيد عنها في الوثائق الرسمية لمستودع flash-attn. تأكد أيضًا من تحميل نموذجك في نصف الدقة (مثل `torch.float16`)

لتحميل وتشغيل نموذج باستخدام Flash Attention 2، راجع المقتطف أدناه:

```python
>>> import torch
>>> from transformers import AutoTokenizer, AutoModel

>>> device = "cuda" # الجهاز الذي سيتم تحميل النموذج عليه

>>> tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
>>> model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased", torch_dtype=torch.float16, attn_implementation="flash_attention_2")

>>> text = "استبدلني بأي نص تريده."

>>> encoded_input = tokenizer(text, return_tensors='pt').to(device)
>>> model.to(device)

>>> output = model(**encoded_input)
```


## DistilBertConfig

[[autodoc]] DistilBertConfig

## DistilBertTokenizer

[[autodoc]] DistilBertTokenizer

## DistilBertTokenizerFast

[[autodoc]] DistilBertTokenizerFast

<frameworkcontent>
<pt>

## DistilBertModel

[[autodoc]] DistilBertModel
    - forward

## DistilBertForMaskedLM

[[autodoc]] DistilBertForMaskedLM
    - forward

## DistilBertForSequenceClassification

[[autodoc]] DistilBertForSequenceClassification
    - forward

## DistilBertForMultipleChoice

[[autodoc]] DistilBertForMultipleChoice
    - forward

## DistilBertForTokenClassification

[[autodoc]] DistilBertForTokenClassification
    - forward

## DistilBertForQuestionAnswering

[[autodoc]] DistilBertForQuestionAnswering
    - forward

</pt>
<tf>

## TFDistilBertModel

[[autodoc]] TFDistilBertModel
    - call

## TFDistilBertForMaskedLM

[[autodoc]] TFDistilBertForMaskedLM
    - call

## TFDistilBertForSequenceClassification

[[autodoc]] TFDistilBertForSequenceClassification
    - call

## TFDistilBertForMultipleChoice

[[autodoc]] TFDistilBertForMultipleChoice
    - call

## TFDistilBertForTokenClassification

[[autodoc]] TFDistilBertForTokenClassification
    - call

## TFDistilBertForQuestionAnswering

[[autodoc]] TFDistilBertForQuestionAnswering
    - call

</tf>
<jax>

## FlaxDistilBertModel

[[autodoc]] FlaxDistilBertModel
    - __call__

## FlaxDistilBertForMaskedLM

[[autodoc]] FlaxDistilBertForMaskedLM
    - __call__

## FlaxDistilBertForSequenceClassification

[[autodoc]] FlaxDistilBertForSequenceClassification
    - __call__

## FlaxDistilBertForMultipleChoice

[[autodoc]] FlaxDistilBertForMultipleChoice
    - __call__

## FlaxDistilBertForTokenClassification

[[autodoc]] FlaxDistilBertForTokenClassification
    - __call__

## FlaxDistilBertForQuestionAnswering

[[autodoc]] FlaxDistilBertForQuestionAnswering
    - __call__

</jax>
</frameworkcontent>