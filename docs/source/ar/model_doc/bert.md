# BERT

## نظرة عامة

اقترح نموذج BERT في ورقة "BERT: معالجة تمهيدية لمحولات ثنائية الاتجاه عميقة لفهم اللغة" بواسطة Jacob Devlin و Ming-Wei Chang و Kenton Lee و Kristina Toutanova. وهو عبارة عن محول ثنائي الاتجاه معالج مسبقًا باستخدام مزيج من أهداف نمذجة اللغة المقنعة والتنبؤ بالجملة التالية على مجموعة بيانات كبيرة تتكون من مجموعة بيانات Toronto Book Corpus وويكيبيديا.

ملخص الورقة هو ما يلي:

> "نقدم نموذج تمثيل لغة جديد يسمى BERT، وهو اختصار لـ Bidirectional Encoder Representations from Transformers. وعلى عكس نماذج تمثيل اللغة الحديثة، صمم BERT لمعالجة التمثيلات ثنائية الاتجاه مسبقًا من النص غير الموسوم عن طريق الشرط المشترك على السياق الأيسر والأيمن في جميع الطبقات. ونتيجة لذلك، يمكن ضبط نموذج BERT مسبقًا باستخدام طبقة إخراج إضافية واحدة فقط لإنشاء نماذج رائدة في الصناعة لمجموعة واسعة من المهام، مثل الإجابة على الأسئلة والاستدلال اللغوي، دون تعديلات معمارية محددة للمهمة."

> "إن BERT بسيط من الناحية المفاهيمية وقوي من الناحية التجريبية. فهو يحقق نتائج جديدة رائدة في الصناعة في إحدى عشرة مهمة لمعالجة اللغات الطبيعية، بما في ذلك رفع نتيجة اختبار GLUE إلى 80.5% (تحسن مطلق بنسبة 7.7%)، ودقة MultiNLI إلى 86.7% (تحسن مطلق بنسبة 4.6%)، ونتيجة F1 لاختبار SQuAD v1.1 للإجابة على الأسئلة إلى 93.2 (تحسن مطلق بنسبة 1.5 نقطة) ونتيجة F1 لاختبار SQuAD v2.0 إلى 83.1 (تحسن مطلق بنسبة 5.1 نقطة). "

## نصائح الاستخدام

- BERT هو نموذج مع تضمين موضع مطلق، لذلك يُنصح عادةً بإضافة مسافات إلى الإدخالات من اليمين بدلاً من اليسار.

- تم تدريب BERT باستخدام أهداف نمذجة اللغة المقنعة (MLM) والتنبؤ بالجملة التالية (NSP). إنه فعال في التنبؤ بالرموز المقنعة وفي فهم اللغة الطبيعية بشكل عام، ولكنه ليس الأمثل لتوليد النصوص.

- يقوم بتشويه الإدخالات باستخدام التعتيم العشوائي، وبشكل أكثر تحديدًا، أثناء المعالجة المسبقة، يتم تعتيم نسبة مئوية معينة من الرموز (عادة 15%) بالطرق التالية:

  - رمز قناع خاص باحتمالية 0.8
  - رمز عشوائي مختلف عن الرمز المعتم باحتمالية 0.1
  - نفس الرمز باحتمالية 0.1

- يجب على النموذج التنبؤ بالجملة الأصلية، ولكن لديه هدف ثانٍ: الإدخالات عبارة عن جملتين A و B (مع وجود رمز فصل بينهما). باحتمال 50%، تكون الجمل متتالية في المجموعة، وفي الـ 50% المتبقية لا تكون ذات صلة. يجب على النموذج التنبؤ بما إذا كانت الجمل متتالية أم لا.

### استخدام الانتباه لمنتج النقاط المحدد (SDPA)

يتضمن PyTorch مشغل اهتمام منتج النقاط المحدد الأصلي (SDPA) كجزء من `torch.nn.functional`. تشمل هذه الوظيفة عدة تنفيذات يمكن تطبيقها اعتمادًا على الإدخالات والأجهزة المستخدمة. راجع [الوثائق الرسمية](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) أو صفحة [GPU Inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention) لمزيد من المعلومات.

يتم استخدام SDPA بشكل افتراضي لـ `torch>=2.1.1` عندما يكون التنفيذ متاحًا، ولكن يمكنك أيضًا تعيين `attn_implementation="sdpa"` في `from_pretrained()` لطلب استخدام SDPA بشكل صريح.

```
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa")
...
```

للحصول على أفضل التحسينات، نوصي بتحميل النموذج بنصف الدقة (على سبيل المثال، `torch.float16` أو `torch.bfloat16`).

على معيار محلي (A100-80GB، CPUx12، RAM 96.6GB، PyTorch 2.2.0، نظام التشغيل Ubuntu 22.04) مع `float16`، رأينا التحسينات التالية أثناء التدريب والاستدلال.

#### التدريب

| batch_size | seq_len | Time per batch (eager - s) | Time per batch (sdpa - s) | Speedup (%) | Eager peak mem (MB) | sdpa peak mem (MB) | Mem saving (%) |
| ---------- | ------ | ------------------------- | ------------------------ | ----------- | ------------------- | ------------------ | -------------- |
| 4          | 256    | 0.023                     | 0.017                    | 35.472      | 939.213            | 764.834           | 22.800        |
| 4          | 512    | 0.023                     | 0.018                    | 23.687      | 1970.447           | 1227.162          | 60.569        |
| 8          | 256    | 0.023                     | 0.018                    | 23.491      | 1594.295           | 1226.114          | 30.028        |
| 8          | 512    | 0.035                     | 0.025                    | 43.058      | 3629.401           | 2134.262          | 70.054        |
| 16         | 256    | 0.030                     | 0.024                    | 25.583      | 2874.426           | 2134.262          | 34.680        |
| 16         | 512    | 0.064                     | 0.044                    | 46.223      | 6964.659           | 3961.013          | 75.830        |

#### الاستدلال

| batch_size | seq_len | Per token latency eager (ms) | Per token latency SDPA (ms) | Speedup (%) | Mem eager (MB) | Mem BT (MB) | Mem saved (%) |
| ---------- | ------ | ---------------------------- | --------------------------- | ----------- | -------------- | ----------- | ------------- |
| 1          | 128    | 5.736                        | 4.987                       | 15.022      | 282.661       | 282.924    | -0.093        |
| 1          | 256    | 5.689                        | 4.945                       | 15.055      | 298.686       | 298.948    | -0.088        |
| 2          | 128    | 6.154                        | 4.982                       | 23.521      | 314.523       | 314.785    | -0.083        |
| 2          | 256    | 6.201                        | 4.949                       | 25.303      | 347.546       | 347.033    | 0.148         |
| 4          | 128    | 6.049                        | 4.987                       | 21.305      | 378.895       | 379.301    | -0.107        |
| 4          | 256    | 6.285                        | 5.364                       | 17.166      | 443.209       | 444.382    | -0.264        |
## الموارد

قائمة بموارد Hugging Face الرسمية وموارد المجتمع (يشار إليها بالرمز 🌎) لمساعدتك على البدء مع BERT. إذا كنت مهتمًا بتقديم مورد لإضافته هنا، فيرجى فتح طلب سحب Pull Request وسنقوم بمراجعته! ويفضل أن يظهر المورد شيئًا جديدًا بدلاً من تكرار مورد موجود.

- منشور مدونة حول [تصنيف النصوص باستخدام BERT بلغة مختلفة](https://www.philschmid.de/bert-text-classification-in-a-different-language).
- دفتر ملاحظات حول [ضبط دقة BERT (وأصدقائه) لتصنيف النصوص متعدد التصنيفات](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb).
- دفتر ملاحظات حول كيفية [ضبط دقة BERT لتصنيف متعدد التصنيفات باستخدام PyTorch](https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb). 🌎
- دفتر ملاحظات حول كيفية [بدء تشغيل نموذج EncoderDecoder باستخدام BERT للتلخيص](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/BERT2BERT_for_CNN_Dailymail.ipynb).
- [`BertForSequenceClassification`] مدعوم بواسطة [سكريبت مثال](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) و[دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb).
- [`TFBertForSequenceClassification`] مدعوم بواسطة [سكريبت مثال](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/text-classification) و[دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb).
- [`FlaxBertForSequenceClassification`] مدعوم بواسطة [سكريبت مثال](https://github.com0/huggingface/transformers/tree/main/examples/flax/text-classification) و[دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification_flax.ipynb).
- دليل مهام تصنيف النصوص [Text classification task guide](../tasks/sequence_classification)

- منشور مدونة حول كيفية [استخدام Hugging Face Transformers مع Keras: ضبط دقة BERT غير الإنجليزي للتعرف على الكيانات المسماة](https://www.philschmid.de/huggingface-transformers-keras-tf).
- دفتر ملاحظات حول [ضبط دقة BERT للتعرف على الكيانات المسماة](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb) باستخدام قطعة الكلمة الأولى فقط من كل كلمة في تسمية الكلمة أثناء عملية التجزئة. ولنشر تسمية الكلمة إلى جميع القطع، راجع [هذا الإصدار](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb) من دفتر الملاحظات بدلاً من ذلك.
- [`BertForTokenClassification`] مدعوم بواسطة [سكريبت مثال](https://github.com/huggingface/transformers/tree/main/examples/pytorch/token-classification) و[دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb).
- [`TFBertForTokenClassification`] مدعوم بواسطة [سكريبت مثال](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/token-classification) و[دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification-tf.ipynb).
- [`FlaxBertForTokenClassification`] مدعوم بواسطة [سكريبت مثال](https://github.com/huggingface/transformers/tree/main/examples/flax/token-classification).
- فصل [تصنيف الرموز](https://huggingface.co/course/chapter7/2?fw=pt) في دورة 🤗 Hugging Face.
- دليل مهام تصنيف الرموز [Token classification task guide](../tasks/token_classification)

- [`BertForMaskedLM`] مدعوم بواسطة [سكريبت مثال](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) و[دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
- [`TFBertForMaskedLM`] مدعوم بواسطة [سكريبت مثال](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) و[دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).
- [`FlaxBertForMaskedLM`] مدعوم بواسطة [سكريبت مثال](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) و[دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb).
- فصل [نمذجة اللغة المقنعة](https://huggingface.co/course/chapter7/3?fw=pt) في دورة 🤗 Hugging Face.
- دليل مهام نمذجة اللغة المقنعة [Masked language modeling task guide](../tasks/masked_language_modeling)

- [`BertForQuestionAnswering`] مدعوم بواسطة [سكريبت مثال](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) و[دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb).
- [`TFBertForQuestionAnswering`] مدعوم بواسطة [سكريبت مثال](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/question-answering) و[دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering-tf.ipynb).
- [`FlaxBertForQuestionAnswering`] مدعوم بواسطة [سكريبت مثال](https://github.com/huggingface/transformers/tree/main/examples/flax/question-answering).
- فصل [الإجابة على الأسئلة](https://huggingface.co/course/chapter7/7?fw=pt) في دورة 🤗 Hugging Face.
- دليل مهام الإجابة على الأسئلة [Question answering task guide](../tasks/question_answering)

**الاختيار من متعدد**

- [`BertForMultipleChoice`] مدعوم بواسطة [سكريبت مثال](https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice) و[دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb).
- [`TFBertForMultipleChoice`] مدعوم بواسطة [سكريبت مثال](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/multiple-choice) و[دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice-tf.ipynb).
- دليل مهام الاختيار من متعدد [Multiple choice task guide](../tasks/multiple_choice)

⚡️ **الاستنتاج**

- منشور مدونة حول كيفية [تسريع استنتاج BERT باستخدام Hugging Face Transformers وAWS Inferentia](https://huggingface.co/blog/bert-inferentia-sagemaker).
- منشور مدونة حول كيفية [تسريع استنتاج BERT باستخدام DeepSpeed-Inference على وحدات معالجة الرسوميات GPU](https://www.philschmid.de/bert-deepspeed-inference).

⚙️ **التدريب المسبق**

- منشور مدونة حول [التدريب المسبق لـ BERT باستخدام Hugging Face Transformers وHabana Gaudi](https://www.philschmid.de/pre-training-bert-habana).

🚀 **النشر**

- منشور مدونة حول كيفية [تحويل Transformers إلى ONNX باستخدام Hugging Face Optimum](https://www.philschmid.de/convert-transformers-to-onnx).
- منشور مدونة حول كيفية [إعداد بيئة التعلم العميق لـ Hugging Face Transformers مع Habana Gaudi على AWS](https://www.philschmid.de/getting-started-habana-gaudi#conclusion).
- منشور مدونة حول [التوسيع التلقائي لـ BERT باستخدام Hugging Face Transformers وAmazon SageMaker ونماذج Terraform](https://www.philschmid.de/terraform-huggingface-amazon-sagemaker-advanced).
- منشور مدونة حول [تطبيق BERT بدون خادم باستخدام HuggingFace وAWS Lambda وDocker](https://www.philschmid.de/serverless-bert-with-huggingface-aws-lambda-docker).
- منشور مدونة حول [ضبط دقة BERT باستخدام Hugging Face Transformers وAmazon SageMaker وTraining Compiler](https://www.philschmid.de/huggingface-amazon-sagemaker-training-compiler).
- منشور مدونة حول [التقطير المعرفي المحدد للمهمة لـ BERT باستخدام Transformers وAmazon SageMaker](https://www.philschmid.de/knowledge-distillation-bert-transformers).

## BertConfig

[[autodoc]] BertConfig

- all

## BertTokenizer

[[autodoc]] BertTokenizer

- build_inputs_with_special_tokens
- get_special_tokens_mask
- create_token_type_ids_from_sequences
- save_vocabulary

<frameworkcontent>

<pt>

## BertTokenizerFast

[[autodoc]] BertTokenizerFast

</pt>

<tf>

## TFBertTokenizer

[[autodoc]] TFBertTokenizer

</tf>

</frameworkcontent>

## المخرجات الخاصة بـ Bert

[[autodoc]] models.bert.modeling_bert.BertForPreTrainingOutput

[[autodoc]] models.bert.modeling_tf_bert.TFBertForPreTrainingOutput

[[autodoc]] models.bert.modeling_flax_bert.FlaxBertForPreTrainingOutput

<frameworkcontent>

<pt>

## BertModel

[[autodoc]] BertModel

- forward

## BertForPreTraining

[[autodoc]] BertForPreTraining

- forward

## BertLMHeadModel

[[autodoc]] BertLMHeadModel

- forward

## BertForMaskedLM

[[autodoc]] BertForMaskedLM

- forward

## BertForNextSentencePrediction

[[autodoc]] BertForNextSentencePrediction

- forward

## BertForSequenceClassification

[[autodoc]] BertForSequenceClassification

- forward

## BertForMultipleChoice

[[autodoc]] BertForMultipleChoice

- forward

## BertForTokenClassification

[[autodoc]] BertForTokenClassification

- forward

## BertForQuestionAnswering

[[autodoc]] BertForQuestionAnswering

- forward

</pt>

<tf>

## TFBertModel

[[autodoc]] TFBertModel

- call

## TFBertForPreTraining

[[autodoc]] TFBertForPreTraining

- call

## TFBertModelLMHeadModel

[[autodoc]] TFBertLMHeadModel

- call

## TFBertForMaskedLM

[[autodoc]] TFBertForMaskedLM

- call

## TFBertForNextSentencePrediction

[[autodoc]] TFBertForNextSentencePrediction

- call

## TFBertForSequenceClassification

[[autodoc]] TFBertForSequenceClassification

- call

## TFBertForMultipleChoice

[[autodoc]] TFBertForMultipleChoice

- call

## TFBertForTokenClassification

[[autodoc]] TFBertForTokenClassification

- call

## TFBertForQuestionAnswering

[[autodoc]] TFBertForQuestionAnswering

- call

</tf>

<jax>

## FlaxBertModel

[[autodoc]] FlaxBertModel

- __call__

## FlaxBertForPreTraining

[[autodoc]] FlaxBertForPreTraining

- __call__
## FlaxBertForCausalLM
نموذج FlaxBertForCausalLM هو نموذج لغة قائم على تحويل فلامنغوBERT الذي تم تدريبه على التنبؤ بالكلمة التالية في جملة ما. يمكن استخدامه لتوليد النصوص أو لإكمال الجمل بشكل تلقائي. يتم تدريب النموذج على مجموعة كبيرة من النصوص غير الموسومة، مما يمكنه من فهم سياق اللغة والتنبؤ بالكلمات التالية في الجملة.

## FlaxBertForMaskedLM
يعد FlaxBertForMaskedLM نموذجًا للغة يستخدم بنية BERT الشهيرة. تم تدريبه على مهمة لغة Masked LM، والتي تتضمن التنبؤ بالكلمات الناقصة أو "المقنعة" في الجملة. يمكن استخدام هذا النموذج في مجموعة متنوعة من مهام معالجة اللغة الطبيعية، مثل فهم اللغة الطبيعية، والتصنيف النصي، واسترجاع المعلومات.

## FlaxBertForNextSentencePrediction
FlaxBertForNextSentencePrediction هو نموذج تعلم عميق مصمم للتنبؤ بما إذا كانت جملتين متتاليتين تشكلان جملة متماسكة منطقيًا. إنه يعتمد على بنية BERT الشهيرة، والتي تم تدريبها على كميات هائلة من البيانات النصية. يمكن استخدام هذا النموذج في تطبيقات مثل فهم اللغة الطبيعية، وتحليل المشاعر، وتصنيف النصوص.

## FlaxBertForSequenceClassification
FlaxBertForSequenceClassification هو نموذج تعلم آلي مصمم لتصنيف التسلسلات أو الجمل النصية إلى فئات أو تسميات محددة. يعتمد على بنية BERT الشهيرة، والتي تم تدريبها على كميات كبيرة من النصوص غير الموسومة. يمكن استخدام هذا النموذج في مجموعة متنوعة من مهام معالجة اللغة الطبيعية، مثل تصنيف المشاعر، وتصنيف الموضوعات، وتحديد القصد.

## FlaxBertForMultipleChoice
FlaxBertForMultipleChoice هو نموذج تعلم آلي مصمم للإجابة على أسئلة الاختيار من متعدد. إنه يعتمد على بنية BERT الشهيرة، والتي تم تدريبها على فهم اللغة الطبيعية ومعالجة النصوص المعقدة. يمكن للنموذج تحليل السياق والمحتوى في السؤال والخيارات، وجعل التنبؤات الدقيقة.

## FlaxBertForTokenClassification
FlaxBertForTokenClassification هو نموذج تعلم آلي قوي مصمم لتصنيف الرموز أو الكلمات في النص إلى فئات أو تسميات محددة. يمكن أن يكون هذا التصنيف على مستوى الكلمة، مثل تسمية الأجزاء النحوية أو تسمية الكيانات المسماة، أو على مستوى الرمز، مثل تصنيف الرموز حسب نوعها أو وظيفتها.

## FlaxBertForQuestionAnswering
FlaxBertForQuestionAnswering هو نموذج تعلم آلي مصمم للإجابة على الأسئلة باستخدام سياق أو مستند نصي معين. يعتمد على بنية BERT الشهيرة، والتي تم تدريبها على فهم اللغة الطبيعية ومعالجة السياق المعقد. يمكن للنموذج استخراج الإجابات من النص، والتعامل مع الأسئلة المعقدة، وتوفير إجابات دقيقة ومتماسكة.