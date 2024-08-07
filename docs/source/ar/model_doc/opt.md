# OPT

## نظرة عامة
اقترح نموذج OPT في [نماذج المحول اللغوي المسبقة التدريب المفتوحة](https://arxiv.org/pdf/2205.01068) بواسطة Meta AI.
OPT عبارة عن سلسلة من نماذج اللغة السببية مفتوحة المصدر الكبيرة التي تتمتع بأداء مشابه لـ GPT3.

المستخلص من الورقة هو ما يلي:

*أظهرت نماذج اللغة الكبيرة، التي غالبًا ما يتم تدريبها لمئات الآلاف من أيام الحوسبة، قدرات ملحوظة للتعلم بدون بيانات أو بقدر قليل من البيانات. ونظرًا لتكلفتها الحسابية، يصعب تكرار هذه النماذج بدون رأس مال كبير. وبالنسبة للقليل منها المتاح من خلال واجهات برمجة التطبيقات، لا يُسمح بالوصول إلى أوزان النموذج الكاملة، مما يجعل دراستها أمرًا صعبًا. نقدم Open Pre-trained Transformers (OPT)، وهي مجموعة من المحولات المُشفرة مسبقًا والتي تتراوح من 125 مليون إلى 175 مليار معامل، والتي نهدف إلى مشاركتها بالكامل وبشكل مسؤول مع الباحثين المهتمين. ونظهر أن OPT-175B مماثل لـ GPT-3، في حين أنه يتطلب فقط 1/7 من البصمة الكربونية لتطويره. كما أننا نطلق دفتر ملاحظاتنا الذي يفصل التحديات التي واجهناها على مستوى البنية التحتية، إلى جانب التعليمات البرمجية لتجربة جميع النماذج التي تم إصدارها.*

تمت المساهمة بهذا النموذج من قبل [Arthur Zucker](https://huggingface.co/ArthurZ) و [Younes Belkada](https://huggingface.co/ybelkada) و [Patrick Von Platen](https://huggingface.co/patrickvonplaten).
يمكن العثور على الكود الأصلي [هنا](https://github.com/facebookresearch/metaseq).

النصائح:

- لدي OPT نفس بنية [`BartDecoder`].
- على عكس GPT2، يضيف OPT رمز EOS `</s>` في بداية كل موجه.

## الموارد

فيما يلي قائمة بموارد Hugging Face الرسمية وموارد المجتمع (المشار إليها بـ 🌎) لمساعدتك في البدء في استخدام OPT. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فيرجى فتح طلب سحب وسنراجعه.

من الناحية المثالية، يجب أن يثبت المورد شيئًا جديدًا بدلاً من تكرار مورد موجود.

<PipelineTag pipeline="text-generation" />

- دفتر ملاحظات حول [ضبط نموذج OPT الدقيق باستخدام PEFT و bitsandbytes و Transformers](https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing). 🌎
- منشور مدونة حول [استراتيجيات فك التشفير باستخدام OPT](https://huggingface.co/blog/introducing-csearch#62-example-two---opt).
- الفصل الخاص بـ [نمذجة اللغة السببية](https://huggingface.co/course/en/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch) من دورة 🤗 Hugging Face.
- [`OPTForCausalLM`] مدعوم بواسطة [مثال على نص برمجي لنمذجة اللغة السببية](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
- [`TFOPTForCausalLM`] مدعوم بواسطة [مثال على نص برمجي لنمذجة اللغة السببية](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).
- [`FlaxOPTForCausalLM`] مدعوم بواسطة [مثال على نص برمجي لنمذجة اللغة السببية](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#causal-language-modeling).

<PipelineTag pipeline="text-classification" />

- دليل [مهمة تصنيف النص](sequence_classification.md)
- [`OPTForSequenceClassification`] مدعوم بواسطة [مثال على نص برمجي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb).

<PipelineTag pipeline="question-answering" />

- [`OPTForQuestionAnswering`] مدعوم بواسطة [مثال على نص برمجي للإجابة على الأسئلة](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb).
- الفصل الخاص بـ [الإجابة على الأسئلة](https://huggingface.co/course/chapter7/7?fw=pt) من دورة 🤗 Hugging Face.

⚡️ الاستنتاج

- منشور مدونة حول [كيفية تشغيل 🤗 Accelerate لنماذج كبيرة جدًا بفضل PyTorch](https://huggingface.co/blog/accelerate-large-models) مع OPT.

## الجمع بين OPT و Flash Attention 2

أولاً، تأكد من تثبيت أحدث إصدار من Flash Attention 2 لتضمين ميزة نافذة الاهتمام المنزلقة.

```bash
pip install -U flash-attn --no-build-isolation
```

تأكد أيضًا من أن لديك أجهزة متوافقة مع Flash-Attention 2. اقرأ المزيد عنها في الوثائق الرسمية لمستودع flash-attn. تأكد أيضًا من تحميل نموذجك في نصف الدقة (على سبيل المثال `torch.float16``).

لتحميل وتشغيل نموذج باستخدام Flash Attention 2، راجع المقتطف أدناه:

```python
>>> import torch
>>> from transformers import OPTForCausalLM, GPT2Tokenizer
>>> device = "cuda" # الجهاز الذي سيتم تحميل النموذج عليه

>>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
>>> tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")

>>> prompt = ("A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the "
"Statue of Liberty.\nHuman: Where do you live?\nStatue: New York City.\nHuman: How long have you lived "
"there?")

>>> model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
>>> model.to(device)

>>> generated_ids = model.generate(**model_inputs, max_new_tokens=30, do_sample=False)
>>> tokenizer.batch_decode(generated_ids)[0]
'</s>A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the Statue of Liberty.\nHuman: Where do you live?\nStatue: New York City.\nHuman: How long have you lived there?\nStatue: I have lived here for about a year.\nHuman: What is your favorite place to eat?\nStatue: I love'
```

### تسريع المتوقع

فيما يلي رسم بياني للتسريع المتوقع الذي يقارن وقت الاستدلال النقي بين التنفيذ الأصلي في المحولات باستخدام نقطة تفتيش `facebook/opt-2.7b` وإصدار Flash Attention 2 من النموذج باستخدام طولين تسلسليين مختلفين.

<div style="text-align: center">
<img src="https://user-images.githubusercontent.com/49240599/281101546-d2fca6d2-ee44-48f3-9534-ba8d5bee4531.png">
</div>

فيما يلي رسم بياني للتسريع المتوقع الذي يقارن وقت الاستدلال النقي بين التنفيذ الأصلي في المحولات باستخدام نقطة تفتيش `facebook/opt-350m` وإصدار Flash Attention 2 من النموذج باستخدام طولين تسلسليين مختلفين.

<div style="text-align-center">
<img src="https://user-images.githubusercontent.com/49240599/281101682-d1144e90-0dbc-46f4-8fc8-c6206cb793c9.png">
</div>

## OPTConfig

[[autodoc]] OPTConfig

<frameworkcontent>
<pt>

## OPTModel

[[autodoc]] OPTModel

- forward

## OPTForCausalLM

[[autodoc]] OPTForCausalLM

- forward

## OPTForSequenceClassification

[[autodoc]] OPTForSequenceClassification

- forward

## OPTForQuestionAnswering

[[autodoc]] OPTForQuestionAnswering

- forward

</pt>
<tf>

## TFOPTModel

[[autodoc]] TFOPTModel

- call

## TFOPTForCausalLM

[[autodoc]] TFOPTForCausalLM

- call

</tf>
<jax>

## FlaxOPTModel

[[autodoc]] FlaxOPTModel

- __call__

## FlaxOPTForCausalLM

[[autodoc]] FlaxOPTForCausalLM

- __call__

</jax>
</frameworkcontent>