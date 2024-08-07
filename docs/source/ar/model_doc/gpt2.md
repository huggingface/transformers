# OpenAI GPT2

## نظرة عامة
اقترح نموذج OpenAI GPT-2 في [نماذج اللغة غير الخاضعة للإشراف هي متعلمات متعددة المهام](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) بواسطة Alec Radford و Jeffrey Wu و Rewon Child و David Luan و Dario Amodei و Ilya Sutskever من [OpenAI] (https://huggingface.co/openai). إنه محول أحادي الاتجاه تم تدريبه مسبقًا باستخدام نمذجة اللغة على مجموعة بيانات كبيرة جدًا تبلغ حوالي 40 جيجابايت من بيانات النص.

المستخلص من الورقة هو ما يلي:

> "GPT-2 هو نموذج لغة كبير قائم على المحول مع 1.5 مليار معلمة، تم تدريبه على مجموعة بيانات [1] من 8 ملايين صفحة ويب. يتم تدريب GPT-2 بهدف بسيط: التنبؤ بالكلمة التالية، بالنظر إلى جميع الكلمات السابقة داخل بعض النصوص. يؤدي تنوع مجموعة البيانات إلى احتواء هذا الهدف البسيط بشكل طبيعي على عروض توضيحية للعديد من المهام عبر المجالات المتنوعة. يعد GPT-2 عملية توسيع مباشرة لـ GPT، مع أكثر من 10X من المعلمات وتم تدريبه على أكثر من 10X من كمية البيانات."

[اكتب باستخدام المحول](https://transformer.huggingface.co/doc/gpt2-large) هو تطبيق ويب تم إنشاؤه واستضافته بواسطة Hugging Face والذي يوضح القدرات التوليدية لعدة نماذج. يعد GPT-2 أحد هذه النماذج وهو متاح بخمسة أحجام مختلفة: صغير ومتوسط وكبير وxl وإصدار مقطر من نقطة تفتيش صغيرة: *distilgpt-2*.

تمت المساهمة بهذا النموذج من قبل [thomwolf](https://huggingface.co/thomwolf). يمكن العثور على الكود الأصلي [هنا](https://openai.com/blog/better-language-models/).

## نصائح الاستخدام

- GPT-2 هو نموذج مع تضمين الموضع المطلق، لذلك يُنصح عادةً بتعبئة المدخلات على اليمين بدلاً من اليسار.

- تم تدريب GPT-2 بهدف نمذجة اللغة السببية (CLM) وبالتالي فهو قوي في التنبؤ بالرمز التالي في تسلسل. الاستفادة من هذه الميزة تسمح لـ GPT-2 بتوليد نص متماسك من الناحية التركيبية كما يمكن ملاحظته في مثال *run_generation.py* النصي.

- يمكن أن يأخذ النموذج *past_key_values* (لـ PyTorch) أو *past* (لـ TF) كإدخال، وهو أزواج الاهتمام الرئيسية/القيم المحسوبة مسبقًا. باستخدام هذا (*past_key_values* أو *past*) القيمة يمنع النموذج من إعادة حساب القيم المحسوبة مسبقًا في سياق توليد النص. بالنسبة لـ PyTorch، راجع حجة *past_key_values* لطريقة [`GPT2Model.forward`]، أو بالنسبة لـ TF، راجع حجة *past* لطريقة [`TFGPT2Model.call`] لمزيد من المعلومات حول استخدامه.

- سيطبق تمكين علامات *scale_attn_by_inverse_layer_idx* و *reorder_and_upcast_attn* تحسينات الاستقرار التدريبي من [Mistral](https://github.com/stanford-crfm/mistral/) (لـ PyTorch فقط).

## مثال الاستخدام

يمكن استخدام طريقة `generate()` لتوليد نص باستخدام نموذج GPT2.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")

>>> prompt = "GPT2 is a model developed by OpenAI."

>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

>>> gen_tokens = model.generate(
...     input_ids,
...     do_sample=True,
...     temperature=0.9,
...     max_length=100,
... )
>>> gen_text = tokenizer.batch_decode(gen_tokens)[0]
```

## استخدام فلاش الاهتمام 2

Flash Attention 2 هو إصدار أسرع وأكثر تحسينًا من حساب درجات الاهتمام والذي يعتمد على نواة `cuda`.

### التثبيت

أولاً، تحقق مما إذا كان الأجهزة الخاصة بك متوافقة مع Flash Attention 2. يمكن العثور على أحدث قائمة للأجهزة المتوافقة في [الوثائق الرسمية](https://github.com/Dao-AILab/flash-attention#installation-and-features). إذا لم يكن الأجهزة الخاص بك متوافقًا مع Flash Attention 2، فيمكنك الاستفادة من تحسينات نواة الاهتمام من خلال دعم المحول الأفضل المشمول [أعلاه](https://huggingface.co/docs/transformers/main/en/model_doc/bark#using-better-transformer).

بعد ذلك، قم بتثبيت أحدث إصدار من Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

### الاستخدام

لتحميل نموذج باستخدام Flash Attention 2، يمكننا تمرير الحجة `attn_implementation="flash_attention_2"` إلى [`.from_pretrained`](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained). سنقوم أيضًا بتحميل النموذج في نصف الدقة (على سبيل المثال `torch.float16`)، حيث يؤدي ذلك إلى انخفاض كبير في استخدام الذاكرة وسرعة الاستدلال مع انخفاض طفيف في جودة الصوت:

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> device = "cuda" # the device to load the model onto

>>> model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")

>>> prompt = "def hello_world():"

>>> model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
>>> model.to(device)

>>> generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
```

### تسريع المتوقع

فيما يلي رسم بياني للتسريع المتوقع الذي يقارن وقت الاستدلال النقي بين التنفيذ الأصلي في المحولات باستخدام نقطة تفتيش `gpt2` وإصدار Flash Attention 2 من النموذج باستخدام طول تسلسل يبلغ 512.

<div style="text-align: center">
<img src="https://huggingface.co/datasets/EduardoPacheco/documentation-images/resolve/main/gpt2_flash_attention_2_speedup.jpg">
</div>

## الموارد

فيما يلي قائمة بموارد Hugging Face الرسمية والمجتمعية (مشار إليها برمز 🌎) لمساعدتك في البدء باستخدام GPT2. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فالرجاء فتح طلب سحب وسنراجعه! يجب أن يوضح المورد بشكل مثالي شيء جديد بدلاً من تكرار مورد موجود.

- مدونة حول كيفية [ضبط نموذج GPT-2 غير الإنجليزي باستخدام Hugging Face](https://www.philschmid.de/fine-tune-a-non-english-gpt-2-model-with-huggingface).

- مدونة حول [كيفية توليد النص: استخدام طرق فك مختلفة لتوليد اللغة مع المحولات](https://huggingface.co/blog/how-to-generate) مع GPT-2.

- مدونة حول [تدريب CodeParrot 🦜 من الصفر](https://huggingface.co/blog/codeparrot)، وهو نموذج GPT-2 كبير.

- مدونة حول [توليد نص أسرع مع TensorFlow و XLA](https://huggingface.co/blog/tf-xla-generate) مع GPT-2.

- مدونة حول [كيفية تدريب نموذج لغة باستخدام Megatron-LM](https://huggingface.co/blog/megatron-training) مع نموذج GPT-2.

- دفتر ملاحظات حول كيفية [ضبط دقيق لـ GPT2 لتوليد كلمات الأغاني على غرار فنانك المفضل](https://colab.research.google.com/github/AlekseyKorshuk/huggingartists/blob/master/huggingartists-demo.ipynb). 🌎

- دفتر ملاحظات حول كيفية [ضبط دقيق لـ GPT2 لتوليد التغريدات على غرار مستخدم Twitter المفضل لديك](https://colab.research.google.com/github/borisdayma/huggingtweets/blob/master/huggingtweets-demo.ipynb). 🌎

- فصل [نمذجة اللغة السببية](https://huggingface.co/course/en/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch) من دورة 🤗 Hugging Face Course.

- [`GPT2LMHeadModel`] مدعوم بواسطة هذا [مثال على النص البرمجي لنمذجة اللغة السببية](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling)، [مثال على النص البرمجي لتوليد النص](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation)، و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).

- [`TFGPT2LMHeadModel`] مدعوم بواسطة هذا [مثال على النص البرمجي لنمذجة اللغة السببية](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).

- [`FlaxGPT2LMHeadModel`] مدعوم بواسطة هذا [مثال على النص البرمجي لنمذجة اللغة السببية](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#causal-language-modeling) و [دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/causal_language_modeling_flax.ipynb).

- دليل مهمة تصنيف النص [](../tasks/sequence_classification)

- دليل مهمة تصنيف الرموز [](../tasks/token_classification)

- دليل مهمة نمذجة اللغة السببية [](../tasks/language_modeling)

## GPT2Config

[[autodoc]] GPT2Config

## GPT2Tokenizer

[[autodoc]] GPT2Tokenizer

- save_vocabulary

## GPT2TokenizerFast

[[autodoc]] GPT2TokenizerFast

## مخرجات GPT2 المحددة

[[autodoc]] models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput

[[autodoc]] models.gpt2.modeling_tf_gpt2.TFGPT2DoubleHeadsModelOutput

<frameworkcontent>
<pt>

## GPT2Model

[[autodoc]] GPT2Model

- forward

## GPT2LMHeadModel

[[autodoc]] GPT2LMHeadModel

- forward

## GPT2DoubleHeadsModel

[[autodoc]] GPT2DoubleHeadsModel

- forward

## GPT2ForQuestionAnswering

[[autodoc]] GPT2ForQuestionAnswering

- forward

## GPT2ForSequenceClassification

[[autodoc]] GPT2ForSequenceClassification


- forward

## GPT2ForTokenClassification

[[autodoc]] GPT2ForTokenClassification


- forward

</pt>
<tf>

## TFGPT2Model

[[autodoc]] TFGPT2Model

- call

## TFGPT2LMHeadModel


[[autodoc]] TFGPT2LMHeadModel

- call

## TFGPT2DoubleHeadsModel

[[autodoc]] TFGPT2DoubleHeadsModel

- call

## TFGPT2ForSequenceClassification

[[autodoc]] TFGPT2ForSequenceClassification

- call

## TFSequenceClassifierOutputWithPast

[[autodoc]] modeling_tf_outputs.TFSequenceClassifierOutputWithPast

## TFGPT2Tokenizer

[[autodoc]] TFGPT2Tokenizer

</tf>
<jax>

## FlaxGPT2Model

[[autodoc]] FlaxGPT2Model

- __call__

## FlaxGPT2LMHeadModel

[[autodoc]] FlaxGPT2LMHeadModel

- __call__

</jax>
</frameworkcontent>