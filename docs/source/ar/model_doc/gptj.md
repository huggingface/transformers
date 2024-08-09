# GPT-J

## نظرة عامة
نموذج GPT-J هو نموذج لغوي احتمالي تم إصداره في مستودع [kingoflolz/mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax) بواسطة Ben Wang و Aran Komatsuzaki. وهو يشبه نموذج GPT-2 حيث تم تدريبه على مجموعة بيانات [the Pile](https://pile.eleuther.ai/) لإنتاج النصوص بشكل تلقائي.

تمت المساهمة بهذا النموذج من قبل [Stella Biderman](https://huggingface.co/stellaathena).

## نصائح الاستخدام
- لتحميل [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B) بتنسيق float32، ستحتاج إلى ضعف حجم الذاكرة العشوائية للنموذج، أي 1x لحجم الأوزان الأولية و 1x لتحميل نقطة التفتيش. لذلك، بالنسبة لـ GPT-J، ستكون هناك حاجة إلى 48 جيجابايت من الذاكرة العشوائية كحد أدنى لتحميل النموذج فقط. لتقليل استخدام الذاكرة العشوائية، هناك بعض الخيارات. يمكن استخدام وسيط `torch_dtype` لتهيئة النموذج بنصف الدقة على جهاز CUDA فقط. هناك أيضًا فرع fp16 والذي يخزن أوزان fp16، والذي يمكن استخدامه لتقليل استخدام الذاكرة العشوائية:

```python
>>> from transformers import GPTJForCausalLM
>>> import torch

>>> device = "cuda"
>>> model = GPTJForCausalLM.from_pretrained(
...     "EleutherAI/gpt-j-6B",
...     revision="float16",
...     torch_dtype=torch.float16,
... ).to(device)
```

- يجب أن يتناسب النموذج مع ذاكرة GPU بسعة 16 جيجابايت للتنفيذ. ولكن بالنسبة للتدريب/الضبط الدقيق، سيتطلب الأمر ذاكرة GPU أكبر بكثير. على سبيل المثال، يقوم مُحسِّن Adam بإنشاء أربع نسخ من النموذج: النموذج، والمشتقات، ومتوسط المشتقات، ومتوسط المربعات للمشتقات. لذلك، ستحتاج إلى ذاكرة GPU بحجم 4x على الأقل لحجم النموذج، حتى مع الدقة المختلطة، لأن تحديثات المشتقات تكون في تنسيق fp32. هذا لا يشمل المصفوفات والدفعات، والتي تتطلب أيضًا المزيد من ذاكرة GPU. لذلك، يجب استكشاف حلول مثل DeepSpeed لتدريب/ضبط النموذج. الخيار الآخر هو استخدام الكود الأصلي لتدريب/ضبط النموذج على وحدة معالجة الرسوميات (TPU)، ثم تحويل النموذج إلى تنسيق Transformers للتنفيذ. يمكن العثور على التعليمات الخاصة بذلك [هنا](https://github.com/kingoflolz/mesh-transformer-jax/blob/master/howto_finetune.md).

- على الرغم من أن مصفوفة التعليقات التوضيحية لها حجم 50400، إلا أن محدد GPT-2 يستخدم فقط 50257 إدخالًا. تم إضافة هذه الرموز الإضافية من أجل الكفاءة على وحدات معالجة الرسوميات (TPUs). ولتجنب عدم التطابق بين حجم مصفوفة التعليقات التوضيحية وحجم المفردات، يحتوي محدد الـ GPT-J على 143 رمزًا إضافيًا `<|extratoken_1|>... <|extratoken_143|>`، وبالتالي يصبح حجم مفردات المحلل 50400 أيضًا.

## أمثلة الاستخدام
يمكن استخدام طريقة [`~generation.GenerationMixin.generate`] لتوليد النصوص باستخدام نموذج GPT-J.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
>>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

>>> prompt = (
...     "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
...     "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
...     "researchers was the fact that the unicorns spoke perfect English."
... )

>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

>>> gen_tokens = model.generate(
...     input_ids,
...     do_sample=True,
...     temperature=0.9,
...     max_length=100,
... )
>>> gen_text = tokenizer.batch_decode(gen_tokens)[0]
```

أو باستخدام الدقة العائمة 16:

```python
>>> from transformers import GPTJForCausalLM, AutoTokenizer
>>> import torch

>>> device = "cuda"
>>> model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16).to(device)
>>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

>>> prompt = (
...     "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
...     "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
...     "researchers was the fact that the unicorns spoke perfect English."
... )

>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

>>> gen_tokens = model.generate(
...     input_ids,
...     do_sample=True,
...     temperature=0.9,
...     max_length=100,
... )
>>> gen_text = tokenizer.batch_decode(gen_tokens)[0]
```

## الموارد
فيما يلي قائمة بموارد Hugging Face الرسمية وموارد المجتمع (المشار إليها برمز 🌎) لمساعدتك في البدء في استخدام GPT-J. إذا كنت مهتمًا بتقديم مورد لإدراجه هنا، فيرجى فتح طلب سحب Pull Request وسنقوم بمراجعته! يجب أن يُظهر المورد بشكل مثالي شيئًا جديدًا بدلاً من تكرار مورد موجود.

- وصف [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6B).
- مدونة حول كيفية [نشر GPT-J 6B للتنفيذ باستخدام Hugging Face Transformers وAmazon SageMaker](https://huggingface.co/blog/gptj-sagemaker).
- مدونة حول كيفية [تسريع تنفيذ GPT-J باستخدام DeepSpeed-Inference على وحدات معالجة الرسوميات](https://www.philschmid.de/gptj-deepspeed-inference).
- منشور مدونة يقدم [GPT-J-6B: 6B JAX-Based Transformer](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/). 🌎
- دفتر ملاحظات لـ [GPT-J-6B Inference Demo](https://colab.research.google.com/github/kingoflolz/mesh-transformer-jax/blob/master/colab_demo.ipynb). 🌎
- دفتر ملاحظات آخر يوضح [Inference with GPT-J-6B](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/GPT-J-6B/Inference_with_GPT_J_6B.ipynb).
- فصل [نمذجة اللغة السببية](https://huggingface.co/course/en/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch) من دورة 🤗 Hugging Face Course.
- [`GPTJForCausalLM`] مدعوم بواسطة [مثال برمجة النماذج اللغوية السببية](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling)، و[مثال النص النصي](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation)، و[دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
- [`TFGPTJForCausalLM`] مدعوم بواسطة [مثال برمجة النماذج اللغوية السببية](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy) و[دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).
- [`FlaxGPTJForCausalLM`] مدعوم بواسطة [مثال برمجة النماذج اللغوية السببية](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#causal-language-modeling) و[دفتر الملاحظات](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/causal_language_modeling_flax.ipynb).

**موارد التوثيق**

- دليل مهام تصنيف النصوص [Text classification task guide](../tasks/sequence_classification)
- دليل مهام الإجابة على الأسئلة [Question answering task guide](../tasks/question_answering)
- دليل مهام نمذجة اللغة السببية [Causal language modeling task guide](../tasks/language_modeling)

## GPTJConfig

[[autodoc]] GPTJConfig

- all

<frameworkcontent>

<pt>

## GPTJModel

[[autodoc]] GPTJModel

- forward

## GPTJForCausalLM

[[autodoc]] GPTJForCausalLM

- forward

## GPTJForSequenceClassification

[[autodoc]] GPTJForSequenceClassification

- forward

## GPTJForQuestionAnswering

[[autodoc]] GPTJForQuestionAnswering

- forward

</pt>

<tf>

## TFGPTJModel

[[autodoc]] TFGPTJModel

- call

## TFGPTJForCausalLM

[[autodoc]] TFGPTJForCausalLM

- call

## TFGPTJForSequenceClassification

[[autodoc]] TFGPTJForSequenceClassification

- call

## TFGPTJForQuestionAnswering

[[autodoc]] TFGPTJForQuestionAnswering

- call

</tf>

<jax>

## FlaxGPTJModel

[[autodoc]] FlaxGPTJModel

- __call__

## FlaxGPTJForCausalLM

[[autodoc]] FlaxGPTJForCausalLM

- __call__

</jax>

</frameworkcontent>