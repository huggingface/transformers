# GPT Neo

## نظرة عامة
نموذج GPTNeo تم إصداره في مستودع [EleutherAI/gpt-neo](https://github.com/EleutherAI/gpt-neo) بواسطة سيد بلاك، وستيلا بيدرمان، وليو جاو، وفيل وانج، وكونور ليهي. وهو نموذج لغوي توجيهي مشابه لـ GPT2، تم تدريبه على مجموعة بيانات [Pile](https://pile.eleuther.ai/).

يتشابه التصميم المعماري لـ GPTNeo مع GPT2، باستثناء أن GPT Neo يستخدم الانتباه المحلي في كل طبقة أخرى بحجم نافذة يبلغ 256 رمزًا.

تمت المساهمة بهذا النموذج بواسطة [valhalla](https://huggingface.co/valhalla).

## مثال على الاستخدام
يمكن استخدام طريقة `generate()` لتوليد النص باستخدام نموذج GPT Neo.

```python
>>> from transformers import GPTNeoForCausalLM, GPT2Tokenizer

>>> model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
>>> tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

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

## الجمع بين GPT-Neo وFlash Attention 2

أولاً، تأكد من تثبيت أحدث إصدار من Flash Attention 2 لتضمين ميزة اهتمام نافذة الانزلاق، وتأكد من أن أجهزتك متوافقة مع Flash-Attention 2. تتوفر المزيد من التفاصيل [هنا](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2) حول التثبيت.

تأكد أيضًا من تحميل نموذجك بنصف الدقة (على سبيل المثال `torch.float16`).

لتحميل وتشغيل نموذج باستخدام Flash Attention 2، راجع المقطع أدناه:

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> device = "cuda" # the device to load the model onto

>>> model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
>>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

>>> prompt = "def hello_world():"

>>> model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
>>> model.to(device)

>>> generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"def hello_world():\n\t>>> run_script(\"hello.py\")\n\t>>> exit(0)\n<|endoftext|>"
```

### تسريع المتوقع

فيما يلي رسم بياني للتسريع المتوقع الذي يقارن وقت الاستنتاج النقي بين التنفيذ الأصلي في المحولات باستخدام نقطة تفتيش `EleutherAI/gpt-neo-2.7B` وإصدار Flash Attention 2 من النموذج.

لاحظ أنه بالنسبة لـ GPT-Neo، لا يمكن التدريب / التشغيل على سياق طويل جدًا حيث تكون القيمة القصوى [تضمين الموضع](https://huggingface.co/EleutherAI/gpt-neo-2.7B/blob/main/config.json#L58) محدودة بـ 2048 - ولكن هذا ينطبق على جميع النماذج gpt-neo وليس محددًا لـ FA-2.

<div style="text-align: center">
<img src="https://user-images.githubusercontent.com/49240599/272241893-b1c66b75-3a48-4265-bc47-688448568b3d.png">
</div>

## الموارد

- [دليل مهام تصنيف النصوص](../tasks/sequence_classification)
- [دليل مهام نمذجة اللغة السببية](../tasks/language_modeling)

## GPTNeoConfig

[[autodoc]] GPTNeoConfig

<frameworkcontent>
<pt>

## GPTNeoModel

[[autodoc]] GPTNeoModel

- forword

## GPTNeoForCausalLM

[[autodoc]] GPTNeoForCausalLM

- forword

## GPTNeoForQuestionAnswering

[[autodoc]] GPTNeoForQuestionAnswering

- forword

## GPTNeoForSequenceClassification

[[autodoc]] GPTNeoForSequenceClassification

- forword

## GPTNeoForTokenClassification

[[autodoc]] GPTNeoForTokenClassification

- forword

</pt>
<jax>

## FlaxGPTNeoModel

[[autodoc]] FlaxGPTNeoModel

- __call__

## FlaxGPTNeoForCausalLM

[[autodoc]] FlaxGPTNeoForCausalLM

- __call__

</jax>

</frameworkcontent>