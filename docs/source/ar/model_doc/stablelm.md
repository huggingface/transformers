# StableLM

## نظرة عامة
`StableLM 3B 4E1T` هو نموذج مقترح في ["StableLM 3B 4E1T": تقرير فني](https://stability.wandb.io/stability-llm/stable-lm/reports/StableLM-3B-4E1T--VmlldzoyMjU4?accessToken=u3zujipenkx5g7rtcj9qojjgxpconyjktjkli2po09nffrffdhhchq045vp0wyfo) بواسطة Stability AI وهو أول نموذج في سلسلة من النماذج اللغوية متعددة الفترات التدريبية.

### تفاصيل النموذج
`StableLM 3B 4E1T` هو نموذج لغة أساسي يعتمد على فك التشفير فقط، تم تدريبه مسبقًا على تريليون رمز من مجموعات بيانات متنوعة باللغة الإنجليزية والرمز لأربع فترات.

تعتمد بنية النموذج على محول مع تضمين موضع دوار جزئي، وتنشيط SwiGLU، وLayerNorm، وما إلى ذلك.

نقدم أيضًا `StableLM Zephyr 3B`، وهو إصدار تمت تهيئته تعليميًا من النموذج ويمكن استخدامه لتطبيقات الدردشة.

### نصائح الاستخدام
- البنية مماثلة لـ LLaMA ولكن مع تطبيق RoPE على 25% من أبعاد تضمين الرأس، وLayerNorm بدلاً من RMSNorm، وشروط انحياز QKV الاختيارية.
- تستخدم النماذج المستندة إلى `StableLM 3B 4E1T` نفس المحلل اللغوي كما في [`GPTNeoXTokenizerFast`].

يمكن العثور على `StableLM 3B 4E1T` و`StableLM Zephyr 3B` على [Huggingface Hub](https://huggingface.co/stabilityai)

توضح مقتطفات الشفرة التالية كيفية استخدام `StableLM 3B 4E1T` للاستنتاج:

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
>>> device = "cuda" # الجهاز لتحميل النموذج عليه

>>> set_seed(0)

>>> tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
>>> model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-3b-4e1t")
>>> model.to(device)  # doctest: +IGNORE_RESULT

>>> model_inputs = tokenizer("The weather is always wonderful in", return_tensors="pt").to(model.device)

>>> generated_ids = model.generate(**model_inputs, max_length=32, do_sample=True)
>>> responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
>>> responses
['The weather is always wonderful in Costa Rica, which makes it a prime destination for retirees. That’s where the Pensionado program comes in, offering']
```

## الجمع بين StableLM وFlash Attention 2
أولاً، تأكد من تثبيت أحدث إصدار من Flash Attention v2.

```bash
pip install -U flash-attn --no-build-isolation
```

تأكد أيضًا من أن أجهزتك متوافقة مع Flash-Attention 2. اقرأ المزيد عنها في الوثائق الرسمية لمستودع [`flash-attn`](https://github.com/Dao-AILab/flash-attention). ملاحظة: يجب تحميل نموذجك في نصف الدقة (على سبيل المثال `torch.bfloat16`).

الآن، لتشغيل النموذج مع Flash Attention 2، راجع المقتطف أدناه:

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
>>> device = "cuda" # الجهاز لتحميل النموذج عليه

>>> set_seed(0)

>>> tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
>>> model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-3b-4e1t", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")  # doctest: +SKIP
>>> model.to(device)  # doctest: +SKIP

>>> model_inputs = tokenizer("The weather is always wonderful in", return_tensors="pt").to(model.device)

>>> generated_ids = model.generate(**model_inputs, max_length=32, do_sample=True)  # docticast: +SKIP
>>> responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)  # doctest: +SKIP
>>> responses  # doctest: +SKIP
['The weather is always wonderful in Costa Rica, which makes it a prime destination for retirees. That’s where the Pensionado program comes in, offering']
```

## StableLmConfig
[[autodoc]] StableLmConfig

## StableLmModel
[[autodoc]] StableLmModel
- forward

## StableLmForCausalLM
[[autodoc]] StableLmForCausalLM
- forward

## StableLmForSequenceClassification
[[autodoc]] StableLmForSequenceClassification
- forward

## StableLmForTokenClassification
[[autodoc]] StableLmForTokenClassification
- forward