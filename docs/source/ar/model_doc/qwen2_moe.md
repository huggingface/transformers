# Qwen2MoE

## نظرة عامة

Qwen2MoE هي سلسلة النماذج الجديدة لنماذج اللغة الكبيرة من فريق Qwen. في السابق، أصدرنا سلسلة Qwen، بما في ذلك Qwen-72B وQwen-1.8B وQwen-VL وQwen-Audio، وغيرها.

### تفاصيل النموذج

Qwen2MoE هي سلسلة نماذج لغة تتضمن نماذج فك تشفير لغة بأحجام نماذج مختلفة. بالنسبة لكل حجم، نقوم بإطلاق نموذج اللغة الأساسي ونموذج الدردشة المتوافق. تمتاز Qwen2MoE بالخيارات المعمارية التالية:

- تستند Qwen2MoE إلى بنية Transformer مع تنشيط SwiGLU، وتحيز QKV للاهتمام، واهتمام الاستعلام الجماعي، ومزيج من الاهتمام بنافذة الانزلاق والاهتمام الكامل، وما إلى ذلك. بالإضافة إلى ذلك، لدينا محسن محسن قابل للتكيف مع العديد من اللغات الطبيعية والأكواد.

- تستخدم Qwen2MoE بنية Mixture of Experts (MoE)، حيث يتم إعادة تدوير النماذج من نماذج اللغة الكثيفة. على سبيل المثال، "Qwen1.5-MoE-A2.7B" معاد تدويره من "Qwen-1.8B". يحتوي على 14.3 مليار معامل في الإجمالي و2.7 مليار معامل نشط أثناء وقت التشغيل، بينما يحقق أداءًا مماثلًا لـ "Qwen1.5-7B"، باستخدام 25% فقط من موارد التدريب.

للحصول على مزيد من التفاصيل، يرجى الرجوع إلى [منشور المدونة](https://qwenlm.github.io/blog/qwen-moe/).

## نصائح الاستخدام

يمكن العثور على "Qwen1.5-MoE-A2.7B" و"Qwen1.5-MoE-A2.7B-Chat" على [Huggingface Hub](https://huggingface.co/Qwen)

فيما يلي، نوضح كيفية استخدام "Qwen1.5-MoE-A2.7B-Chat" للاستنتاج. لاحظ أننا استخدمنا تنسيق ChatML للحوار، وفي هذا العرض التوضيحي، نوضح كيفية الاستفادة من "apply_chat_template" لهذا الغرض.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> device = "cuda" # الجهاز الذي سيتم تحميل النموذج عليه

>>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat")

>>> prompt = "قدم لي مقدمة موجزة عن نموذج اللغة الكبير."

>>> messages = [{"role": "user", "content": prompt}]

>>> text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

>>> model_inputs = tokenizer([text], return_tensors="pt").to(device)

>>> generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)

>>> generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

>>> response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## Qwen2MoeConfig

[[autodoc]] Qwen2MoeConfig

## Qwen2MoeModel

[[autodoc]] Qwen2MoeModel

- forward

## Qwen2MoeForCausalLM

[[autodoc]] Qwen2MoeForCausalLM

- forward

## Qwen2MoeForSequenceClassification

[[autodoc]] Qwen2MoeForSequenceClassification

- forward

## Qwen2MoeForTokenClassification

[[autodoc]] Qwen2MoeForTokenClassification

- forward