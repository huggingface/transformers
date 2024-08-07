# Qwen2

## نظرة عامة

Qwen2 هي سلسلة النماذج اللغوية الجديدة للنماذج اللغوية الضخمة من فريق Qwen. في السابق، أصدرنا سلسلة Qwen، بما في ذلك Qwen-72B وQwen-1.8B وQwen-VL وQwen-Audio، وغيرها.

### تفاصيل النموذج

Qwen2 هي سلسلة من نماذج اللغة التي تشمل نماذج فك تشفير اللغة بأحجام نماذج مختلفة. بالنسبة لكل حجم، نقوم بإطلاق نموذج اللغة الأساسي ونموذج الدردشة المتوافق. إنه يعتمد على بنية Transformer مع تنشيط SwiGLU، وتحيز QKV للاهتمام، واهتمام الاستعلام الجماعي، ومزيج من انتباه النافذة المنزلقة والاهتمام الكامل، وما إلى ذلك. بالإضافة إلى ذلك، لدينا محسن محلل نحوي متكيف مع عدة لغات ورمز طبيعي.

## نصائح الاستخدام

يمكن العثور على `Qwen2-7B-beta` و`Qwen2-7B-Chat-beta` على [Huggingface Hub](https://huggingface.co/Qwen)

فيما يلي، نوضح كيفية استخدام `Qwen2-7B-Chat-beta` للاستنتاج. لاحظ أننا استخدمنا تنسيق ChatML للحوار، وفي هذا العرض التوضيحي، نوضح كيفية الاستفادة من `apply_chat_template` لهذا الغرض.

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> device = "cuda" # الجهاز الذي سيتم تحميل النموذج عليه

>>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B-Chat", device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")

>>> prompt = "قدم لي مقدمة موجزة عن النموذج اللغوي الضخم."

>>> messages = [{"role": "user", "content": prompt}]

>>> text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

>>> model_inputs = tokenizer([text], return_tensors="pt").to(device)

>>> generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)

>>> generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

>>> response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## Qwen2Config

[[autodoc]] Qwen2Config

## Qwen2Tokenizer

[[autodoc]] Qwen2Tokenizer

- save_vocabulary

## Qwen2TokenizerFast

[[autodoc]] Qwen2TokenizerFast

## Qwen2Model

[[autodoc]] Qwen2Model

- forward

## Qwen2ForCausalLM

[[autodoc]] Qwen2ForCausalLM

- forward

## Qwen2ForSequenceClassification

[[autodoc]] Qwen2ForSequenceClassification

- forward

## Qwen2ForTokenClassification

[[autodoc]] Qwen2ForTokenClassification

- forward