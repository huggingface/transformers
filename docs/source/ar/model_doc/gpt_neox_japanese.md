# GPT-NeoX-Japanese

## نظرة عامة

نحن نقدم GPT-NeoX-Japanese، وهو نموذج لغة مولّد للغة اليابانية، تم تدريبه بناءً على [https://github.com/EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox).

تتميز اللغة اليابانية بأنها لغة فريدة من نوعها، حيث تمتلك مفردات كبيرة ومزيجًا من أنماط الكتابة هيراغانا، وكاتاكانا، وكانجي.

لمعالجة هذا البناء المميز للغة اليابانية، نستخدم [محللًا نحويًا خاصًا للكلمات الفرعية](https://github.com/tanreinama/Japanese-BPEEncoder_V2). ونحن ممتنون للغاية لـ *tanreinama* على إتاحته لمحلل الكلمات هذا مفتوح المصدر، والذي كان مفيدًا للغاية.

وبناءً على التوصيات المنبثقة عن بحث Google حول [PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html)، قمنا بإزالة معلمات التحيز من كتل المحول، مما أدى إلى تحسين أداء النموذج. يرجى الرجوع إلى [هذه المقالة](https://medium.com/ml-abeja/training-a-better-gpt-2-93b157662ae4) لمزيد من التفاصيل.

تمت قيادة عملية تطوير النموذج من قبل [شينيا أوتاني](https://github.com/SO0529)، و[تاكايوشي ماكابي](https://github.com/spider-man-tm)، و[أنوج أرورا](https://github.com/Anuj040)، و[كيو هاتوري](https://github.com/go5paopao) من [شركة ABEJA](https://www.abejainc.com/). لمزيد من المعلومات حول نشاط بناء النموذج هذا، يرجى الرجوع إلى [هنا (ja)](https://tech-blog.abeja.asia/entry/abeja-gpt-project-202207).

### مثال على الاستخدام

يمكن استخدام الدالة `generate()` لتوليد نص باستخدام نموذج GPT NeoX Japanese.

```python
>>> from transformers import GPTNeoXJapaneseForCausalLM, GPTNeoXJapaneseTokenizer

>>> model = GPTNeoXJapaneseForCausalLM.from_pretrained("abeja/gpt-neox-japanese-2.7b")
>>> tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")

>>> prompt = "人とAIが協調するためには、"

>>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

>>> gen_tokens = model.generate(
...     input_ids,
...     do_sample=True,
...     temperature=0.9,
...     max_length=100,
... )
>>> gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

>>> print(gen_text)
AI との共存を実現し、AI を正しく理解することが必要です。
```

## الموارد

- [دليل مهمة نمذجة اللغة السببية](../tasks/language_modeling)

## GPTNeoXJapaneseConfig

[[autodoc]] GPTNeoXJapaneseConfig

## GPTNeoXJapaneseTokenizer

[[autodoc]] GPTNeoXJapaneseTokenizer

## GPTNeoXJapaneseModel

[[autodoc]] GPTNeoXJapaneseModel

- forward

## GPTNeoXJapaneseForCausalLM

[[autodoc]] GPTNeoXJapaneseForCausalLM

- forward