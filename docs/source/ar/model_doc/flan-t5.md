# FLAN-T5

## نظرة عامة
تم إصدار FLAN-T5 في الورقة البحثية [Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416.pdf) - وهي نسخة محسنة من T5 تم ضبطها بشكل دقيق في مزيج من المهام.

يمكن استخدام أوزان FLAN-T5 مباشرة دون ضبط نموذج الدقة:

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

>>> inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
>>> outputs = model.generate(**inputs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['Pour a cup of bolognese into a large bowl and add the pasta']
```

يتضمن FLAN-T5 التحسينات نفسها الموجودة في الإصدار T5 1.1 (راجع [هنا](https://huggingface.co/docs/transformers/model_doc/t5v1.1) للاطلاع على التفاصيل الكاملة لتحسينات النموذج).

أصدرت Google المتغيرات التالية:

- [google/flan-t5-small](https://huggingface.co/google/flan-t5-small)
- [google/flan-t5-base](https://huggingface.co/google/flan-t5-base)
- [google/flan-t5-large](https://huggingface.co/google/flan-t5-large)
- [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl)
- [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl).

يمكن العثور على نقاط التفتيش الأصلية [هنا](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints).

<Tip>
راجع [صفحة وثائق T5](t5) للحصول على جميع مراجع API وأمثلة التعليمات البرمجية ومفكرات Jupyter. لمزيد من التفاصيل حول تدريب وتقييم FLAN-T5، راجع بطاقة النموذج.
</Tip>