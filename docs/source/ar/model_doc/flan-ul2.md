# FLAN-UL2

## نظرة عامة

Flan-UL2 هو نموذج ترميز وفك ترميز يعتمد على بنية T5. يستخدم نفس تكوين نموذج [UL2](ul2) الذي تم إصداره في وقت سابق من العام الماضي.

تم ضبطه بدقة باستخدام ضبط "Flan" وجمع البيانات. وعلى غرار `Flan-T5`، يمكن استخدام أوزان FLAN-UL2 مباشرة دون ضبط دقيق للنموذج:

وفقًا للمدونة الأصلية، فيما يلي التحسينات الملحوظة:

- تم تدريب نموذج UL2 الأصلي فقط باستخدام حقل استقبال يبلغ 512، مما جعله غير مثالي لـ N-shot prompting حيث N كبير.
- تستخدم نقطة تفتيش Flan-UL2 حقل استقبال يبلغ 2048 مما يجعلها أكثر ملاءمة للتعلم القائم على السياق القليل.
- كان لنموذج UL2 الأصلي أيضًا رموز تبديل الوضع التي كانت إلزامية إلى حد ما للحصول على أداء جيد. ومع ذلك، كانت مزعجة بعض الشيء لأنها تتطلب في كثير من الأحيان بعض التغييرات أثناء الاستدلال أو الضبط الدقيق. في هذا التحديث/التغيير، نواصل تدريب UL2 20B لمدة 100 ألف خطوة إضافية (بمجموعة صغيرة) لنسيان "رموز الوضع" قبل تطبيق ضبط تعليمات Flan. لا تتطلب نقطة تفتيش Flan-UL2 هذه رموز الوضع بعد الآن.

أصدرت Google المتغيرات التالية:

يمكن العثور على نقاط التفتيش الأصلية [هنا](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-ul2-checkpoints).

## التشغيل على أجهزة ذات موارد منخفضة

النموذج ثقيل للغاية (~40 جيجابايت في نصف الدقة)، لذا إذا كنت تريد تشغيل النموذج فقط، فتأكد من تحميل نموذجك في 8 بت، واستخدم `device_map="auto"` للتأكد من عدم وجود أي مشكلة في الذاكرة!

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-ul2", load_in_8bit=True, device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

>>> inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
>>> outputs = model.generate(**inputs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['In a large skillet, brown the ground beef and onion over medium heat. Add the garlic']
```

<Tip>

راجع صفحة وثائق T5 للرجوع إلى API والنصائح وأمثلة التعليمات البرمجية ومفكرات Jupyter.

</Tip>