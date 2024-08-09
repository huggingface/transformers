# MADLAD-400

## نظرة عامة

تم إصدار نماذج MADLAD-400 في الورقة [MADLAD-400: مجموعة بيانات كبيرة ومتعددة اللغات وعلى مستوى المستندات تمت مراجعتها يدويًا](MADLAD-400: A Multilingual And Document-Level Large Audited Dataset).

وفيما يلي الملخص من الورقة:

> "نقدم MADLAD-400، وهي مجموعة بيانات أحادية اللغة يدوية المراجعة وعامة المجال و3T رمزية تستند إلى CommonCrawl، وتغطي 419 لغة. نناقش القيود التي كشفت عنها المراجعة الذاتية لـ MADLAD-400، ودور مراجعة البيانات في عملية إنشاء مجموعة البيانات. بعد ذلك، نقوم بتدريب وإطلاق نموذج ترجمة آلية متعدد اللغات بحجم 10.7 مليار معلمة على 250 مليار رمز تغطي أكثر من 450 لغة باستخدام البيانات المتاحة للجمهور، ونجد أنه قادر على المنافسة مع النماذج الأكبر حجمًا بشكل ملحوظ، ونقدم النتائج على مجالات مختلفة. بالإضافة إلى ذلك، نقوم بتدريب نموذج لغة بحجم 8 مليارات معلمة، وتقييم النتائج على الترجمة القليلة التصوير. نجعل النماذج الأساسية 1 متاحة لمجتمع الباحثين."

أضاف هذا النموذج [Juarez Bochi](https://huggingface.co/jbochi). يمكن العثور على نقاط التفتيش الأصلية [هنا](https://github.com/google-research/google-research/tree/master/madlad_400).

هذا نموذج ترجمة آلية يدعم العديد من اللغات منخفضة الموارد، وهو قادر على المنافسة مع النماذج الأكبر حجمًا بشكل ملحوظ.

يمكن استخدام أوزان MADLAD-400 مباشرة دون ضبط دقيق للنموذج:

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/madlad400-3b-mt")
>>> tokenizer = AutoTokenizer.from_pretrained("google/madlad400-3b-mt")

>>> inputs = tokenizer("<2pt> I love pizza!", return_tensors="pt")
>>> outputs = model.generate(**inputs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['Eu amo pizza!']
```

أطلقت Google المتغيرات التالية:

- [google/madlad400-3b-mt](https://huggingface.co/google/madlad400-3b-mt)
- [google/madlad400-7b-mt](https://huggingface.co/google/madlad400-7b-mt)
- [google/madlad400-7b-mt-bt](https://huggingface.co/google/madlad400-7b-mt-bt)
- [google/madlad400-10b-mt](https://huggingface.co/google/madlad400-10b-mt)

يمكن العثور على نقاط التفتيش الأصلية [هنا](https://github.com/google-research/google-research/tree/master/madlad_400).

<Tip>

راجع [صفحة وثائق T5](t5) لجميع المراجع API وأمثلة التعليمات البرمجية ومفكرات الدفاتر. لمزيد من التفاصيل حول تدريب وتقييم MADLAD-400، راجع بطاقة النموذج.

</Tip>