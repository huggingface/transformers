# MPT

## نظرة عامة
اقترح فريق [MosaicML](https://www.mosaicml.com/) نموذج MPT وأطلقه بأحجام ومتغيرات دقيقة متعددة. ونماذج MPT عبارة عن سلسلة من نماذج اللغة المفتوحة المصدر والقابلة للاستخدام تجاريًا التي تم تدريبها مسبقًا على 1T من الرموز.

نماذج MPT هي محولات فك تشفير على غرار GPT مع العديد من التحسينات: تطبيقات الطبقة المُحسَّنة للأداء، وتغييرات معمارية توفر استقرارًا أكبر للتدريب، وإزالة حدود طول السياق عن طريق استبدال embeddings الموضعية بـ ALiBi.

- MPT base: نماذج MPT base التي تم تدريبها مسبقًا على التنبؤ بالرمز التالي
- MPT instruct: نماذج MPT base التي تمت معايرتها بدقة على مهام تعتمد على التعليمات
- MPT storywriter: نماذج MPT base التي تمت معايرتها بدقة لمدة 2500 خطوة على مقتطفات من 65000 رمز من كتب الخيال الموجودة في مجموعة كتب books3، مما يمكن النموذج من التعامل مع التسلسلات الطويلة جدًا

يمكن العثور على الكود الأصلي في مستودع [`llm-foundry`](https://github.com/mosaicml/llm-foundry/tree/main).

اقرأ المزيد عنه [في منشور المدونة](https://www.mosaicml.com/blog/mpt-7b)

## نصائح الاستخدام

- تعرف على بعض التقنيات وراء تدريب النموذج [في هذا القسم من مستودع llm-foundry](https://github.com/mosaicml/llm-foundry/blob/main/TUTORIAL.md#faqs)
- إذا كنت تريد استخدام الإصدار المتقدم من النموذج (نواة Triton، تكامل Flash attention المباشر)، فيمكنك استخدام تنفيذ النموذج الأصلي عن طريق إضافة `trust_remote_code=True` عند استدعاء `from_pretrained`.

## الموارد

- [دفتر ضبط الدقة](https://colab.research.google.com/drive/1HCpQkLL7UXW8xJUJJ29X7QAeNJKO0frZ?usp=sharing) حول كيفية ضبط نموذج MPT-7B بدقة على مثيل Google Colab مجاني لتحويل النموذج إلى دردشة آلية.

## MptConfig

[[autodoc]] MptConfig

- all

## MptModel

[[autodoc]] MptModel

- forward

## MptForCausalLM

[[autodoc]] MptForCausalLM

- forward

## MptForSequenceClassification

[[autodoc]] MptForSequenceClassification

- forward

## MptForTokenClassification

[[autodoc]] MptForTokenClassification

- forward

## MptForQuestionAnswering

[[autodoc]] MptForQuestionAnswering

- forward