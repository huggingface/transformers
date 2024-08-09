# RetriBERT

**ملاحظة:**  *هذا النموذج في وضع الصيانة فقط، لذلك لن نقبل أي طلبات سحب (Pull Requests) جديدة لتغيير شفرته. إذا واجهتك أي مشكلات أثناء تشغيل هذا النموذج، يرجى إعادة تثبيت الإصدار الأخير الذي يدعم هذا النموذج: v4.30.0. يمكنك القيام بذلك عن طريق تشغيل الأمر التالي: `pip install -U transformers==4.30.0`.*

## نظرة عامة
اقتُرح نموذج RetriBERT في التدوينة التالية: [Explain Anything Like I'm Five: A Model for Open Domain Long Form Question Answering](https://yjernite.github.io/lfqa.html) (شرح أي شيء كما لو كان لطفل في الخامسة: نموذج للإجابة على الأسئلة المفتوحة النطاق طويلة النص). RetriBERT هو نموذج صغير يستخدم إما مشفر BERT واحدًا أو زوجًا من مشفرات BERT مع إسقاط الأبعاد المنخفضة للفهرسة الدلالية الكثيفة للنص.

ساهم بهذا النموذج [yjernite](https://huggingface.co/yjernite). يمكن العثور على الشفرة البرمجية لتدريب النموذج واستخدامه [هنا](https://github.com/huggingface/transformers/tree/main/examples/research-projects/distillation).

## RetriBertConfig

[[autodoc]] RetriBertConfig

## RetriBertTokenizer

[[autodoc]] RetriBertTokenizer

## RetriBertTokenizerFast

[[autodoc]] RetriBertTokenizerFast

## RetriBertModel

[[autodoc]] RetriBertModel

- forward