# PhoBERT

## نظرة عامة
تم اقتراح نموذج PhoBERT في [PhoBERT: نماذج اللغة المُدربة مسبقًا للفيتنامية](https://www.aclweb.org/anthology/2020.findings-emnlp.92.pdf) بواسطة دات كيو نجوين، وآن توان نجوين.

المُلخص من الورقة البحثية هو التالي:

*نُقدم PhoBERT بإصدارين، PhoBERT-base وPhoBERT-large، أول نماذج اللغة أحادية اللغة كبيرة الحجم والمُدربة مسبقًا والمُتاحة للعامة للفيتنامية. تُظهر النتائج التجريبية أن PhoBERT يتفوق باستمرار على أفضل نموذج مُدرب مسبقًا مُتعدد اللغات XLM-R (Conneau et al.)، ويحسن الحالة الراهنة في العديد من مهام معالجة اللغات الطبيعية الخاصة باللغة الفيتنامية بما في ذلك وسم أجزاء الكلام، وتحليل الإعراب، وتعريف الكيانات المُسماة، والاستدلال اللغوي الطبيعي.*

تمت المساهمة بهذا النموذج بواسطة [dqnguyen](https://huggingface.co/dqnguyen). يمكن العثور على الكود الأصلي [هنا](https://github.com/VinAIResearch/PhoBERT).

## مثال على الاستخدام

```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> phobert = AutoModel.from_pretrained("vinai/phobert-base")
>>> tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

>>> # يجب أن يكون نص الإدخال مُجزأً مسبقًا إلى كلمات!
>>> line = "Tôi là sinh_viên trường đại_học Công_nghệ ."

>>> input_ids = torch.tensor([tokenizer.encode(line)])

>>> with torch.no_grad():
...     features = phobert(input_ids) # مُخرجات النماذج الآن عبارة عن توابع

>>> # باستخدام TensorFlow 2.0+:
>>> # from transformers import TFAutoModel
>>> # phobert = TFAutoModel.from_pretrained("vinai/phobert-base")
```

<Tip>

يتم تنفيذ PhoBERT بنفس طريقة BERT، باستثناء عملية التجزئة إلى رموز. يُرجى الرجوع إلى [توثيق EART](bert) للحصول على معلومات حول فئات التهيئة ومعاييرها. يتم توثيق مُجزئ الرموز المُخصص لـ PhoBERT أدناه.

</Tip>

## PhobertTokenizer

[[autodoc]] PhobertTokenizer