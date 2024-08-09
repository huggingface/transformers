# BERTweet

## نظرة عامة

اقترح نموذج BERTweet في ورقة بحثية بعنوان "BERTweet: A pre-trained language model for English Tweets" من قبل Dat Quoc Nguyen وThanh Vu وAnh Tuan Nguyen.

ملخص الورقة البحثية هو كما يلي:

*نحن نقدم BERTweet، وهو أول نموذج لغة كبير مُدرب مسبقًا ومُتاح للجمهور للتغريدات باللغة الإنجليزية. يتم تدريب نموذج BERTweet الخاص بنا، والذي يحتوي على نفس البنية المعمارية مثل BERT-base (Devlin et al.، 2019)، باستخدام إجراء التدريب المسبق RoBERTa (Liu et al.، 2019). تُظهر التجارب أن BERTweet يتفوق على خطي الأساس القويين RoBERTa-base وXLM-R-base (Conneau et al.، 2020)، مما يحقق نتائج أداء أفضل من النماذج السابقة المتقدمة في ثلاث مهام لمعالجة اللغات الطبيعية في التغريدات: تحليل القواعد، وتعريف الكيانات المسماة، وتصنيف النصوص.*

تمت المساهمة بهذا النموذج من قبل [dqnguyen](https://huggingface.co/dqnguyen). يمكن العثور على الكود الأصلي [هنا](https://github.com/VinAIResearch/BERTweet).

## مثال على الاستخدام

```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> bertweet = AutoModel.from_pretrained("vinai/bertweet-base")

>>> # For transformers v4.x+:
>>> tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

>>> # For transformers v3.x:
>>> # tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

>>> # تغريدة الإدخال تم تطبيعها بالفعل!
>>> line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:"

>>> input_ids = torch.tensor([tokenizer.encode(line)])

>>> with torch.no_grad():
...     features = bertweet(input_ids) # مخرجات النماذج الآن عبارة عن توبلات

>>> # باستخدام TensorFlow 2.0+:
>>> # from transformers import TFAutoModel
>>> # bertweet = TFAutoModel.from_pretrained("vinai/bertweet-base")
```

<Tip>

هذا التنفيذ مطابق لـ BERT، باستثناء طريقة التعامل مع الرموز. راجع [توثيق BERT](bert) للحصول على معلومات مرجعية حول واجهة برمجة التطبيقات.

</Tip>

## BertweetTokenizer

[[autodoc]] BertweetTokenizer