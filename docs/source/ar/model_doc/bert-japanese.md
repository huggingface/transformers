# BertJapanese
## نظرة عامة
نموذج BERT الذي تم تدريبه على النص الياباني.

هناك نموذجان بأسلوبي توكينيزيشن مختلفين:

- توكينيزيشن باستخدام MeCab وWordPiece. يتطلب هذا بعض الاعتماديات الإضافية، [fugashi](https://github.com/polm/fugashi) وهو عبارة عن ملف تعريف ارتباط حول [MeCab](https://taku910.github.io/mecab/).

- توكينيزيشن إلى أحرف.

لاستخدام *MecabTokenizer*، يجب عليك تثبيت `pip install transformers ["ja"]` (أو `pip install -e . ["ja"]` إذا كنت تقوم بالتثبيت من المصدر) لتثبيت الاعتماديات.

راجع [التفاصيل على مستودع cl-tohoku](https://github.com/cl-tohoku/bert-japanese).

مثال على استخدام نموذج مع توكينيزيشن MeCab وWordPiece:

```python
>>> import torch
>>> from transformers import AutoModel, AutoTokenizer

>>> bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
>>> tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

>>> ## إدخال النص الياباني
>>> line = "吾輩は猫である。"

>>> inputs = tokenizer(line, return_tensors="pt")

>>> print(tokenizer.decode(inputs["input_ids"][0]))
[CLS] 吾輩 は 猫 で ある 。 [SEP]

>>> outputs = bertjapanese(**inputs)
```

مثال على استخدام نموذج مع توكينيزيشن الأحرف:

```python
>>> bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese-char")
>>> tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")

>>> ## إدخال النص الياباني
>>> line = "吾輩は猫である。"

>>> inputs = tokenizer(line, return_tensors="pt")

>>> print(tokenizer.decode(inputs["input_ids"][0]))
[CLS] 吾 輩 は 猫 で あ る 。 [SEP]

>>> outputs = bertjapanese(**inputs)
```

تمت المساهمة بهذا النموذج من قبل [cl-tohoku](https://huggingface.co/cl-tohoku).

<Tip>

هذا التنفيذ هو نفسه BERT، باستثناء طريقة التوكنيزيشن. راجع [توثيق BERT](bert) للحصول على معلومات مرجعية حول واجهة برمجة التطبيقات.

</Tip>

## BertJapaneseTokenizer

[[autodoc]] BertJapaneseTokenizer